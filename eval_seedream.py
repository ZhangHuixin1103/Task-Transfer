import argparse
import base64
import json
import logging
import os
import shutil
import sys
from io import BytesIO
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

from eval import hashed_id, generate_text_prompt, evaluate_generated, eval_quality

viescore_path = '/data1/tzz/huixin/Task-Transfer/VIEScore'
if viescore_path not in sys.path:
    sys.path.append(viescore_path)

DATA_TASKS_DIR = "data/tasks"
EVAL_DATASET_JSON = "data/dataset/eval_dataset.json"
OUTPUT_DIR = "data/output/baseline/seedream/output_qwen"
TMP_DIR = "data/tmp/tmp_seedream"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

BASE_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
CHECKPOINT_PATH = "Qwen3-VL/qwen-vl-finetune/output/checkpoint-4875"

FAL_KEY = "insert-your-fal-api-key-here"  # Create a fal API key and set it as an environment variable or directly here
UPLOAD_CACHE_JSON = "data/output/baseline/seedream/fal_upload_cache.json"
SEEDREAM_ENDPOINT = "fal-ai/bytedance/seedream/v4/edit"

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def on_queue_update(update):
    import fal_client

    if isinstance(update, fal_client.InProgress):
        for log in update.logs or []:
            print(log["message"])


def load_upload_cache():
    if os.path.exists(UPLOAD_CACHE_JSON):
        try:
            with open(UPLOAD_CACHE_JSON, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_upload_cache(cache):
    with open(UPLOAD_CACHE_JSON, 'w') as f:
        json.dump(cache, f, indent=4)


def upload_image_to_fal(path, cache):
    import fal_client

    abs_path = os.path.abspath(path)
    stat = os.stat(abs_path)
    cache_key = f"{abs_path}|{stat.st_size}|{int(stat.st_mtime)}"

    if cache_key in cache:
        return cache[cache_key]

    logging.info(f"Uploading to fal CDN: {abs_path}")
    url = fal_client.upload_file(abs_path)
    cache[cache_key] = url
    save_upload_cache(cache)
    return url


def _decode_data_uri(data_uri):
    header, encoded = data_uri.split(",", 1)
    raw = base64.b64decode(encoded)
    return Image.open(BytesIO(raw)).convert("RGB")


def download_seedream_image(image_obj):
    if isinstance(image_obj, str):
        url = image_obj
    else:
        url = image_obj.get("url") or image_obj.get("content") or image_obj.get("data")

    if not url:
        raise ValueError(f"Seedream image output has no URL/data field: {image_obj}")

    if url.startswith("data:image/"):
        return _decode_data_uri(url)

    parsed = urlparse(url)
    if not parsed.scheme.startswith("http"):
        raise ValueError(f"Unsupported Seedream image URL: {url}")

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def generate_image_seedream(taskA_in, taskA_out, taskB_in, text_prompt, args, upload_cache, attempt_idx=0):
    import fal_client

    taskA_in_path = os.path.join(DATA_TASKS_DIR, taskA_in)
    taskA_out_path = os.path.join(DATA_TASKS_DIR, taskA_out)
    taskB_in_path = os.path.join(DATA_TASKS_DIR, taskB_in)

    with Image.open(taskB_in_path) as taskB_img:
        target_size = taskB_img.size

    image_urls = [
        upload_image_to_fal(taskA_in_path, upload_cache),
        upload_image_to_fal(taskA_out_path, upload_cache),
        upload_image_to_fal(taskB_in_path, upload_cache),
    ]

    arguments = {
        "prompt": text_prompt,
        "image_size": {
            "width": target_size[0],
            "height": target_size[1],
        },
        "num_images": 1,
        "max_images": 1,
        "sync_mode": args.sync_mode,
        "enable_safety_checker": args.enable_safety_checker,
        "enhance_prompt_mode": args.enhance_prompt_mode,
        "image_urls": image_urls,
    }
    if args.seed is not None:
        arguments["seed"] = args.seed + attempt_idx

    try:
        result = fal_client.subscribe(
            SEEDREAM_ENDPOINT,
            arguments=arguments,
            with_logs=args.with_logs,
            on_queue_update=on_queue_update if args.with_logs else None,
        )
        images = result.get("images", [])
        if not images:
            logging.error(f"Seedream returned no images: {result}")
            return None

        image = download_seedream_image(images[0])
        if args.match_taskb_size and image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        logging.error(f"Seedream generation failed: {e}")
        return None


def run_evaluation(args):
    os.environ["FAL_KEY"] = FAL_KEY
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError(
            "FAL_KEY is not set. Create a fal API key and run: export FAL_KEY='your-api-key'"
        )

    with open(EVAL_DATASET_JSON, 'r') as f:
        eval_data = json.load(f)

    grouped = {}
    for entry in eval_data:
        taskA = entry['taskA_input'].split('/')[0]
        taskB = entry['taskB_input'].split('/')[0]
        pair_key = f"{taskA}__{taskB}"
        grouped.setdefault(pair_key, []).append(entry)

    final_results = {}
    upload_cache = load_upload_cache()

    if args.use_qwen_for_prompt:
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        logging.info("Loading Qwen for prompt enhancement...")
        if os.path.exists(os.path.join(CHECKPOINT_PATH, "adapter_config.json")):
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_PATH, torch_dtype="auto", device_map="auto"
            )
            prompt_qwen_model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
            prompt_qwen_model = prompt_qwen_model.merge_and_unload()
        else:
            prompt_qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                CHECKPOINT_PATH, torch_dtype="auto", device_map="auto"
            )
        prompt_qwen_model.eval()
        prompt_qwen_processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    else:
        prompt_qwen_model = None
        prompt_qwen_processor = None

    for pair_key, entries in grouped.items():
        logging.info(f"Processing pair: {pair_key}")
        pair_res_dir = os.path.join(OUTPUT_DIR, pair_key)
        os.makedirs(pair_res_dir, exist_ok=True)
        log_path = os.path.join(pair_res_dir, "evaluation_log.jsonl")

        existing_combo_ids = set()
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    try:
                        existing_combo_ids.add(json.loads(line)['combo_id'])
                    except:
                        continue

        with open(log_path, 'a') as log_file:
            for entry in entries[:args.max_samples]:
                taskA_in = entry['taskA_input']
                taskA_out = entry['taskA_output']
                taskB_in = entry['taskB_input']
                taskB_out = entry['taskB_output']

                combo_id = hashed_id(taskA_in, taskB_in)
                final_path = os.path.join(pair_res_dir, f"{combo_id}.png")

                if os.path.exists(final_path):
                    if combo_id in existing_combo_ids:
                        logging.info(f"COMPLETE: Skipping combo {combo_id}, image and metrics already exist.")
                        continue
                    else:
                        logging.info(f"RESUMING: Found image for {combo_id}, calculating and logging metrics...")
                        try:
                            psnr, ssim, viescore = evaluate_generated(
                                os.path.join(DATA_TASKS_DIR, taskB_out), final_path,
                                taskA_in, taskA_out, taskB_in,
                                pair_key.split('__')[0], pair_key.split('__')[1]
                            )
                            log_entry = {"combo_id": combo_id, "final_image": final_path,
                                         "psnr": psnr, "ssim": ssim, "viescore": viescore}
                            log_file.write(json.dumps(log_entry) + '\n')
                            log_file.flush()
                            os.fsync(log_file.fileno())
                            logging.info(f"SUCCESS: Logged metrics for existing image {combo_id}.")
                        except Exception as e:
                            logging.error(f"FAILURE: Could not evaluate existing image {final_path}. Error: {e}")
                        continue

                logging.info(f"STARTING: Processing new combo {combo_id}.")
                combo_tmp_dir = os.path.join(TMP_DIR, pair_key, combo_id)
                os.makedirs(combo_tmp_dir, exist_ok=True)

                text_prompt = generate_text_prompt(
                    taskA_in, taskA_out, taskB_in,
                    model=prompt_qwen_model,
                    processor=prompt_qwen_processor,
                    use_qwen=args.use_qwen_for_prompt,
                    fixed_prompt=args.fixed_prompt
                )

                best_psnr = -np.inf
                best_gen_path = None
                taskB_gt_path = os.path.join(DATA_TASKS_DIR, taskB_out)

                for i in range(args.num_tries):
                    gen_image = generate_image_seedream(
                        taskA_in, taskA_out, taskB_in, text_prompt, args, upload_cache,
                        attempt_idx=i
                    )
                    if gen_image:
                        logging.info(f"Attempt {i+1}: Successfully received an image from Seedream.")
                        temp_path = os.path.join(combo_tmp_dir, f"gen_{i}.png")
                        gen_image.save(temp_path)
                        curr_psnr, _ = eval_quality(taskB_gt_path, temp_path)
                        logging.info(f"Attempt {i+1}: Saved to {temp_path}, PSNR: {curr_psnr:.2f}")
                        if curr_psnr > best_psnr:
                            best_psnr = curr_psnr
                            best_gen_path = temp_path
                            logging.info(f"Attempt {i+1}: New best image found!")
                    else:
                        logging.warning(f"Attempt {i+1}: Seedream API call succeeded but returned NO image.")

                if best_gen_path:
                    shutil.move(best_gen_path, final_path)

                    psnr, ssim, viescore = evaluate_generated(
                        taskB_gt_path, final_path,
                        taskA_in, taskA_out, taskB_in,
                        pair_key.split('__')[0], pair_key.split('__')[1]
                    )
                    log_entry = {
                        "combo_id": combo_id,
                        "final_image": final_path,
                        "psnr": psnr,
                        "ssim": ssim,
                        "viescore": viescore
                    }
                    log_file.write(json.dumps(log_entry) + '\n')
                    log_file.flush()
                    os.fsync(log_file.fileno())
                    logging.info(f"Combo {combo_id}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, VIEScore={viescore:.2f}")
                    shutil.rmtree(combo_tmp_dir, ignore_errors=True)

            all_scores = []
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    for line in f:
                        try:
                            res_entry = json.loads(line)
                            if all(k in res_entry for k in ("psnr", "ssim", "viescore")):
                                all_scores.append(res_entry)
                        except:
                            continue

            if all_scores:
                metrics = {
                    "num_samples": len(all_scores),
                    "avg_psnr": np.mean([s['psnr'] for s in all_scores]),
                    "avg_ssim": np.mean([s['ssim'] for s in all_scores]),
                    "avg_viescore": np.mean([s['viescore'] for s in all_scores])
                }
                with open(os.path.join(pair_res_dir, "evaluation_results.json"), 'w') as f:
                    json.dump(metrics, f, indent=4)
                final_results[pair_key] = metrics

    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), 'w') as f:
        json.dump(final_results, f, indent=4)

    logging.info("Evaluation completed. Results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_qwen_for_prompt", action="store_true", default=False,
                        help="Use Qwen for generating text prompt")
    parser.add_argument("--fixed_prompt", type=str, default=None,
                        help="Fixed text prompt if not using Qwen")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--num_tries", type=int, default=5,
                        help="Number of generation attempts per combination")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sync_mode", action="store_true", default=False)
    parser.add_argument("--disable_safety_checker", dest="enable_safety_checker",
                        action="store_false", default=True)
    parser.add_argument("--enhance_prompt_mode", type=str, default="standard",
                        choices=["standard", "fast"])
    parser.add_argument("--with_logs", action="store_true", default=False)
    parser.add_argument("--no_match_taskb_size", dest="match_taskb_size",
                        action="store_false", default=True)
    args = parser.parse_args()
    run_evaluation(args)
