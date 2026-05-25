import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

from eval import hashed_id, generate_text_prompt, evaluate_generated

firered_root = os.path.join(os.getcwd(), "model/FireRed-Image-Edit")
if firered_root not in sys.path:
    sys.path.append(firered_root)

viescore_path = '/data1/tzz/huixin/Task-Transfer/VIEScore'
if viescore_path not in sys.path:
    sys.path.append(viescore_path)

DATA_TASKS_DIR = "data/tasks"
EVAL_DATASET_JSON = "data/dataset/eval_dataset.json"
OUTPUT_DIR = "data/output/baseline/firered/output_qwen"
FIRERED_MODEL_PATH = "FireRedTeam/FireRed-Image-Edit-1.1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
CHECKPOINT_PATH = "Qwen3-VL/qwen-vl-finetune/output/checkpoint-4875"

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def load_firered_pipeline(model_path, optimized=False):
    logging.info(f"Loading FireRed-Image-Edit from {model_path}...")

    if optimized:
        try:
            from utils.fast_pipeline import load_fast_pipeline
        except Exception as e:
            raise ImportError(
                "Failed to import FireRed optimized pipeline. "
                "Please run from a workspace that has model/FireRed-Image-Edit "
                "or install the official FireRed repository."
            ) from e

        pipe = load_fast_pipeline(model_path)
    else:
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")

    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_image_firered(pipe, taskA_in, taskA_out, taskB_in, text_prompt,
                           seed=42, true_cfg_scale=4.0, num_inference_steps=40):
    images = [
        Image.open(os.path.join(DATA_TASKS_DIR, taskA_in)).convert("RGB"),
        Image.open(os.path.join(DATA_TASKS_DIR, taskA_out)).convert("RGB"),
        Image.open(os.path.join(DATA_TASKS_DIR, taskB_in)).convert("RGB"),
    ]

    try:
        inputs = {
            "image": images,
            "prompt": text_prompt,
            "generator": torch.Generator(device="cuda").manual_seed(seed),
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": " ",
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": 1,
        }

        with torch.inference_mode():
            output = pipe(**inputs)
        return output.images[0]
    except Exception as e:
        logging.error(f"FireRed generation failed: {e}")
        return None


def run_evaluation(args):
    with open(EVAL_DATASET_JSON, 'r') as f:
        eval_data = json.load(f)
        # eval_data.reverse()

    grouped = {}
    for entry in eval_data:
        taskA = entry['taskA_input'].split('/')[0]
        taskB = entry['taskB_input'].split('/')[0]
        pair_key = f"{taskA}__{taskB}"
        grouped.setdefault(pair_key, []).append(entry)

    final_results = {}

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

    pipe = load_firered_pipeline(args.model_path, optimized=args.optimized)

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

                text_prompt = generate_text_prompt(
                    taskA_in, taskA_out, taskB_in,
                    model=prompt_qwen_model,
                    processor=prompt_qwen_processor,
                    use_qwen=args.use_qwen_for_prompt,
                    fixed_prompt=args.fixed_prompt
                )

                gen_image = generate_image_firered(
                    pipe, taskA_in, taskA_out, taskB_in, text_prompt,
                    seed=args.seed,
                    true_cfg_scale=args.true_cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                )
                if gen_image:
                    logging.info(f"Successfully received an image from FireRed.")
                    gen_image.save(final_path)

                    psnr, ssim, viescore = evaluate_generated(
                        os.path.join(DATA_TASKS_DIR, taskB_out), final_path,
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
    parser.add_argument("--model_path", type=str, default=FIRERED_MODEL_PATH)
    parser.add_argument("--use_qwen_for_prompt", action="store_true", default=False,
                        help="Use Qwen for generating text prompt")
    parser.add_argument("--fixed_prompt", type=str, default=None,
                        help="Fixed text prompt if not using Qwen")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--optimized", action="store_true", default=False)
    args = parser.parse_args()
    run_evaluation(args)
