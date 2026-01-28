import argparse
import hashlib
import json
import logging
import os
import shutil
import sys

import numpy as np
import torch
from diffusers import AutoModel, Flux2Pipeline
from diffusers.utils import load_image
from PIL import Image
from tqdm import tqdm
from transformers import Mistral3ForConditionalGeneration

viescore_path = '/data1/tzz/huixin/Task-Transfer/VIEScore'
if viescore_path not in sys.path:
    sys.path.append(viescore_path)

from VIEScore.paper_implementation.imagen_museum.utils import \
    write_entry_to_json_file

DATA_TASKS_DIR = "data/tasks"
EVAL_DATASET_JSON = "data/dataset/eval_dataset.json"
OUTPUT_DIR = "data/output/baseline/flux/output_qwen"
REPO_ID = "diffusers/FLUX.2-dev-bnb-4bit"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
CHECKPOINT_PATH = "Qwen3-VL/qwen-vl-finetune/output/checkpoint-4875"

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def load_flux_pipeline():
    logging.info(f"Loading FLUX.2 4-bit from {REPO_ID}...")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        REPO_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu"
    )
    dit = AutoModel.from_pretrained(
        REPO_ID, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cpu"
    )
    pipe = Flux2Pipeline.from_pretrained(
        REPO_ID, text_encoder=text_encoder, transformer=dit, torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    return pipe

def generate_image_flux(pipe, taskA_in, taskA_out, taskB_in, text_prompt):
    ref_images = [
        load_image(os.path.join(DATA_TASKS_DIR, taskA_in)),
        load_image(os.path.join(DATA_TASKS_DIR, taskA_out)),
        load_image(os.path.join(DATA_TASKS_DIR, taskB_in))
    ]
    
    try:
        output = pipe(
            prompt=text_prompt,
            image=ref_images,
            generator=torch.Generator(device="cuda:0").manual_seed(42),
            num_inference_steps=30,
            guidance_scale=4.0,
        ).images[0]
        return output
    except Exception as e:
        logging.error(f"FLUX.2 generation failed: {e}")
        return None


def run_evaluation(args):
    with open(EVAL_DATASET_JSON, 'r') as f:
        eval_data = json.load(f)
    
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

    pipe = load_flux_pipeline()

    for pair_key, entries in grouped.items():
        logging.info(f"Processing pair: {pair_key}")
        pair_res_dir = os.path.join(OUTPUT_DIR, pair_key)
        os.makedirs(pair_res_dir, exist_ok=True)
        log_path = os.path.join(pair_res_dir, "evaluation_log.jsonl")

        existing_combo_ids = set()
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    try: existing_combo_ids.add(json.loads(line)['combo_id'])
                    except: continue

        with open(log_path, 'a') as log_file:
            for entry in entries[:args.max_samples]:
                taskA_in = entry['taskA_input']
                taskA_out = entry['taskA_output']
                taskB_in = entry['taskB_input']
                taskB_out = entry['taskB_output']
                
                from eval import hashed_id
                combo_id = hashed_id(taskA_in, taskB_in)

                if combo_id in existing_combo_ids:
                    logging.info(f"SKIP: {combo_id} already exists.")
                    continue

                from eval import generate_text_prompt
                text_prompt = generate_text_prompt(
                    taskA_in, taskA_out, taskB_in,
                    model=prompt_qwen_model,
                    processor=prompt_qwen_processor,
                    use_qwen=args.use_qwen_for_prompt,
                    fixed_prompt=args.fixed_prompt
                )

                gen_image = generate_image_flux(pipe, taskA_in, taskA_out, taskB_in, text_prompt)

                if gen_image:
                    final_path = os.path.join(pair_res_dir, f"{combo_id}.png")
                    gen_image.save(final_path)
                    
                    from eval import evaluate_generated
                    psnr, ssim, viescore = evaluate_generated(
                        os.path.join(DATA_TASKS_DIR, taskB_out), 
                        final_path, taskA_in, taskA_out, taskB_in, 
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
                    logging.info(f"ID {combo_id} Done. VIEScore: {viescore}")

        all_scores = []
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    try:
                        res_entry = json.loads(line)
                        if all(k in res_entry for k in ("psnr", "ssim", "viescore")):
                            all_scores.append(res_entry)
                    except: continue

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
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()
    run_evaluation(args)
