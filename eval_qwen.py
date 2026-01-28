import argparse
import base64
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import torch
from google import genai
from google.genai import types
from peft import PeftModel
from PIL import Image, ImageFilter
from qwen_vl_utils import process_vision_info
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from eval import (evaluate_generated, generate_eval_dataset,
                  generate_text_prompt, hashed_id)
from VIEScore.paper_implementation.imagen_museum.utils import \
    write_entry_to_json_file
from diffusers import QwenImageEditPlusPipeline

# Add VIEScore path
viescore_path = '/data1/tzz/huixin/Task-Transfer/VIEScore'
if viescore_path not in sys.path:
    sys.path.append(viescore_path)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

DATA_TASKS_DIR = "data/tasks"
EVAL_DATASET_JSON = "data/dataset/eval_dataset.json"
OUTPUT_DIR = "data/output/baseline/qwen/output_qwen"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_MODEL = "qwen-3-vl-4b-instruct"
CHECKPOINT_PATH = "Qwen3-VL/qwen-vl-finetune/output/checkpoint-4875"


def generate_image_qwen(pipeline, img_paths, text_prompt):
    """
    Generate image using Qwen-Image-Edit-2511.
    img_paths: [taskA_input, taskA_output, taskB_input]
    """
    images = [Image.open(os.path.join(DATA_TASKS_DIR, p)).convert("RGB") for p in img_paths]

    inputs = {
        "image": images,
        "prompt": text_prompt,
        "negative_prompt": "低分辨率, 低画质, 肢体畸形, 手指畸形, 画面过饱和, 蜡像感, 人脸无细节, 过度光滑, 画面具有AI感。构图混乱。文字模糊, 扭曲",
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 4.0,
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        return output.images[0]


def run_evaluation(args):
    eval_data, grouped = generate_eval_dataset()
    
    print(f"Loading Pipeline: {args.model_id}")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16
    )
    pipeline.to('cuda')
    # pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    if args.use_qwen_for_prompt:
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        
        if os.path.exists(os.path.join(CHECKPOINT_PATH, "adapter_config.json")):
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_PATH, torch_dtype="auto", device_map="auto"
            )
            prompt_model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
            try:
                prompt_model = prompt_model.merge_and_unload()
            except Exception as e:
                logging.warning(f"Failed to merge LoRA: {e}")
        else:
            prompt_model = Qwen3VLForConditionalGeneration.from_pretrained(
                CHECKPOINT_PATH, torch_dtype="auto", device_map="auto"
            )

        prompt_model.eval()
        prompt_processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    else:
        prompt_model = None
        prompt_processor = None

    final_results = {}

    for pair_key, entries in grouped.items():
        logging.info(f"Processing pair: {pair_key}")
        pair_res_dir = os.path.join(OUTPUT_DIR, pair_key)
        os.makedirs(pair_res_dir, exist_ok=True)
        log_path = os.path.join(pair_res_dir, "evaluation_log.jsonl")

        existing_combo_ids = set()
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    log_entry = json.loads(line)
                    existing_combo_ids.add(log_entry['combo_id'])

        with open(log_path, 'a') as log_file:
            for entry in entries[:args.max_samples]:
                taskA, taskB = pair_key.split('__', 1)
                combo_id = hashed_id(entry['taskA_input'], entry['taskB_input'])
                
                gt_path = os.path.join(DATA_TASKS_DIR, entry['taskB_output'])
                gt_ext = os.path.splitext(gt_path)[1]
                save_name = f"{os.path.basename(entry['taskA_input'])}_{os.path.basename(entry['taskB_input'])}_{combo_id}{gt_ext}"
                final_path = os.path.join(pair_res_dir, save_name)

                if os.path.exists(final_path):
                    if combo_id in existing_combo_ids:
                        logging.info(f"COMPLETE: Skipping combo {combo_id}, image and metrics already exist.")
                        continue
                    else:
                        logging.info(f"RESUMING: Calculating metrics for {combo_id}...")
                        try:
                            psnr, ssim, viescore = evaluate_generated(
                                gt_path, final_path, 
                                entry['taskA_input'], entry['taskA_output'], entry['taskB_input'],
                                taskA, taskB
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
                        except Exception as e:
                            logging.error(f"Failed to calculate metrics for existing image {combo_id}: {e}")
                        continue

                logging.info(f"Generating for combo: {combo_id}")
                text_prompt = generate_text_prompt(
                    entry['taskA_input'], entry['taskA_output'], entry['taskB_input'],
                    model=prompt_model, processor=prompt_processor,
                    use_qwen=args.use_qwen_for_prompt, fixed_prompt=args.fixed_prompt
                )

                try:
                    img_list = [entry['taskA_input'], entry['taskA_output'], entry['taskB_input']]
                    generated_img = generate_image_qwen(pipeline, img_list, text_prompt)
                    generated_img.save(final_path)

                    psnr, ssim, viescore = evaluate_generated(
                        gt_path, final_path, 
                        entry['taskA_input'], entry['taskA_output'], entry['taskB_input'],
                        taskA, taskB
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
                except Exception as e:
                    logging.error(f"Failed to process {combo_id}: {e}")

        all_scores = []
        with open(log_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                all_scores.append(entry)
        
        if all_scores:
            avg_metrics = {
                "num_samples": len(all_scores),
                "avg_psnr": np.mean([s['psnr'] for s in all_scores]),
                "avg_ssim": np.mean([s['ssim'] for s in all_scores]),
                "avg_viescore": np.mean([s['viescore'] for s in all_scores])
            }
            with open(os.path.join(pair_res_dir, "evaluation_results.json"), 'w') as f:
                json.dump(avg_metrics, f, indent=4)
            final_results[pair_key] = avg_metrics

    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), 'w') as f:
        json.dump(final_results, f, indent=4)

    logging.info("Qwen Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit-2511 Evaluation")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--use_qwen_for_prompt", action="store_true", default=False)
    parser.add_argument("--fixed_prompt", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    run_evaluation(args)
