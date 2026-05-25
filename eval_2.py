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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from GrAInS.src.attribution.gradient.vlm_grad import \
    get_token_attributions_contrastive
from GrAInS.src.utils.config import MODEL_NAME_MAP
from GrAInS.src.utils.model import load_vlm_model_and_processor
from VIEScore.paper_implementation.imagen_museum.utils import \
    write_entry_to_json_file

# Add VIEScore path
viescore_path = '/data1/tzz/huixin/Task-Transfer/VIEScore'
if viescore_path not in sys.path:
    sys.path.append(viescore_path)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Set constants
DATA_TASKS_DIR = "data/tasks"
TRAIN_DATASET_JSON = "data/dataset/train_dataset.json"
EVAL_DATASET_JSON = "data/dataset/eval_dataset_2.json"
OUTPUT_DIR = "data/output/ablation/output_same_qwen"
TMP_DIR = "data/tmp/ablation/tmp_same_qwen"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

BASE_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_MODEL = "qwen-3-vl-4b-instruct"
# CHECKPOINT_PATH = "Qwen3-VL/qwen-vl-finetune/output/checkpoint-4875"

GEMINI_API_KEY = "insert_your_gemini_api_key_here"
GEMINI_MODEL = "gemini-2.5-flash-image-preview:generateContent"
BASE_URL = "https://globalai.vip"
# BASE_URL = "http://82.29.71.210:5300"
API_KEY_HEADER = "api-key"

# How many chars to show when printing long model responses
TRUNCATE_LEN = 2000


def _shorten(text: str, n: int = TRUNCATE_LEN) -> str:
    """Return at most n characters of text, with an indicator if truncated."""
    if not text:
        return ""
    try:
        if len(text) <= n:
            return text
        return text[:n] + f"... (truncated, total {len(text)} chars)"
    except Exception:
        return text


# Function to generate unique ID
def hashed_id(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:10]


# Function to create Gemini client
def create_gemini_client():
    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            base_url=BASE_URL,
            headers={API_KEY_HEADER: GEMINI_API_KEY}
        )
    )


# Function to generate eval_dataset.json
def generate_eval_dataset():
    """Generate new image combinations (Same-Task) not in train data."""
    if os.path.exists(EVAL_DATASET_JSON):
        logging.info(f"Loading existing {EVAL_DATASET_JSON}")
        with open(EVAL_DATASET_JSON, 'r') as f:
            eval_data = json.load(f)
    else:
        random.seed(42)
        with open(TRAIN_DATASET_JSON, 'r') as f:
            train_data = json.load(f)

        # Extract existing input images to avoid overlap
        # We track individual input files to ensure we don't reuse specific images used in training
        used_inputs = set()
        for entry in train_data:
            used_inputs.add(entry['taskA_input'])
            used_inputs.add(entry['taskB_input'])

        # Get all images from data/tasks and pair by stem
        all_tasks = [d for d in os.listdir(DATA_TASKS_DIR)
                     if os.path.isdir(os.path.join(DATA_TASKS_DIR, d))]

        task_pairs_dict = {}
        for task in all_tasks:
            input_dir = os.path.join(DATA_TASKS_DIR, task, 'input')
            output_dir = os.path.join(DATA_TASKS_DIR, task, 'output')
            inputs = [f for f in os.listdir(input_dir)
                      if f.endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(input_dir) else []
            outputs = [f for f in os.listdir(output_dir)
                       if f.endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(output_dir) else []

            pairs = []
            for inp in inputs:
                stem, ext = os.path.splitext(inp)
                matching_out = next(
                    (out for out in outputs if os.path.splitext(out)[0] == stem), None)
                if matching_out:
                    pairs.append({
                        "input": os.path.join(task, 'input', inp),
                        "output": os.path.join(task, 'output', matching_out)
                    })
            task_pairs_dict[task] = pairs

        # Generate new Same-Task combos
        eval_data = []
        for task in all_tasks:
            pairs = task_pairs_dict.get(task, [])
            # Filter out pairs where the input image was used in training
            available_pairs = [p for p in pairs if p['input'] not in used_inputs]
            if len(available_pairs) < 2:
                continue

            # Create combinations where Pair 1 != Pair 2
            candidates = []
            random.shuffle(available_pairs)
            for i in range(len(available_pairs) - 1):
                ctx_pair = available_pairs[i]
                target_pair = available_pairs[i+1]
                candidates.append({
                    "taskA_input": ctx_pair['input'],
                    "taskA_output": ctx_pair['output'],
                    "taskB_input": target_pair['input'],
                    "taskB_output": target_pair['output']
                })

            selected = candidates[:100]
            if len(selected) < 100:
                logging.warning(f"Only {len(selected)} combos for {task}!")
            eval_data.extend(selected)

        with open(EVAL_DATASET_JSON, 'w') as f:
            json.dump(eval_data, f, indent=4)

        logging.info(f"Generated {len(eval_data)} new Same-Task combos for eval dataset.")

    # Group by task name
    grouped = {}
    for entry in eval_data:
        task_name = entry['taskA_input'].split('/')[0]
        grouped.setdefault(task_name, []).append(entry)

    return eval_data, grouped


# Function to generate text prompt
def generate_text_prompt(taskA_input, taskA_output, taskB_input, model, processor, use_qwen=True, fixed_prompt=None):
    """Generate text prompt using Base Qwen or fixed prompt."""
    if not use_qwen and fixed_prompt is None:
        return "This is a visual in-context learning task. The first two images are an input and output of an image processing task. The third image is the input for the SAME task. The goal is to perform this task on the third image and generate output image, learning from the pair before."

    elif not use_qwen and fixed_prompt is not None:
        taskA = taskA_input.split('/')[0]
        prompt = fixed_prompt.replace('[TASK_NAME]', taskA)
        logging.info(f"Using fixed prompt:\n{prompt}")
        return prompt

    # Build messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": os.path.join(DATA_TASKS_DIR, taskA_input),
                    "min_pixels": 12544,
                    "max_pixels": 401408,
                },
                {
                    "type": "image",
                    "image": os.path.join(DATA_TASKS_DIR, taskA_output),
                    "min_pixels": 12544,
                    "max_pixels": 401408,
                },
                {
                    "type": "image",
                    "image": os.path.join(DATA_TASKS_DIR, taskB_input),
                    "min_pixels": 12544,
                    "max_pixels": 401408,
                },
                {
                    "type": "text",
                    "text": (
                        "You are an expert in analyzing image processing tasks. Below are three images.\nThe Picture 1 and 2 form an example pair (Input -> Output) representing a specific vision task; the third image Picture 3 is input of the SAME task.\nPlease simply describe the input images, focus on the visual changes from input to output.\nDon't give me long descriptions or explanations; keep it concise and to the point.\nDon't tell me exactly what the tasks are (e.g., denoising, colorization, or shadow removal); instead, use implicit words and highlight how input and output images differ and the task's objective and effect.\nFit your answer into 2 sentences: 1) input image descriptions (what need to be done in Picture 1 and 3); 2) visual changes (what the task actually does).\nI know you can't see output of Picture 3 but it requires the same visual task.\nAgain, do not explicitly name the task (e.g., deblurring), use implicit descriptive words."
                    ),
                },
            ],
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    # Generate output
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            do_sample=True,
            use_cache=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Context string for Gemini
    instruct_text = "This is a visual in-context learning task. The first two images are an input and output of an image processing task. The third image is the input for the SAME task. The goal is to perform this task on the third image and generate output image, learning from the pair before."
    Qwen_text = instruct_text + output_text[0] if output_text else instruct_text

    print(f"Generated prompt:\n{Qwen_text if output_text else instruct_text}")
    return Qwen_text if output_text else instruct_text


def _extract_image_from_parts(parts):
    """
    Extracts an image from the parts returned by Gemini.
    1. First, it looks for binary `inline_data`.
    2. Then, it tries to parse a base64-encoded image from the `text` part.
    Returns a PIL.Image object if found, otherwise returns None.
    """
    # Loop through each part of the response
    for p in parts:
        # Case 1: The part contains direct binary image data.
        # Use getattr for safe access to avoid errors if the attribute doesn't exist.
        if getattr(p, "inline_data", None) and getattr(p.inline_data, "data", None):
            try:
                # Try to open the binary data as an image
                return Image.open(BytesIO(p.inline_data.data))
            except Exception as e:
                # If opening fails, log it and continue to the next part
                logging.warning(f"Could not open inline_data as image: {e}")
                pass

        # Case 2: The part contains text, which might have a base64 image embedded.
        if getattr(p, "text", None):
            # Use regex to find the base64 image pattern
            m = re.search(
                r"data:image/(?:png|jpeg|jpg);base64,([A-Za-z0-9+/=\s\r\n]+)",
                p.text
            )
            if m:
                # If a match is found, extract the base64 string (group 1)
                b64_str = m.group(1)
                try:
                    # Decode the base64 string into raw bytes
                    raw_bytes = base64.b64decode(b64_str)
                    # Try to open the bytes as an image
                    return Image.open(BytesIO(raw_bytes))
                except Exception as e:
                    # If decoding or opening fails, log it and continue
                    logging.warning(f"Could not decode or open base64 image: {e}")

            url_match = re.search(r"!\[.*?\]\((https?://[^\s)]+)\)", p.text)
            if url_match:
                image_url = url_match.group(1)
                logging.info(f"Found Markdown Image URL: {image_url}")
                try:
                    import requests
                    import io
                    resp = requests.get(image_url, timeout=10)
                    if resp.status_code == 200:
                        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                        return image
                    else:
                        logging.warning(f"Could not download the image URL, code={resp.status_code}")
                except Exception as e:
                    logging.warning(f"Could not download image from URL: {e}")

    # If no image is found in any part, return None
    return None


# Function to generate image with Gemini
def generate_image(taskA_input, taskA_output, taskB_input, text_prompt, gt_ext='.jpg'):
    """Generate task B output using Gemini."""
    client = create_gemini_client()

    mime_a_in = "image/png" if taskA_input.endswith('.png') else "image/jpeg"
    mime_a_out = "image/png" if taskA_output.endswith('.png') else "image/jpeg"
    mime_b_in = "image/png" if taskB_input.endswith('.png') else "image/jpeg"

    image1_part = types.Part.from_bytes(
        data=open(os.path.join(DATA_TASKS_DIR, taskA_input), 'rb').read(),
        mime_type=mime_a_in)
    image2_part = types.Part.from_bytes(
        data=open(os.path.join(DATA_TASKS_DIR, taskA_output), 'rb').read(),
        mime_type=mime_a_out)
    image3_part = types.Part.from_bytes(
        data=open(os.path.join(DATA_TASKS_DIR, taskB_input), 'rb').read(),
        mime_type=mime_b_in)

    prompt_text = text_prompt

    contents = [types.Part(text=prompt_text),
                image1_part, image2_part, image3_part]

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        parts = response.candidates[0].content.parts

        for p in parts:
            if getattr(p, "text", None):
                logging.info(f"Gemini returned text:\n---\n{_shorten(p.text)}\n---")
                break

        image = _extract_image_from_parts(parts)
        if image:
            # If an image was found, return it
            return image
        else:
            # If no image was found, log a specific warning and return None
            logging.warning("Gemini response had no image (neither inline_data nor base64 in text).")
            return None

    except Exception as e:
        logging.warning(f"Gemini generation failed: {e}")

    return None


# Function to evaluate
def eval_quality(gt_path, gen_path):
    gt_img = Image.open(gt_path).convert("RGB")
    gen_img = Image.open(gen_path).convert("RGB")
    gen_img = gen_img.resize(gt_img.size, Image.BICUBIC)
    gt_np = np.array(gt_img)
    pred_np = np.array(gen_img)

    psnr = peak_signal_noise_ratio(gt_np, pred_np,
                                   data_range=255)
    ssim = structural_similarity(gt_np, pred_np,
                                 channel_axis=-1, data_range=255)

    return psnr, ssim


def evaluate_generated(gt_path, gen_path, taskA_input, taskA_output, taskB_input, taskA, taskB):
    """Evaluate PSNR, SSIM, and VIEScore."""

    # PSNR / SSIM
    psnr, ssim = eval_quality(gt_path, gen_path)

    # Build VIEScore prompt
    viescore_prompt = f"""
        The first two images show an example of visual task.
        The first image is the input of the task [TASK_NAME], and the second is the output.
        The third image is a new input of the same task [TASK_NAME].
        The goal is to apply a similar visual task transfer from the first example to the new input.
        Please evaluate the fourth image, which is the model's generated output for the [TASK_NAME] task.
        Rate the fourth image based on two criteria:
        1. **Semantic Consistency (SC):** How well does the fourth image successfully obey the [TASK_NAME], similar to how the [TASK_NAME] was done in the example? (1-10)
        2. **Perceptual Quality (PQ):** Is the fourth image of high visual quality? (1-10)
        Return JSON strictly in this format: {{"score": [SC, PQ], "reasoning": "..."}}
    """.replace('[TASK_NAME]', taskA)

    # Pack images as Parts
    image_list = [
        os.path.join(DATA_TASKS_DIR, taskA_input),
        os.path.join(DATA_TASKS_DIR, taskA_output),
        os.path.join(DATA_TASKS_DIR, taskB_input),
        gen_path
    ]
    parts = [types.Part(text=viescore_prompt)]
    for path in image_list:
        mime_type = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        with open(path, "rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime_type))

    # Call Gemini API
    # mllm_model = Gemini()
    # prompt = mllm_model.prepare_prompt(image_list, viescore_prompt)
    client = create_gemini_client()

    # Adjusted to run evaluation
    viescore = 0.0
    is_verified = False
    tries, max_tries = 0, 2
    tmp_file_path = os.path.join(TMP_DIR, "viescore_log.json")
    uid = hashed_id(taskA_input, taskB_input, gen_path)

    while not is_verified and tries < max_tries:
        try:
            # result = mllm_model.get_parsed_output(prompt)
            # print("Raw result from Gemini:\n", result)
            resp = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024
                )
            )
            result_text = resp.candidates[0].content.parts[0].text
            print("Raw result from Gemini:\n", result_text)

            is_verified = write_entry_to_json_file(
                input_string=result_text,
                uid=uid,
                prompt_input=viescore_prompt,
                vision_input=image_list,
                output_file_name=tmp_file_path,
                give_up_parsing=False
            )

            if is_verified is True:
                with open(tmp_file_path, "r") as f:
                    data = json.load(f)
                scores = data[uid].get("score", [])
                if len(scores) == 2:
                    sc, pq = scores
                    viescore = (sc + pq) / 2
                elif len(scores) == 1:
                    viescore = scores[0]
                break
            elif is_verified == "rate_limit_exceeded":
                logging.warning("Gemini rate limit exceeded.")
                break
            else:
                logging.warning(f"Parsing failed on try {tries+1}")

        except Exception as e:
            logging.warning(f"Error during Gemini evaluation: {e}")

        tries += 1

    if not is_verified:
        logging.error(f"Failed to get valid VIEScore for {gen_path}")

    return psnr, ssim, viescore


# Main pipeline
def run_evaluation(args):
    """Run the full evaluation pipeline."""
    eval_data, grouped = generate_eval_dataset()
    final_results = {}

    # Load model
    prompt_qwen_model, prompt_qwen_processor = load_vlm_model_and_processor(
        MODEL_NAME_MAP[QWEN_MODEL])

    if args.use_qwen_for_prompt:
        prompt_qwen_model.eval()
    else:
        prompt_qwen_model = None
        prompt_qwen_processor = None

    for pair_key, entries in grouped.items():
        logging.info(f"Processing pair: {pair_key}")
        pair_res_dir = os.path.join(OUTPUT_DIR, pair_key)
        os.makedirs(pair_res_dir, exist_ok=True)
        log_path = os.path.join(pair_res_dir, "evaluation_log.jsonl")

        # Read existing logs to avoid duplicates
        existing_combo_ids = set()
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    existing_combo_ids.add(entry['combo_id'])

        with open(log_path, 'a') as log_file:
            for entry in entries:
                taskA = pair_key
                taskB = pair_key
                taskA_input = entry['taskA_input']
                taskA_output = entry['taskA_output']
                taskB_input = entry['taskB_input']
                taskB_output = entry['taskB_output']
                taskB_gt_path = os.path.join(DATA_TASKS_DIR, taskB_output)
                gt_ext = os.path.splitext(taskB_gt_path)[1]

                combo_id = hashed_id(taskA_input, taskB_input)

                a_name = os.path.basename(taskA_input)
                b_name = os.path.basename(taskB_input)
                final_path = os.path.join(pair_res_dir, f"{a_name}_{b_name}_{combo_id}{gt_ext}")

                # Step 1: Check if the final image already exists
                if os.path.exists(final_path):
                    # If the image exists, check if its metrics are already logged
                    if combo_id in existing_combo_ids:
                        logging.info(f"COMPLETE: Skipping combo {combo_id}, image and metrics already exist.")
                        continue
                    else:
                        # If image exists but metrics are missing, calculate and log them now
                        logging.info(f"RESUMING: Found image for {combo_id}, calculating and logging metrics...")
                        try:
                            psnr, ssim, viescore = evaluate_generated(taskB_gt_path, final_path,
                                                                      taskA_input, taskA_output,
                                                                      taskB_input, taskA, taskB)
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
                            logging.info(f"SUCCESS: Logged metrics for existing image {combo_id}.")
                        except Exception as e:
                            logging.error(f"FAILURE: Could not evaluate existing image {final_path}. Error: {e}")
                        continue

                # If we reach here, it means neither image nor log exists, so we proceed
                logging.info(f"STARTING: Processing new combo {combo_id}.")
                combo_tmp_dir = os.path.join(TMP_DIR, pair_key, combo_id)
                os.makedirs(combo_tmp_dir, exist_ok=True)

                # Step 2: Generate text prompt
                text_prompt = generate_text_prompt(taskA_input, taskA_output,
                                                   taskB_input,
                                                   model=prompt_qwen_model,
                                                   processor=prompt_qwen_processor,
                                                   use_qwen=args.use_qwen_for_prompt,
                                                   fixed_prompt=args.fixed_prompt)

                # Step 3: Generate many images, select best by PSNR
                best_psnr = -np.inf
                best_gen_path = None
                for i in range(args.num_tries):
                    try:
                        gen_image = generate_image(taskA_input, taskA_output,
                                                   taskB_input, text_prompt,
                                                   gt_ext)
                        if gen_image:
                            logging.info(f"Attempt {i+1}: Successfully received an image from Gemini.")
                            temp_path = os.path.join(combo_tmp_dir,
                                                     f"gen_{i}{gt_ext}")
                            gen_image.save(temp_path)
                            curr_psnr, _ = eval_quality(taskB_gt_path,
                                                        temp_path)
                            logging.info(f"Attempt {i+1}: Saved to {temp_path}, PSNR: {curr_psnr:.2f}")
                            if curr_psnr > best_psnr:
                                best_psnr = curr_psnr
                                best_gen_path = temp_path
                                logging.info(f"Attempt {i+1}: New best image found!")
                        else:
                            logging.warning(f"Attempt {i+1}: Gemini API call succeeded but returned NO image.")

                    except Exception as e:
                        logging.warning(f"Generation attempt {i} failed: {e}")

                if best_gen_path:
                    shutil.move(best_gen_path, final_path)

                    # Step 4: Evaluate best
                    psnr, ssim, viescore = evaluate_generated(taskB_gt_path, final_path,
                                                              taskA_input, taskA_output,
                                                              taskB_input, taskA, taskB)
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

            # Average for pair
            all_scores = []
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if all(k in entry for k in ("psnr", "ssim", "viescore")):
                                all_scores.append(entry)
                        except Exception:
                            continue
            if all_scores:
                avg_psnr = np.mean([s['psnr'] for s in all_scores])
                avg_ssim = np.mean([s['ssim'] for s in all_scores])
                avg_viescore = np.mean([s['viescore'] for s in all_scores])
                pair_metrics_path = os.path.join(pair_res_dir,
                                                 "evaluation_results.json")
                with open(pair_metrics_path, 'w') as f:
                    json.dump({
                        "num_samples": len(all_scores),
                        "avg_psnr": avg_psnr,
                        "avg_ssim": avg_ssim,
                        "avg_viescore": avg_viescore
                    }, f, indent=4)
                final_results[pair_key] = {
                    "num_samples": len(all_scores),
                    "avg_psnr": avg_psnr,
                    "avg_ssim": avg_ssim,
                    "avg_viescore": avg_viescore
                }

    # Save final results
    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), 'w') as f:
        json.dump(final_results, f, indent=4)

    logging.info("Evaluation completed. Results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VICL Evaluation Pipeline")
    parser.add_argument("--use_qwen_for_prompt", action="store_true",
                        default=False, help="Use Qwen for generating text prompt")
    parser.add_argument("--fixed_prompt", type=str, default=None,
                        # This is a visual in-context learning task. The first two images are an input and output of Task: [TASK_NAME]. The third image is the input for the SAME task. The goal is to perform this task on the third image and generate output image, learning from the pair before.
                        # This is a visual in-context learning task. The first two images are an input and output of an image processing task. The third image is the input for the SAME task. The goal is to perform this task on the third image and generate output image, learning from the pair before.
                        help="Fixed text prompt if not using Qwen")
    parser.add_argument("--num_tries", type=int, default=5,
                        help="Number of generation attempts per combination")
    args = parser.parse_args()

    run_evaluation(args)
