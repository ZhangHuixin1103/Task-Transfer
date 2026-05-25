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
from google import genai
from google.genai import types
from peft import PeftModel
from PIL import Image, ImageFilter
from qwen_vl_utils import process_vision_info
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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
EVAL_DATASET_JSON = "data/dataset/eval_dataset_1.json"
OUTPUT_DIR = "data/output/ablation/output_one_fix"
TMP_DIR = "data/tmp/ablation/output_one_fix"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

GEMINI_API_KEY = "insert_your_gemini_api_key_here"
GEMINI_MODEL = "gemini-2.5-flash-image-preview:generateContent"
BASE_URL = "https://globalai.vip"
# BASE_URL = "http://82.29.71.210:5300"
API_KEY_HEADER = "api-key"

# How many chars to show when printing long model responses
TRUNCATE_LEN = 2000

# Task Descriptions
TASK_DEFINITIONS = {
    "colorization": {
        "description": "**Input Image Descriptions:** The input image is a black-and-white photograph, while the output image is in color.\n **Visual Changes:** This task transformed the black-and-white image into a full-color image, adding vivid colors to the input figure, enhancing its visual appeal and detail. It focuses on colorization, converting a grayscale image into a colorful one, which significantly alters the perception and mood of the image by introducing bright, lively colors."
    },
    "harmonization": {
        "description": "**Input Image Descriptions:** Both images show similar objects and background, but the output image seems to make one object compatible with the background.\n **Visual Changes:** There are no significant visual changes between the two images, and the task might involve maintaining the original appearance while adjusting the color, illumination, and style of foreground to be compatible with the background. The output image seems more realistic."
    },
    "style_transfer": {
        "description": "**Input Image Descriptions:** The output image shows a similar landscape to the input image with varying lighting conditions and style, indicating a change in weather or season or time of day. It depicts a similar scene but indicating a change in the environment or perspective.\n **Visual Changes:** The transformation from the input to the output image involves a shift between appearances of the same landscape, highlighting the contrast between different weather conditions and seasons. This task focuses on adjusting lighting or time-related effects."
    },
    "deblurring": {
        "description": "**Input Image Descriptions:** The input image shows a blurry scene, lacking sharpness and detail, while the output image is clearer.\n **Visual Changes:** The transformation from the input to the output image involves sharpening the focus, making the details of the objects in image more distinct and clear. The task focuses on improving the sharpness and clarity of the image, addressing issues like blurriness."
    },
    "dehazing": {
        "description": "**Input Image Descriptions:** The input image depicts a setting with fog. The image is overexposed, making it difficult to discern details. The output image becomes clearer and has reduced exposure, allowing for better visibility.\n **Visual Changes:** This task emphasizes adjusting the lighting, brightness and contrast to enhance the overall visibility and color saturation of the scene. It addresses visual distortions to enhance detail visibility."
    },
    "demoireing": {
        "description": "**Input Image Descriptions:** The input image has a slight color distortion and depicts moire patterns (colored stripes). The output image is more naturally vibrant.\n **Visual Changes:** The visual change from the input to the output image appears to involve fixing the unnatural stripes. The output image corrects the distortion (moire pattern), restoring natural colors without altering the structural details of the scene."
    },
    "deraining": {
        "description": "**Input Image Descriptions:** The input image depicts a rainy scene including rain or water droplets, while the output image is clear.\n **Visual Changes:** The output image appears cleaner, with reduced visibility of water droplets or obstructions, suggesting the removal of artifacts caused by rain."
    },
    "denoising": {
        "description": "**Input Image Descriptions:** Both images show similar objects and background, but the input image slightly shows noise above the picture.\n **Visual Changes:** The output image exhibits a slight shift in color balance and saturation, with some colors appearing more vivid or altered in hue compared to the input. The overall brightness and contrast seem to have been adjusted, enhancing the visibility and removing the noise."
    },
    "inpainting": {
        "description": "**Input Image Descriptions:** There are black lines and boxes overlaid on the input image, which appear to be empty areas that need to be filled.\n **Visual Changes:** The output image has the black lines and boxes removed and refilled, resulting in a cleaner and more focused view of the input image without any distractions. This task focuses on removing unwanted graphical elements (lines and boxes) and making the image complete."
    },
    "light_enhancement": {
        "description": "**Input Image Descriptions:** The input image is dark and underexposed and lacks visibility, making it difficult to discern details. The output image is significantly brighter and shows a well-lit scene, indicating that the task involves improving visibility.\n **Visual Changes:** The task enhances the lighting and clarity of the input image, making the contents visible and detailed, similar to the second image. This suggests an improvement in brightness and contrast."
    },
    "reflection_removal": {
        "description": "**Input Image Descriptions:** Both images show similar objects and background where there's a glass wall or window in the front, but the output image doesn't have any reflections on glass.\n **Visual Changes:** The task appears to involve adjusting the lighting or contrast of the image, separating a desired background scene from unwanted reflections, and making the silhouettes and background more distinct."
    },
    "shadow_removal": {
        "description": "**Input Image Descriptions:** The input image shows a shadow cast on the ground or surface. The output image displays the same scene without the shadow.\n **Visual Changes:** The task removes the shadow from the ground or surface, altering the appearance by eliminating the dark area caused by the shadow."
    }
}


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
    """Generate single image tasks not in train data."""
    if os.path.exists(EVAL_DATASET_JSON):
        logging.info(f"Loading existing {EVAL_DATASET_JSON}")
        with open(EVAL_DATASET_JSON, 'r') as f:
            eval_data = json.load(f)
    else:
        random.seed(42)
        with open(TRAIN_DATASET_JSON, 'r') as f:
            train_data = json.load(f)

        # Load train data to record used images
        used_inputs = set()
        for entry in train_data:
            used_inputs.add(entry['taskA_input'])
            used_inputs.add(entry['taskB_input'])

        # Get all tasks
        all_tasks = [d for d in os.listdir(DATA_TASKS_DIR)
                     if os.path.isdir(os.path.join(DATA_TASKS_DIR, d))]
        eval_data = []
        for task_name in all_tasks:
            input_dir = os.path.join(DATA_TASKS_DIR, task_name, 'input')
            output_dir = os.path.join(DATA_TASKS_DIR, task_name, 'output')
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
                    input_rel_path = os.path.join(task_name, 'input', inp)
                    if input_rel_path not in used_inputs:
                        pairs.append({
                            "task": task_name,
                            "input": input_rel_path,
                            "output": os.path.join(task_name, 'output', matching_out)
                        })
            random.shuffle(pairs)
            selected = pairs[:100]
            # If not enough, we might need to reuse some
            if len(selected) < 100:
                logging.warning(f"Only {len(selected)} samples for {task_name}!")
            eval_data.extend(selected)

        with open(EVAL_DATASET_JSON, 'w') as f:
            json.dump(eval_data, f, indent=4)

        logging.info(f"Generated {len(eval_data)} samples for eval dataset.")

    # Group by task name
    grouped = {}
    for entry in eval_data:
        grouped.setdefault(entry['task'], []).append(entry)

    return eval_data, grouped


# Function to generate text prompt
def generate_text_prompt(task_name):
    """Generate text prompt from the definition dictionary."""
    if task_name not in TASK_DEFINITIONS:
        return f"Please perform the {task_name} task on this image."

    info = TASK_DEFINITIONS[task_name]
    description = info["description"]

    # Constructing instruction prompt
    prompt = f"You are an expert in analyzing image processing tasks. This picture is an input of an image processing task.\nHere is the description: {description}\nThe goal is to perform this task on the input image and generate output image."
    print(f"Generated prompt:\n{prompt}")
    return prompt


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
def generate_image(input_path, text_prompt):
    """Generate task output using Gemini."""
    client = create_gemini_client()

    mime_type = "image/png" if input_path.endswith('.png') else "image/jpeg"

    image_part = types.Part.from_bytes(
        data=open(os.path.join(DATA_TASKS_DIR, input_path), 'rb').read(),
        mime_type=mime_type
    )
    contents = [types.Part(text=text_prompt), image_part]

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        image = _extract_image_from_parts(response.candidates[0].content.parts)
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


def evaluate_generated(gt_path, gen_path, input_path, task_name):
    """Evaluate PSNR, SSIM, and VIEScore."""

    # PSNR / SSIM
    psnr, ssim = eval_quality(gt_path, gen_path)

    # Build VIEScore prompt
    viescore_prompt = f"""
        The first image is the input of a visual task {task_name}, and the second image is the generated output after applying the task.
        Please evaluate the second image, which is the model's generated output for the {task_name} task.
        Rate the second image based on two criteria:
        1. **Semantic Consistency (SC):** How well does the output image successfully obey the {task_name}, transforming the input appropriately? (1-10)
        2. **Perceptual Quality (PQ):** Is the generated image of high visual quality (realistic, no artifacts)? (1-10)
        Return JSON strictly in this format: {{"score": [SC, PQ], "reasoning": "..."}}
    """

    # Pack images as Parts
    image_list = [
        os.path.join(DATA_TASKS_DIR, input_path),
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
    uid = hashed_id(input_path, gen_path)

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

    for task_name, entries in grouped.items():
        logging.info(f"Processing task: {task_name}")
        task_res_dir = os.path.join(OUTPUT_DIR, task_name)
        os.makedirs(task_res_dir, exist_ok=True)
        log_path = os.path.join(task_res_dir, "evaluation_log.jsonl")

        # Read existing logs to avoid duplicates
        existing_ids = set()
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    existing_ids.add(json.loads(line)['sample_id'])

        with open(log_path, 'a') as log_file:
            for entry in entries:
                input_path = entry['input']
                gt_path = entry['output']
                full_gt_path = os.path.join(DATA_TASKS_DIR, gt_path)
                gt_ext = os.path.splitext(gt_path)[1]

                sample_id = hashed_id(input_path, gt_path)

                img_name = os.path.basename(input_path)
                final_path = os.path.join(task_res_dir, f"{img_name}_{sample_id}{gt_ext}")

                # Step 1: Check if the final image already exists
                if os.path.exists(final_path):
                    # If the image exists, check if its metrics are already logged
                    if sample_id in existing_ids:
                        logging.info(f"COMPLETE: Skipping sample {sample_id}, image and metrics already exist.")
                        continue
                    else:
                        # If image exists but metrics are missing, calculate and log them now
                        logging.info(f"RESUMING: Found image for {sample_id}, calculating and logging metrics...")
                        try:
                            psnr, ssim, viescore = evaluate_generated(full_gt_path, final_path,
                                                                      input_path, task_name)
                            log_file.write(json.dumps({
                                "sample_id": sample_id, "final_image": final_path,
                                "psnr": psnr, "ssim": ssim, "viescore": viescore
                            }) + '\n')
                            log_file.flush()
                            os.fsync(log_file.fileno())
                            logging.info(f"SUCCESS: Logged metrics for existing image {sample_id}.")
                        except Exception as e:
                            logging.error(f"FAILURE: Could not evaluate existing image {final_path}. Error: {e}")
                        continue

                # If we reach here, it means neither image nor log exists, so we proceed
                logging.info(f"STARTING: Processing new sample {sample_id}.")
                sample_tmp_dir = os.path.join(TMP_DIR, task_name, sample_id)
                os.makedirs(sample_tmp_dir, exist_ok=True)

                # Step 2: Generate text prompt
                text_prompt = generate_text_prompt(task_name)

                # Step 3: Generate many images, select best by PSNR
                best_psnr = -np.inf
                best_gen_path = None
                for i in range(args.num_tries):
                    try:
                        gen_image = generate_image(input_path, text_prompt)
                        if gen_image:
                            logging.info(f"Attempt {i+1}: Successfully received an image from Gemini.")
                            temp_path = os.path.join(sample_tmp_dir,
                                                     f"gen_{i}{gt_ext}")
                            gen_image.save(temp_path)
                            curr_psnr, _ = eval_quality(full_gt_path, temp_path)
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
                    psnr, ssim, viescore = evaluate_generated(full_gt_path, final_path,
                                                              input_path, task_name)

                    log_file.write(json.dumps({
                        "sample_id": sample_id, "final_image": final_path,
                        "psnr": psnr, "ssim": ssim, "viescore": viescore
                    }) + '\n')
                    log_file.flush()
                    os.fsync(log_file.fileno())
                    logging.info(f"Sample {sample_id}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, VIEScore={viescore:.2f}")

                    shutil.rmtree(sample_tmp_dir, ignore_errors=True)

            # Average for task
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
                with open(os.path.join(task_res_dir, "evaluation_results.json"), 'w') as f:
                    json.dump({
                        "num_samples": len(all_scores),
                        "avg_psnr": avg_psnr,
                        "avg_ssim": avg_ssim,
                        "avg_viescore": avg_viescore
                    }, f, indent=4)
                final_results[task_name] = {
                    "avg_psnr": avg_psnr,
                    "avg_ssim": avg_ssim,
                    "avg_viescore": avg_viescore
                }
            else:
                avg_psnr, avg_ssim, avg_viescore = 0.0, 0.0, 0.0

    # Save final results
    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), 'w') as f:
        json.dump(final_results, f, indent=4)

    logging.info("Evaluation completed. Results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VICL Evaluation Pipeline")
    parser.add_argument("--num_tries", type=int, default=5,
                        help="Number of generation attempts per combination")
    args = parser.parse_args()

    run_evaluation(args)
