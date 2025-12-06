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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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
EVAL_DATASET_JSON = "data/dataset/eval_dataset.json"
OUTPUT_DIR = "data/output/output_qwen1"
TMP_DIR = "data/tmp/tmp_qwen1"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

BASE_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_MODEL = "qwen-2.5-vl-3b-instruct"
CHECKPOINT_PATH = "Qwen2.5-VL/qwen-vl-finetune/output/checkpoint-latest"

GEMINI_API_KEY = "sk-Uqq0JFYc56oSgTFmrnGRzZgbtV4NBoNJKm18hvnpQKoFHjJF"
# GEMINI_API_KEY = "sk-2LE9SvYG170QGDDX1ajIUlsuVxt1bqY9nY92BZAKvSZlPWFL"
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
    """Generate new image combinations per task pair not in train data."""
    if os.path.exists(EVAL_DATASET_JSON):
        logging.info(f"Loading existing {EVAL_DATASET_JSON}")
        with open(EVAL_DATASET_JSON, 'r') as f:
            eval_data = json.load(f)
    else:
        random.seed(42)
        with open(TRAIN_DATASET_JSON, 'r') as f:
            train_data = json.load(f)

        # Extract existing combinations and task pairs
        existing_combos = set()
        task_pairs = set()
        for entry in train_data:
            combo = (entry['taskA_input'],
                     entry['taskA_output'],
                     entry['taskB_input'])
            existing_combos.add(combo)
            taskA = entry['taskA_input'].split('/')[0]
            taskB = entry['taskB_input'].split('/')[0]
            task_pairs.add((taskA, taskB))

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

        # Generate new combos for each task pair
        eval_data = []
        for taskA, taskB in task_pairs:
            a_pairs = task_pairs_dict.get(taskA, [])
            b_pairs = task_pairs_dict.get(taskB, [])

            candidates = []
            for a_pair in a_pairs:
                for b_pair in b_pairs:
                    combo = (a_pair['input'],
                             a_pair['output'],
                             b_pair['input'])
                    if combo not in existing_combos:
                        candidates.append({
                            "taskA_input": a_pair['input'],
                            "taskA_output": a_pair['output'],
                            "taskB_input": b_pair['input'],
                            "taskB_output": b_pair['output']
                        })

            random.shuffle(candidates)
            selected = candidates[:100]
            if len(selected) < 100:
                logging.warning(f"Only {len(selected)} combos for {taskA} -> {taskB}!")
            eval_data.extend(selected)

        with open(EVAL_DATASET_JSON, 'w') as f:
            json.dump(eval_data, f, indent=4)

        logging.info(f"Generated {len(eval_data)} new combos for eval dataset.")

    # Group by task pairs
    grouped = {}
    for entry in eval_data:
        taskA = entry['taskA_input'].split('/')[0]
        taskB = entry['taskB_input'].split('/')[0]
        pair_key = f"{taskA}__{taskB}"
        grouped.setdefault(pair_key, []).append(entry)

    return eval_data, grouped


# Function to generate text prompt
def generate_text_prompt(taskA_input, taskA_output, taskB_input, model, processor, use_qwen=True, fixed_prompt=None):
    """Generate text prompt using finetuned Qwen or fixed prompt."""
    if not use_qwen and fixed_prompt is None:
        return fixed_prompt or "You are an expert in analyzing image processing tasks. Fix the third image and generate an output image."

    elif not use_qwen and fixed_prompt is not None:
        # Get task names from file paths
        taskA = taskA_input.split('/')[0]
        taskB = taskB_input.split('/')[0]

        # Replace placeholders with actual task names
        prompt = fixed_prompt.replace('[TASK_A_DEGRADATION]', taskA).replace('[TASK_B_DEGRADATION]', taskB)

        # Logging the final prompt
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
                    "text": "You are an expert in analyzing image processing tasks. Below are two vision tasks, A and B.\nThe Picture 1 and 2 belong to Task A, 1 is input and 2 is output; the third image Picture 3 is input of Task B.\nPlease analyze and describe the key differences between the two tasks shortly. Focus on the degradation to be removed, for example: 'Remove rain streaks', 'Enhance low light areas', or 'Remove shadows'.\nI know you can't see output of task B, but you can guess what task it is based on shortcoming of input.",
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
            max_new_tokens=8192,
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

    # Choose to add this to Qwen prompt or not
    instruct_text = ""
    # instruct_text = "This is a visual in-context learning task. The first two images are an input and output of Task A. The third image is the input for Task B. The goal is to perform Task B on the third image and generate output image, learning from Task A. Only modify the flaws in the third image; do not arbitrarily change the image's color tone (unless it's a task to colorize a black and white image) or the layout of the image content.\n"
    Qwen_text = instruct_text + output_text[0] if output_text else instruct_text

    print(f"Generated prompt:\n{Qwen_text if output_text else instruct_text}")
    return Qwen_text if output_text else instruct_text


def _fallback_simple_mask(b_input_path: str, save_dir: str, blur_r: int = 51, th: float = 0.30, smooth_r: int = 9) -> str:
    """
    Crude luminance/illumination-based mask.
    White = restore, Black = keep.
    Always returns a valid PNG mask path.
    """
    img = Image.open(b_input_path).convert("RGB")
    gray = np.array(img.convert("L")).astype(np.float32) / 255.0
    blur = np.array(img.filter(ImageFilter.GaussianBlur(
        radius=blur_r)).convert("L")).astype(np.float32) / 255.0

    eps = 1e-6
    sal = (blur - gray) / (blur + eps)
    sal = np.clip(sal, 0.0, 1.0)

    sal_img = Image.fromarray(
        (sal * 255).astype(np.uint8), "L").resize(img.size, Image.Resampling.LANCZOS)
    binary = (np.array(sal_img) > (th * 255)).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary, "L").filter(
        ImageFilter.GaussianBlur(radius=smooth_r))

    stem = os.path.splitext(os.path.basename(b_input_path))[0]
    out_path = os.path.join(save_dir, f"mask_{stem}.png")
    mask_img.save(out_path)
    return out_path


def _build_contrastive_texts(text_prompt: str, taskA: str, taskB: str) -> Tuple[str, str, str]:
    """
    Build attribution prompt + positive/negative responses for GrAInS contrastive attribution.
    Task A example is described in text only; Task B input image is passed to the model.
    """
    attribution_prompt = (
        f"You are given two related vision tasks.\n"
        f"- Task A ({taskA}): Image A_in → Image A_out.\n"
        f"- Task B ({taskB}): Image B_in is provided below.\n"
        f"Learn from Task A and decide where Image B_in should be restored/fixed."
    )
    pos_response = (
        f"Successfully perform {taskB} on Image B_in, following Task A -> B transfer."
        f"The goal is: {text_prompt}"
    )
    neg_response = (
        "Fail to restore Image B_in or introduce artifacts, color shifts, texture distortions, or hallucinations."
    )
    return attribution_prompt, pos_response, neg_response


def _select_image_token_scores(pos_scores, pos_ids, tokenizer) -> np.ndarray:
    """
    Try to keep only image-related token attributions.
    If we cannot identify them robustly, return all positive scores.
    """
    try:
        ids = pos_ids[0].detach().cpu().numpy().tolist()
    except Exception:
        return np.array(pos_scores, dtype=float)

    # Heuristics: try to find special vision tokens. If not found, fall back.
    image_like_tokens = {
        "<image>", "<img>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<image_patch>"
    }

    keep_idx = []
    for i, tid in enumerate(ids):
        tok = tokenizer.decode([tid], skip_special_tokens=False)
        tok_norm = (tok or "").strip().lower()
        if any(tag in tok_norm for tag in image_like_tokens):
            keep_idx.append(i)

    if not keep_idx:
        # Some tokenizers do not emit explicit image tags; fall back to all tokens
        return np.array(pos_scores, dtype=float)

    return np.array([pos_scores[i] for i in keep_idx], dtype=float)


def _scores_to_mask(scores: np.ndarray, target_size: Tuple[int, int], percentile: int) -> Optional[Image.Image]:
    """
    Convert a 1D vector of patch scores to a binary mask at target_size.
    """
    n = scores.shape[0]
    if n < 16:
        # Too few tokens to form a reasonable grid
        return None

    g = int(np.sqrt(n))
    if g * g != n:
        # Truncate to nearest smaller square
        g = int(np.floor(np.sqrt(n)))
        n2 = g * g
        if n2 < 16:
            return None
        scores = scores[:n2]

    # Normalize to [0,1]
    smin, smax = float(np.min(scores)), float(np.max(scores))
    if smax > smin:
        norm = (scores - smin) / (smax - smin)
    else:
        norm = np.zeros_like(scores)

    grid = (norm.reshape(g, g) * 255).astype(np.uint8)
    heat = Image.fromarray(grid, "L").resize(target_size,
                                             Image.Resampling.LANCZOS)

    arr = np.array(heat).astype(np.float32) / 255.0
    thr = np.percentile(arr, percentile)
    binary = (arr >= thr).astype(np.uint8) * 255
    return Image.fromarray(binary, "L")


# Function to generate mask
def generate_mask(
    taskA: str,
    taskB: str,
    text_prompt: str,
    taskA_input: Optional[str] = None,
    taskA_output: Optional[str] = None,
    taskB_input: str = None,
    qwen_model=None,
    qwen_processor=None,
    method: str = "integrated_gradients",
    threshold_percentile: int = 60,
) -> Optional[str]:
    """
    Generate a binary (black/white) mask for Task B input.
    - If Qwen+GrAInS are provided, compute contrastive token attribution and derive a spatial mask.
    - Otherwise, fall back to a luminance-based mask (your original approach).

    Args:
        text_prompt: the analysis prompt from your step(2); used to condition pos/neg responses.
        taskA_input, taskA_output: relative paths for Task A sample (needed for GrAInS).
        taskB_input: relative path under DATA_TASKS_DIR to Task B input image.
        qwen_model, qwen_processor: loaded Qwen VL + processor (needed for GrAInS).
        method: "vanilla" | "smoothgrad" | "integrated_gradients" (recommended).
        threshold_percentile: keep top-X% positive attribution as white.

    Returns:
        Path to saved PNG mask (white=restore, black=keep), or None on fatal error.
    """
    # Absolute paths
    b_abs = os.path.join(DATA_TASKS_DIR, taskB_input)
    if not os.path.exists(b_abs):
        logging.error(f"[Mask] Task B input not found: {b_abs}")
        return None

    # Case 1: GrAInS path if everything is provided
    can_grains = (
        (qwen_model is not None)
        and (qwen_processor is not None)
        and (taskA_input is not None)
        and (taskA_output is not None)
    )

    if can_grains:
        try:
            a_in_abs = os.path.join(DATA_TASKS_DIR, taskA_input)
            a_out_abs = os.path.join(DATA_TASKS_DIR, taskA_output)
            imgA_in = Image.open(a_in_abs).convert("RGB")
            imgA_out = Image.open(a_out_abs).convert("RGB")
            imgB_in = Image.open(b_abs).convert("RGB")
        except Exception as e:
            logging.warning(f"[Mask][GrAInS] Failed to open images ({e}).")
            return _fallback_simple_mask(b_abs, TMP_DIR)

        # Build texts for contrastive attribution
        attribution_prompt, pos_resp, neg_resp = _build_contrastive_texts(
            text_prompt, taskA, taskB)

        try:
            # Run contrastive attribution on 3 images
            attrib = get_token_attributions_contrastive(
                model=qwen_model,
                processor=qwen_processor,
                image=imgB_in,
                prompt=attribution_prompt,
                pos_response=pos_resp,
                neg_response=neg_resp,
                method=method,
            )
            pos_scores, pos_ids = attrib["pos"]

            # Try to keep image tokens only; else use all
            scores = _select_image_token_scores(pos_scores, pos_ids,
                                                qwen_processor.tokenizer)

            # Convert vector → grid → upsample → threshold
            mask_img = _scores_to_mask(
                scores, imgB_in.size, threshold_percentile)
            if mask_img is None:
                logging.warning("[Mask][GrAInS] Cannot form a valid patch grid.")
                return _fallback_simple_mask(b_abs, TMP_DIR)

            out_path = os.path.join(
                TMP_DIR, f"mask_grains_{os.path.basename(taskB_input)}.png")
            mask_img.save(out_path)
            logging.info(f"[Mask] Saved GrAInS-based mask → {out_path}")
            return out_path

        except Exception as e:
            logging.warning(f"[Mask][GrAInS] Attribution failed ({e}).")
            return _fallback_simple_mask(b_abs, TMP_DIR)

    # Case 2: Fallback (no Qwen/GrAInS or incomplete inputs)
    return _fallback_simple_mask(b_abs, TMP_DIR)


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
def generate_image(taskA_input, taskA_output, taskB_input, text_prompt, mask_path=None, gt_ext='.jpg'):
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
    if mask_path and os.path.exists(mask_path):
        prompt_text += "\nThe final image, the mask, indicates the regions on Image 3 to be restored. White areas can be restored to remove defects, while black areas are likely be kept unchanged.\nRestore Image 3 by fixing the defects mainly in the white regions of the mask. Leave the black regions of the mask untouched, and do not change colors, textures, or objects in these areas."

    contents = [types.Part(text=prompt_text),
                image1_part, image2_part, image3_part]

    if mask_path and os.path.exists(mask_path):
        mask_part = types.Part.from_bytes(data=open(mask_path, 'rb').read(),
                                          mime_type="image/png")
        contents.append(mask_part)

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
        The first image is the input of the first task [TASK_A_DEGRADATION], and the second is the output.
        The third image is a new input of the second task [TASK_B_DEGRADATION].
        The goal is to apply a similar visual task transfer from the first example to the new input.
        Please evaluate the fourth image, which is the model's generated output for the [TASK_B_DEGRADATION] task.
        Rate the fourth image based on two criteria:
        1. **Semantic Consistency (SC):** How well does the fourth image successfully obey the [TASK_B_DEGRADATION], similar to how the [TASK_A_DEGRADATION] was done in the example? (1-10)
        2. **Perceptual Quality (PQ):** Is the fourth image of high visual quality? (1-10)
        Return JSON strictly in this format: {{"score": [SC, PQ], "reasoning": "..."}}
    """.replace('[TASK_A_DEGRADATION]', taskA).replace('[TASK_B_DEGRADATION]', taskB)

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
    qwen_model, qwen_processor = load_vlm_model_and_processor(
        MODEL_NAME_MAP[QWEN_MODEL])

    if args.use_qwen_for_prompt:
        if os.path.exists(os.path.join(CHECKPOINT_PATH, "adapter_config.json")):
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_PATH, torch_dtype="auto", device_map="auto"
            )
            prompt_qwen_model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
            try:
                prompt_qwen_model = prompt_qwen_model.merge_and_unload()
            except Exception as e:
                logging.warning(f"Failed to merge LoRA: {e}")
        else:
            prompt_qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

        # Read existing logs to avoid duplicates
        existing_combo_ids = set()
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    existing_combo_ids.add(entry['combo_id'])

        with open(log_path, 'a') as log_file:
            pair_best_scores = []

            for entry in entries:
                taskA, taskB = pair_key.split('__', 1)
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
                            pair_best_scores.append(log_entry)
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

                # Step 3: Optional mask
                mask_path = generate_mask(
                    taskA=taskA, taskB=taskB, text_prompt=text_prompt,
                    taskA_input=taskA_input, taskA_output=taskA_output,
                    taskB_input=taskB_input,
                    qwen_model=qwen_model, qwen_processor=qwen_processor,
                    method="integrated_gradients", threshold_percentile=80
                ) if args.use_mask else None

                # Step 4: Generate many images, select best by PSNR
                best_psnr = -np.inf
                best_gen_path = None
                for i in range(args.num_tries):
                    try:
                        gen_image = generate_image(taskA_input, taskA_output,
                                                   taskB_input, text_prompt,
                                                   mask_path, gt_ext)
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

                    # Step 5: Evaluate best
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
                    pair_best_scores.append(log_entry)
                    log_file.flush()
                    os.fsync(log_file.fileno())
                    logging.info(f"Combo {combo_id}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, VIEScore={viescore:.2f}")

                    shutil.rmtree(combo_tmp_dir, ignore_errors=True)

            # Average for pair
            if pair_best_scores:
                avg_psnr = np.mean([s['psnr'] for s in pair_best_scores])
                avg_ssim = np.mean([s['ssim'] for s in pair_best_scores])
                avg_viescore = np.mean([s['viescore']
                                       for s in pair_best_scores])
                pair_metrics_path = os.path.join(pair_res_dir,
                                                 "evaluation_results.json")
                with open(pair_metrics_path, 'w') as f:
                    json.dump({
                        "avg_psnr": avg_psnr,
                        "avg_ssim": avg_ssim,
                        "avg_viescore": avg_viescore
                    }, f, indent=4)
                final_results[pair_key] = {
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
                        # This is a visual in-context learning task. The first two images are an input and output of Task A: [TASK_A_DEGRADATION]. The third image is the input for Task B: [TASK_B_DEGRADATION]. The goal is to perform Task B on the third image and generate output image, learning from Task A.
                        # This is a visual in-context learning task. The first two images are an input and output of Task A. The third image is the input for Task B. The goal is to perform Task B on the third image and generate output image, learning from Task A.
                        help="Fixed text prompt if not using Qwen")
    parser.add_argument("--use_mask", action="store_true",
                        default=False, help="Use mask for generation")
    parser.add_argument("--num_tries", type=int, default=5,
                        help="Number of generation attempts per combination")
    args = parser.parse_args()

    run_evaluation(args)
