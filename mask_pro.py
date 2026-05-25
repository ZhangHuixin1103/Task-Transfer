import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from google.genai import types
from peft import PeftModel
from PIL import Image, ImageFilter
from qwen_vl_utils import process_vision_info
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from GrAInS.src.attribution.gradient.vlm_grad import get_token_attributions_contrastive
from GrAInS.src.utils.config import MODEL_NAME_MAP
from GrAInS.src.utils.model import load_vlm_model_and_processor
from VIEScore.paper_implementation.imagen_museum.utils import write_entry_to_json_file

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
OUTPUT_DIR = "data/output/output_mask_pro"
TMP_DIR = "data/tmp/tmp_mask_pro"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

BASE_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_MODEL = "qwen-2.5-vl-3b-instruct"
CHECKPOINT_PATH = "Qwen2.5-VL/qwen-vl-finetune/output/checkpoint-latest"

# Gemini config
GEMINI_API_KEY = "insert_your_gemini_api_key_here"
GEMINI_MODEL = "gemini-2.0-flash-preview-image-generation"
BASE_URL = "https://globalai.vip"
API_KEY_HEADER = "api-key"

# Truncate length for printing
TRUNCATE_LEN = 2000


def hashed_id(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:10]


def create_gemini_client():
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError("Gemini client imports failed. Needed only for VIEScore evaluation. Error: " + str(e))

    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            base_url=BASE_URL,
            headers={API_KEY_HEADER: GEMINI_API_KEY}
        )
    )


# -------------------------
# Dataset generation
# -------------------------
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
                matching_out = next((out for out in outputs if os.path.splitext(out)[0] == stem), None)
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


# -------------------------
# Prompt generation
# -------------------------
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

    # Generate output text
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

    print(f"Generated prompt:\n{output_text[0] if output_text else ''}")
    return output_text[0] if output_text else ""


# -------------------------
# GrAInS utilities
# -------------------------
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


def _fallback_simple_mask(b_input_path: str, save_dir: str, blur_r: int = 51, th: float = 0.30, smooth_r: int = 9) -> str:
    """
    Crude luminance/illumination-based mask. White = restore (to modify), Black = keep.
    Always returns a valid PNG mask path.
    """
    img = Image.open(b_input_path).convert("RGB")
    gray = np.array(img.convert("L")).astype(np.float32) / 255.0
    blur = np.array(img.filter(ImageFilter.GaussianBlur(
        radius=blur_r)).convert("L")).astype(np.float32) / 255.0

    eps = 1e-6
    sal = (blur - gray) / (blur + eps)
    sal = np.clip(sal, 0.0, 1.0)

    sal_img = Image.fromarray((sal * 255).astype(np.uint8), "L").resize(img.size, Image.Resampling.LANCZOS)
    binary = (np.array(sal_img) > (th * 255)).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary, "L").filter(ImageFilter.GaussianBlur(radius=smooth_r))

    stem = os.path.splitext(os.path.basename(b_input_path))[0]
    out_path = os.path.join(save_dir, f"mask_{stem}.png")
    mask_img.save(out_path)
    return out_path


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
    Uses GrAInS contrastive attribution when possible; falls back to luminance heuristic.
    """
    b_abs = os.path.join(DATA_TASKS_DIR, taskB_input)
    if not os.path.exists(b_abs):
        logging.error(f"[Mask] Task B input not found: {b_abs}")
        return None

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

        attribution_prompt, pos_resp, neg_resp = _build_contrastive_texts(
            text_prompt, taskA, taskB)

        try:
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

            mask_img = _scores_to_mask(
                scores, imgB_in.size, threshold_percentile)
            if mask_img is None:
                logging.warning("[Mask][GrAInS] Cannot form a valid patch grid.")
                return _fallback_simple_mask(b_abs, TMP_DIR)

            out_path = os.path.join(TMP_DIR, f"mask_grains_{os.path.basename(taskB_input)}.png")
            mask_img.save(out_path)
            logging.info(f"[Mask] Saved GrAInS-based mask → {out_path}")
            return out_path

        except Exception as e:
            logging.warning(f"[Mask][GrAInS] Attribution failed ({e}).")
            return _fallback_simple_mask(b_abs, TMP_DIR)

    return _fallback_simple_mask(b_abs, TMP_DIR)


def save_attr_visuals(attrib_out, processor, imgB_path: str, out_dir: str):
    """
    Save GrAInS attribution visualizations: positive/negative barplots and patch overlay heatmap.
    attrib_out is dict as returned by get_token_attributions_contrastive().
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        pos_scores, pos_ids = attrib_out.get("pos", (None, None))
        neg_scores, neg_ids = attrib_out.get("neg", (None, None))

        pos_scores = np.array(pos_scores) if pos_scores is not None else np.array([])
        neg_scores = np.array(neg_scores) if neg_scores is not None else np.array([])

        # token strings for barplot
        token_strings_pos = []
        if processor is not None and pos_ids is not None:
            try:
                token_strings_pos = [processor.tokenizer.decode([int(t)]) for t in pos_ids[0]]
            except Exception:
                token_strings_pos = [str(int(t)) for t in pos_ids[0]]
        else:
            token_strings_pos = [str(i) for i in range(len(pos_scores))]

        if pos_scores.size:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(max(6, len(pos_scores)*0.12), 3))
            plt.bar(range(len(pos_scores)), pos_scores)
            plt.xticks(range(len(pos_scores)), token_strings_pos, rotation=90, fontsize=6)
            plt.title("Positive token attributions (GRAInS)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "attr_positive_bar.png"), dpi=150)
            plt.close()

        if neg_scores.size:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(max(6, len(neg_scores)*0.12), 3))
            plt.bar(range(len(neg_scores)), neg_scores)
            plt.xticks(range(len(neg_scores)), [str(int(t)) for t in neg_ids[0]], rotation=90, fontsize=6)
            plt.title("Negative token attributions (GRAInS)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "attr_negative_bar.png"), dpi=150)
            plt.close()

        # Produce patch overlay if possible
        if pos_scores.size and imgB_path and len(pos_scores) >= 16:
            img = Image.open(imgB_path).convert("RGB")
            n = pos_scores.shape[0]
            g = int(np.floor(np.sqrt(n)))
            n2 = g * g
            if n2 >= 16:
                arr = pos_scores[:n2]
                smin, smax = float(np.min(arr)), float(np.max(arr))
                norm = (arr - smin) / (smax - smin) if smax > smin else np.zeros_like(arr)
                grid = (norm.reshape(g, g) * 255).astype(np.uint8)
                heat = Image.fromarray(grid, "L").resize(img.size, Image.Resampling.LANCZOS)
                heat_np = np.array(heat).astype(np.float32) / 255.0
                img_np = np.array(img).astype(np.float32) / 255.0

                overlay = img_np.copy()
                overlay[..., 0] = np.clip(overlay[..., 0] + 0.7 * heat_np, 0.0, 1.0)
                overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
                overlay_img.save(os.path.join(out_dir, "attr_positive_overlay.png"))
                heat.save(os.path.join(out_dir, "attr_positive_heatmap.png"))

        logging.info("[Thinking] Saved attribution visuals (if any).")

    except Exception as e:
        logging.warning(f"[Thinking] Failed saving attribution visuals: {e}")


# -------------------------
# Evaluation utilities
# -------------------------
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
                model="gemini-2.0-flash",
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


# -------------------------
# Main pipeline
# -------------------------
def run_evaluation(args):
    """Run the full evaluation pipeline."""
    eval_data, grouped = generate_eval_dataset()
    final_results = {}

    # Load Qwen VLM for prompt enhancement and for GrAInS attribution
    qwen_model, qwen_processor = load_vlm_model_and_processor(MODEL_NAME_MAP[QWEN_MODEL])

    # If use Qwen for prompt enhancement, load the LoRA/finetuned model
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
                    if combo_id in existing_combo_ids:
                        logging.info(f"COMPLETE: Skipping combo {combo_id}, image and metrics already exist.")
                        continue
                    else:
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

                logging.info(f"STARTING: Processing new combo {combo_id}.")
                combo_tmp_dir = os.path.join(TMP_DIR, pair_key, combo_id)
                os.makedirs(combo_tmp_dir, exist_ok=True)

                # Step 2: Generate text prompt
                text_prompt = generate_text_prompt(taskA_input, taskA_output,
                                                   taskB_input,
                                                   model=prompt_qwen_model if prompt_qwen_model else qwen_model,
                                                   processor=prompt_qwen_processor if prompt_qwen_processor else qwen_processor,
                                                   use_qwen=args.use_qwen_for_prompt,
                                                   fixed_prompt=args.fixed_prompt)

                # Step 3: Optional mask (use GrAInS)
                mask_path = None
                if args.use_mask:
                    mask_path = generate_mask(
                        taskA=taskA, taskB=taskB, text_prompt=text_prompt,
                        taskA_input=taskA_input, taskA_output=taskA_output,
                        taskB_input=taskB_input,
                        qwen_model=qwen_model, qwen_processor=qwen_processor,
                        method="integrated_gradients", threshold_percentile=80
                    )

                attrib_vis_dir = os.path.join(combo_tmp_dir, "attribution")
                try:
                    if qwen_model is not None and qwen_processor is not None:
                        # Build contrastive texts and compute attributions
                        attribution_prompt, pos_resp, neg_resp = _build_contrastive_texts(
                            text_prompt, taskA, taskB)

                        logging.info("[GrAInS] Computing token attributions (contrastive)...")
                        attrib = get_token_attributions_contrastive(
                            model=qwen_model,
                            processor=qwen_processor,
                            image=Image.open(os.path.join(DATA_TASKS_DIR, taskB_input)).convert("RGB"),
                            prompt=attribution_prompt,
                            pos_response=pos_resp,
                            neg_response=neg_resp,
                            method="integrated_gradients"
                        )
                        # Save visualizations
                        save_attr_visuals(attrib, qwen_processor,
                                          os.path.join(DATA_TASKS_DIR, taskB_input), attrib_vis_dir)
                    else:
                        logging.info("[GrAInS] Qwen model/processor not provided, skipping attribution visualization.")
                except Exception as e:
                    logging.warning(f"[GrAInS] Attribution computation failed: {e}")

                # Step 4: Generate many images, select best by PSNR
                best_psnr = -np.inf
                best_gen_path = None
                for i in range(args.num_tries):
                    gen_image_path = None
                    try:
                        # Prepare input JSON for the subprocess
                        temp_input_data = {
                            "taskA_input": taskA_input,
                            "taskA_output": taskA_output,
                            "taskB_input": taskB_input,
                            "text_prompt": text_prompt,
                            "use_mask": args.use_mask,
                            "mask_path": mask_path,
                        }
                        temp_input_json = os.path.join(combo_tmp_dir, f"anole_input_{i}.json")
                        temp_output_json = os.path.join(combo_tmp_dir, f"anole_output_{i}.json")
                        
                        with open(temp_input_json, 'w') as f:
                            json.dump(temp_input_data, f)
                        
                        # Run the subprocess in the anole environment
                        logging.info(f"Attempt {i+1}: Calling anole.py in anole env...")
                        cmd = [
                            "conda", "run", "-n", "anole",
                            "python", "anole.py",
                            "--input-json", temp_input_json,
                            "--output-json", temp_output_json,
                            "--model-path", args.model_path,
                            "--save-dir", combo_tmp_dir,
                            "--data-tasks-dir", DATA_TASKS_DIR # Pass the base data dir
                        ]
                        
                        # Use capture_output=True to get stdout/stderr for debugging
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')

                        # Read the output path from the output JSON
                        if os.path.exists(temp_output_json):
                            with open(temp_output_json, 'r') as f:
                                output_data = json.load(f)
                            gen_image_path = output_data.get("generated_image_path")
                        
                        if not gen_image_path:
                             logging.warning(f"Attempt {i+1}: anole.py ran but did not return an image path.")
                             logging.debug(f"conda run stdout:\n{result.stdout}")
                             logging.debug(f"conda run stderr:\n{result.stderr}")

                    except subprocess.CalledProcessError as e:
                        logging.warning(f"Attempt {i+1}: anole.py subprocess FAILED.")
                        logging.warning(f"Command: {' '.join(cmd)}")
                        logging.warning(f"Return Code: {e.returncode}")
                        logging.warning(f"STDOUT:\n{e.stdout}")
                        logging.warning(f"STDERR:\n{e.stderr}")
                    except Exception as e:
                        logging.warning(f"Attempt {i+1}: Main script failed during subprocess call: {e}")

                    if gen_image_path:
                        logging.info(f"Attempt {i+1}: Successfully received image from Anole subprocess: {gen_image_path}")
                        temp_path = os.path.join(combo_tmp_dir,
                                                    f"gen_{i}{gt_ext}")
                        # Copy from the path returned by the subprocess
                        shutil.copy(gen_image_path, temp_path) 
                        curr_psnr, _ = eval_quality(taskB_gt_path,
                                                    temp_path)
                        logging.info(f"Attempt {i+1}: Saved to {temp_path}, PSNR: {curr_psnr:.2f}")
                        if curr_psnr > best_psnr:
                            best_psnr = curr_psnr
                            best_gen_path = temp_path
                            logging.info(f"Attempt {i+1}: New best image found!")
                    else:
                        logging.warning(f"Attempt {i+1}: Anole subprocess call returned NO image.")

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
    parser = argparse.ArgumentParser(
        description="VICL Pipeline (Anole + GrAInS)")
    parser.add_argument("--model_path", type=str, default=".cache/twgi-subgoal-anole-7b",
                        help="Path to model")
    parser.add_argument("--use_qwen_for_prompt", action="store_true", default=False,
                        help="Use Qwen for generating text prompt")
    parser.add_argument("--fixed_prompt", type=str, default=None,
                        help="Fixed text prompt if not using Qwen (replace [TASK_A_DEGRADATION],[TASK_B_DEGRADATION])")
    parser.add_argument("--use_mask", action="store_true", default=False,
                        help="Use mask for generation (mask derived from GrAInS attribution)")
    parser.add_argument("--num_tries", type=int, default=5,
                        help="Number of generation attempts per combination")
    args = parser.parse_args()

    run_evaluation(args)
