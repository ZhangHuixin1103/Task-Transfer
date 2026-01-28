import argparse
import hashlib
import json
import logging
import os
import sys

import numpy as np
import torch
from google import genai
from google.genai import types
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from VIEScore.paper_implementation.imagen_museum.utils import \
    write_entry_to_json_file

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "model/Lumina"))
sys.path.insert(0, os.path.join(root_dir, "model/Lumina/lumina_mgpt"))
sys.path.insert(0, os.path.join(root_dir, "model/Lumina/lumina_mgpt/generate_examples"))
from inference_solver import FlexARInferenceSolver
movqgan_path = os.path.join(root_dir, "model/Lumina/lumina_mgpt/movqgan")
if not os.path.exists("./movqgan"):
    try:
        os.symlink(movqgan_path, "./movqgan")
        print(f"Created symlink for movqgan from {movqgan_path}")
    except Exception as e:
        print(f"Symlink warning: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Set constants
DATA_TASKS_DIR = "data/tasks"
EVAL_DATASET_JSON = "data/dataset/eval_dataset.json"
OUTPUT_DIR = "data/output/baseline/lumina/output_qwen"
TMP_DIR = "data/tmp/baseline/lumina/output_qwen"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# Qwen Config
BASE_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_MODEL = "qwen-3-vl-4b-instruct"
CHECKPOINT_PATH = "Qwen3-VL/qwen-vl-finetune/output/checkpoint-4875"

# Gemini Config
GEMINI_API_KEY = "sk-2LE9SvYG170QGDDX1ajIUlsuVxt1bqY9nY92BZAKvSZlPWFL"
GEMINI_MODEL_JUDGE = "gemini-2.5-flash-lite" 
BASE_URL = "https://globalai.vip"
API_KEY_HEADER = "api-key"

# Function to generate unique ID
def hashed_id(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:10]

# Function to create Gemini client
def create_gemini_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            base_url=BASE_URL,
            headers={API_KEY_HEADER: GEMINI_API_KEY}
        )
    )

# Function to generate eval dataset
def generate_eval_dataset():
    """Generate new image combinations per task pair not in train data."""
    if os.path.exists(EVAL_DATASET_JSON):
        logging.info(f"Loading existing {EVAL_DATASET_JSON}")
        with open(EVAL_DATASET_JSON, 'r') as f:
            eval_data = json.load(f)
    else:
        logging.error(f"{EVAL_DATASET_JSON} not found. Please ensure dataset is generated.")
        sys.exit(1)

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
        return "This is a visual in-context learning task. The first two images are an input and output of Task A. The third image is the input for Task B. The goal is to perform Task B on the third image and generate output image, learning from Task A."

    elif not use_qwen and fixed_prompt is not None:
        taskA = taskA_input.split('/')[0]
        taskB = taskB_input.split('/')[0]
        prompt = fixed_prompt.replace('[TASK_A_DEGRADATION]', taskA).replace('[TASK_B_DEGRADATION]', taskB)
        logging.info(f"Using fixed prompt:\n{prompt}")
        return prompt

    # Build messages for Qwen
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(DATA_TASKS_DIR, taskA_input)},
                {"type": "image", "image": os.path.join(DATA_TASKS_DIR, taskA_output)},
                {"type": "image", "image": os.path.join(DATA_TASKS_DIR, taskB_input)},
                {
                    "type": "text",
                    "text": "You are an expert in analyzing image processing tasks. Below are two vision tasks, A and B.\nThe Picture 1 and 2 belong to Task A, 1 is input and 2 is output; the third image Picture 3 is input of Task B.\nPlease simply describe the input images, focus on the visual changes from input to output, and analyze the key differences between them.\nDon't give me long descriptions or explanations; keep it concise and to the point.\nDon't tell me exactly what the tasks are (e.g., denoising, colorization, or shadow removal); instead, use implicit words and highlight how they differ in their objectives and effects.\nFit your answer into 3 sentences: 1) input image descriptions (what need to be done); 2) visual changes (what task A and B did); 3) differences of task A and B.\nI know you can't see output of task B, but you can guess what task it is based on the input.",
                },
            ],
        },
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
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
            max_new_tokens=2048,
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            do_sample=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    instruct_text = "This is a visual in-context learning task. The first two images are an input and output of Task A. The third image is the input for Task B. The goal is to perform Task B on the third image and generate output image, learning from Task A."
    Qwen_text = instruct_text + "\nAnalysis: " + output_text
    
    print(f"Generated prompt:\n{Qwen_text}")
    return Qwen_text

# Lumina Generation Function
def generate_image_with_lumina(solver, taskA_input, taskA_output, taskB_input, text_prompt, args):
    """Generate image using Lumina-mGPT-2.0 based on the provided prompt and images."""
    # 1. 准备图片列表
    img_list = [
            Image.open(os.path.join(DATA_TASKS_DIR, p)).convert("RGB") 
            for p in [taskA_input, taskA_output, taskB_input]
    ]

    # 2. 构造 Prompt
    vicl_prompt = (
        f"Instruction: {text_prompt}\n"
        "Example Task A Input: <|image|>\n"
        "Example Task A Output: <|image|>\n"
        "Target Task B Input: <|image|>\n"
        "Target Task B Output:"
    )

    try:
        generated_text, generated_images = solver.generate(
            images=img_list,
            qas=[[vicl_prompt, None]],
            max_gen_len=10240,
            temperature=args.temperature if args.temperature else 1.0,
            logits_processor=solver.create_logits_processor(
                cfg=args.cfg if args.cfg else 3.0, 
                image_top_k=args.top_k if args.top_k else 2000
            ),
        )

        # 3. 返回生成的图片
        if generated_images and len(generated_images) >= 4:
            return generated_images[3]
        elif generated_images and len(generated_images) > 0:
            return generated_images[-1]
        return None
    except Exception as e:
        print(f"Error during Lumina generation: {e}")
        return None

# Evaluation Metrics
def eval_quality(gt_path, gen_path):
    gt_img = Image.open(gt_path).convert("RGB")
    gen_img = Image.open(gen_path).convert("RGB")
    gen_img = gen_img.resize(gt_img.size, Image.BICUBIC)
    gt_np = np.array(gt_img)
    pred_np = np.array(gen_img)

    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=255)
    ssim = structural_similarity(gt_np, pred_np, channel_axis=-1, data_range=255)
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
    client = create_gemini_client()

    viescore = 0.0
    is_verified = False
    tries, max_tries = 0, 2
    tmp_file_path = os.path.join(TMP_DIR, "viescore_log.json")
    uid = hashed_id(taskA_input, taskB_input, gen_path)

    while not is_verified and tries < max_tries:
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL_JUDGE,
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
                logging.warning("Gemini rate limit exceeded. Waiting...")
                import time; time.sleep(5)
                break # Or continue depending on strategy
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

    # 1. Load Qwen Model
    if args.use_qwen_for_prompt:
        logging.info("Loading Qwen model for prompt generation...")
        try:
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
        except Exception as e:
            logging.error(f"Failed to load Qwen: {e}")
            sys.exit(1)
    else:
        prompt_qwen_model = None
        prompt_qwen_processor = None

    # 2. Load Lumina Model
    print(f"Loading Lumina-mGPT-2.0 from {args.model_path}...")
    solver = FlexARInferenceSolver(
        model_path=args.model_path,
        precision="bf16",
        quant=args.quant,
        sjd=args.speculative_jacobi,
    )

    if args.speculative_jacobi:
        from jacobi_utils_static import renew_pipeline_sampler
        print("Use Speculative Jacobi Decoding to accelerate!")
        w, h = 512, 512
        solver = renew_pipeline_sampler(
            solver,
            jacobi_loop_interval_l = 16,
            jacobi_loop_interval_r = ((h // 8) * (w // 8) + h // 8 - 10),
            max_num_new_tokens = 16,
            guidance_scale = args.cfg,
            seed = None,
            multi_token_init_scheme = 'random',
            do_cfg = True,
            image_top_k = args.top_k,
            text_top_k = 10,
            prefix_token_sampler_scheme = 'speculative_jacobi',
        )
    print("Lumina Model Loaded.")

    # 3. Main Loop
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
                    try:
                        entry = json.loads(line)
                        existing_combo_ids.add(entry['combo_id'])
                    except:
                        pass

        with open(log_path, 'a') as log_file:
            entries_to_process = entries[:args.max_samples]

            for entry in entries_to_process:
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

                # Step 1: Check if already done
                if combo_id in existing_combo_ids:
                    logging.info(f"COMPLETE: Skipping combo {combo_id}, metrics already exist.")
                    continue

                if os.path.exists(final_path) and combo_id not in existing_combo_ids:
                    logging.info(f"RESUMING: Found image for {combo_id}, calculating metrics...")
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
                    existing_combo_ids.add(combo_id)
                    continue

                logging.info(f"STARTING: Processing new combo {combo_id}.")

                # Step 2: Generate text prompt
                text_prompt = generate_text_prompt(
                    taskA_input, taskA_output, taskB_input,
                    model=prompt_qwen_model,
                    processor=prompt_qwen_processor,
                    use_qwen=args.use_qwen_for_prompt,
                    fixed_prompt=args.fixed_prompt
                )

                # Step 3: Generate Image
                gen_image = generate_image_with_lumina(
                    solver, taskA_input, taskA_output, taskB_input, text_prompt, args
                )

                if gen_image:
                    gen_image.save(final_path)

                    # Step 4: Evaluate
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
                else:
                    logging.warning(f"Generation failed for {combo_id}")

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

    logging.info("Evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lumina-mGPT Evaluation Pipeline")
    # Lumina Args
    parser.add_argument("--model_path", type=str, default="Alpha-VLLM/Lumina-mGPT-2.0", help="Path to Lumina-mGPT-2.0")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=4096)
    parser.add_argument("--quant", action='store_true', help="Enable quantization")
    parser.add_argument("--speculative_jacobi", action='store_true', help="Enable SJD acceleration")

    # Eval Args
    parser.add_argument("--use_qwen_for_prompt", action="store_true",
                        default=False, help="Use Qwen for generating text prompt")
    parser.add_argument("--fixed_prompt", type=str, default=None,
                        help="Fixed text prompt if not using Qwen")
    parser.add_argument("--max_samples", type=int, default=50)

    args = parser.parse_args()

    run_evaluation(args)
