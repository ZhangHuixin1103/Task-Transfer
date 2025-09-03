import json
import os
import random
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (AutoProcessor, AutoTokenizer,
                          Qwen2_5_VLForConditionalGeneration)
from transformers.utils.quantization_config import BitsAndBytesConfig

# 1. Configuration Section

# Data and output path configuration
DATA_ROOT = Path("./data/tasks")
OUTPUT_DIR = Path("./data/dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# Task configuration
TRAIN_RATIO = 0.3

# Local VLM Model Configuration
LOCAL_MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"

# Set a small number for testing before running the full dataset
NUM_SAMPLES_TO_PROCESS = 10000

# Load the SentenceTransformer model globally
print("Loading the semantic similarity model...")
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

# 2. Load Local VLM Model and Tokenizer Globally
print(
    f"Loading local VLM: {LOCAL_MODEL_ID}. This will take a significant amount of time and VRAM...")
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_ID, trust_remote_code=True)

# Configuration for 4-bit Quantization to save memory
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model with quantization
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    quantization_config=quantization_config,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_ID, trust_remote_code=True)
print("Local VLM loaded successfully to the device.")


# 3. Function Definitions

def sample_task_data(task_name: str, task_path: Path, train_ratio: float, output_dir: Path) -> list[tuple[Path, Path]]:
    """
    Samples data for a single task, creates a training list file, and returns the sampled pairs.
    """
    print(f"Sampling data for task '{task_name}'...")
    input_dir = task_path / "input"
    output_dir_task = task_path / "output"

    if not input_dir.is_dir() or not output_dir_task.is_dir():
        print(
            f"Warning: Input or output directory for task '{task_name}' not found. Skipping.")
        return []

    files = sorted([p.name for p in input_dir.glob('*') if p.is_file()])
    pairs = [(input_dir / f, output_dir_task / f)
             for f in files if (output_dir_task / f).exists()]

    if not pairs:
        print(
            f"Warning: No valid input/output image pairs found in '{task_name}'.")
        return []

    random.shuffle(pairs)
    train_size = int(len(pairs) * train_ratio)
    train_pairs = pairs[:train_size]

    output_txt_path = output_dir / f"train_list_{task_name}.txt"
    with open(output_txt_path, "w") as f:
        for inp_path, out_path in train_pairs:
            f.write(f"{inp_path.name}\n")

    print(f"Sampled {len(train_pairs)} pairs out of {len(pairs)}.")
    print(f"Training list saved to: {output_txt_path}")
    return train_pairs


def query_local_vlm(image_paths: list[Path], prompt: str) -> str:
    """
    Queries the locally hosted Qwen2-VL model with a list of images and a text prompt.
    """
    messages = [{"role": "user", "content": []}]

    for path in image_paths:
        messages[0]["content"].append(
            {"type": "image", "image": f"file:///{str(path.absolute())}"})

    messages[0]["content"].append({"type": "text", "text": prompt})

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True)[0]
        return response.strip()

    except Exception as e:
        print(f"Error during local model generation: {e}")
        return "Error: Failed to generate response from local model."


def load_processed_data(json_path: Path) -> list:
    """
    Loads data from a potentially incomplete JSON file.
    An incomplete file is one that starts with '[' but doesn't end with ']'.
    """
    if not json_path.exists() or os.path.getsize(json_path) == 0:
        return []

    print(f"Found existing data file at {json_path}. Attempting to resume.")
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Handle cases: empty file, file with only '[', or a complete JSON
    if not content or content == '[':
        return []

    # If the file is incomplete (missing closing ']'), we fix it for parsing
    if content.startswith('[') and not content.endswith(']'):
        # Find the last valid '}' to avoid parsing errors from a partially written object
        last_brace_pos = content.rfind('}')
        if last_brace_pos == -1:
            return []  # No complete objects found
        content = content[:last_brace_pos + 1] + \
            ']'  # Truncate and close the array

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Warning: Could not parse existing JSON file. Starting from scratch.")
        return []


def prepare_dataset_file_for_append(json_path: Path):
    """Prepares the JSON file for appending."""
    # If file is new or effectively empty, write the opening bracket.
    if not json_path.exists() or os.path.getsize(json_path) <= 2:
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write("[\n")
        return

    # If the file exists, we need to ensure it's in a ready-to-append state.
    with open(json_path, 'r+', encoding='utf-8') as f:
        f.seek(0, os.SEEK_END)
        if f.tell() < 3: return

        # Read the last few characters to check the state
        f.seek(f.tell() - 3)
        content_end = f.read().strip()

        # Case 1: File was completed. Remove ']' and add a comma.
        if content_end.endswith(']'):
            print("Found a completed JSON file. Removing ']' to append new data.")
            f.seek(f.tell() - (len(content_end) - content_end.rfind(']')))
            f.truncate()
            f.write(",\n")
        # Case 2: File was interrupted after an object. Add a comma.
        elif content_end.endswith('}'):
             f.seek(0, os.SEEK_END)
             f.write(",\n")
        # Case 3: File already ends with a comma, or just '['. Do nothing.


def main():
    """Main execution function to generate the comparative dataset."""
    # Step 1: Load existing data to determine the state.
    output_json_path = OUTPUT_DIR / "train_dataset.json"
    processed_data = load_processed_data(output_json_path)

    all_tasks = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    if len(all_tasks) < 2:
        print("Error: Could not find at least two task directories.")
        return

    if not processed_data:
        # CASE 1: No existing data. This is a fresh start.
        print("No existing data found. Starting a new random task pair.")
        taskA, taskB = random.sample(all_tasks, 2)
        taskA_name, taskB_name = taskA.name, taskB.name
    else:
        # CASE 2: Resuming from existing data. Infer tasks from the LAST entry.
        last_entry = processed_data[-1]
        try:
            taskA_name = Path(last_entry['taskA_input']).parts[0]
            taskB_name = Path(last_entry['taskB_input']).parts[0]
            print(f"Resuming tasks: '{taskA_name}' and '{taskB_name}'.")
        except (KeyError, IndexError):
            print("Error: Could not parse last entry in dataset. Please check the file.")
            return

    taskA_path = DATA_ROOT / taskA_name
    taskB_path = DATA_ROOT / taskB_name

    trainA_pairs = sample_task_data(taskA_name, taskA_path, TRAIN_RATIO, OUTPUT_DIR)
    trainB_pairs = sample_task_data(taskB_name, taskB_path, TRAIN_RATIO, OUTPUT_DIR)
    if not trainA_pairs or not trainB_pairs:
        print("Sampling list for current tasks is empty. Terminating.")
        return

    # Step 2: Calculate what needs to be done.
    processed_combos_set = set()
    for item in processed_data:
        # Only add items for the current task pair to the set
        if item['taskA_input'].startswith(taskA_name) and item['taskB_input'].startswith(taskB_name):
            key = (item['taskA_input'], item['taskA_output'],
                   item['taskB_input'], item['taskB_output'])
            processed_combos_set.add(key)
    num_already_processed = len(processed_combos_set)
    print(f"Found {num_already_processed} existing entries for the current task pair.")

    num_to_generate = NUM_SAMPLES_TO_PROCESS - num_already_processed
    if num_to_generate <= 0:
        print(f"Target of {NUM_SAMPLES_TO_PROCESS} samples for this task pair is already met. Starting a new pair on the next run.")
        # Ensure the file is properly closed with ']'
        with open(output_json_path, "a", encoding='utf-8') as f:
            f.write("\n]")
        return

    all_combos_for_pair = [(*pairA, *pairB)
                           for pairA in trainA_pairs for pairB in trainB_pairs]
    unprocessed_combos = []
    for (a_in, a_out, b_in, b_out) in all_combos_for_pair:
        key = (str(a_in.relative_to(DATA_ROOT)), str(a_out.relative_to(DATA_ROOT)),
               str(b_in.relative_to(DATA_ROOT)), str(b_out.relative_to(DATA_ROOT)))
        if key not in processed_combos_set:
            unprocessed_combos.append((a_in, a_out, b_in, b_out))

    # If the number of available unprocessed combos is less than what we want to generate,
    # just process all of them. Otherwise, take the amount we need.
    if len(unprocessed_combos) < num_to_generate:
        print(f"Warning: Only {len(unprocessed_combos)} new combinations are available, which is less than the target of {num_to_generate}. Processing all available.")
        num_to_generate = len(unprocessed_combos)
    print(f"Need to generate {num_to_generate} new samples.")

    combos_to_process = unprocessed_combos[:num_to_generate]
    if not combos_to_process:
        print("No new combinations to process for this pair, but target not met. Check your data.")
        return

    # Step 3: Execute the generation.
    prepare_dataset_file_for_append(output_json_path)

    with open(output_json_path, "a", encoding='utf-8') as f:
        is_first_new_write = True
        for (a_in, a_out, b_in, b_out) in tqdm(combos_to_process, desc="Generating Descriptions"):
            prompt = (
                "You are an expert in analyzing image processing tasks. Below are two tasks, A and B, each with an input and an output image. "
                "The first two images belong to Task A, and the next two images belong to Task B. "
                "Please analyze and describe the key differences between them. Focus on the target goal, the type of degradation in the input, and the visual changes from input to output."
            )
            description = query_local_vlm([a_in, a_out, b_in, b_out], prompt)

            result = {
                "taskA_input": str(a_in.relative_to(DATA_ROOT)),
                "taskA_output": str(a_out.relative_to(DATA_ROOT)),
                "taskB_input": str(b_in.relative_to(DATA_ROOT)),
                "taskB_output": str(b_out.relative_to(DATA_ROOT)),
                "description": description
            }

            if not is_first_new_write:
                f.write(",\n")

            json.dump(result, f, indent=2, ensure_ascii=False)
            is_first_new_write = False

    print(f"Data generation complete! The dataset is saved to: {output_json_path}")


if __name__ == "__main__":
    main()
