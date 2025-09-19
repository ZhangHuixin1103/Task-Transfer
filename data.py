import argparse
import json
import os
import random
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (AutoProcessor, AutoTokenizer,
                          Qwen2_5_VLForConditionalGeneration)
from transformers.utils.quantization_config import BitsAndBytesConfig

# 1. Configuration

# Data and output path configuration
DATA_ROOT = Path("./data/tasks")
OUTPUT_DIR = Path("./data/dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# Task configuration
TRAIN_RATIO = 0.3

# Local VLM configuration
LOCAL_MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"

# Set a small number for testing before running the full dataset
NUM_SAMPLES_TO_PROCESS = 2000  # 10000

# 2. Load local VLM model
print(f"Loading local VLM: {LOCAL_MODEL_ID}...")

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
processor = AutoProcessor.from_pretrained(LOCAL_MODEL_ID,
                                          trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID,
                                          trust_remote_code=True)
print("Local VLM loaded successfully to the device.")


# 3. Function Definitions

def sample_task_data(task_name: str, task_path: Path, train_ratio: float, output_dir: Path, resume: bool = False) -> list[tuple[Path, Path]]:
    """
    Samples data for a single task, creates a training list file, and returns the sampled pairs.
    """
    input_folder = task_path / "input"
    output_folder = task_path / "output"
    if not input_folder.is_dir() or not output_folder.is_dir():
        print(f"Warning: Input/output dir for task '{task_name}' not found.")
        return []
    output_txt_path = output_dir / f"train_list_{task_name}.txt"

    if resume and output_txt_path.exists():
        # Load from existing data list for resuming
        print(f"Resuming task '{task_name}', loading data from {output_txt_path}.")
        with open(output_txt_path, "r") as f:
            filenames = [line.strip() for line in f if line.strip()]
        if not filenames:
            print(f"Warning: Existing data list for '{task_name}' is empty. Cannot resume.")
            return []

        pairs = [(input_folder / f, output_folder / f)
                 for f in filenames
                 if (input_folder / f).exists() and (output_folder / f).exists()]
        print(f"Loaded {len(pairs)} pairs for resumption.")
        return pairs

    # Otherwise, perform new sampling
    print(f"Sampling data for task '{task_name}'...")
    files = sorted([p.name for p in input_folder.glob('*') if p.is_file()])
    pairs = [(input_folder / f, output_folder / f)
             for f in files if (output_folder / f).exists()]
    if not pairs:
        print(f"Warning: Input/output image pairs for '{task_name}' not found.")
        return []

    random.shuffle(pairs)
    train_size = int(len(pairs) * train_ratio)
    train_pairs = pairs[:train_size]

    with open(output_txt_path, "w") as f:
        for in_path, out_path in train_pairs:
            f.write(f"{in_path.name}\n")

    print(f"Sampled {len(train_pairs)} pairs out of {len(pairs)}.")
    print(f"Training list saved to: {output_txt_path}")
    return train_pairs


def query_local_vlm(image_paths: list[Path], prompt: str) -> str:
    """
    Queries the locally hosted Qwen2-VL model with a list of images and a text prompt.
    """
    messages = [{"role": "user", "content": []}]

    for path in image_paths:
        messages[0]["content"].append({"type": "image",
                                       "image": f"file:///{str(path.absolute())}"})
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

    # If the file is incomplete last time (missing closing ']')
    if content.startswith('[') and not content.endswith(']'):
        # Find the last valid '}' to avoid parsing errors from a partially written object
        last_brace_pos = content.rfind('}')
        if last_brace_pos == -1:
            return []  # No complete objects found
        # Truncate and close the array
        content = content[:last_brace_pos + 1] + ']'

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Warning: Could not parse existing JSON file. Starting from scratch.")
        return []


def update_train_list(json_path: Path, output_dir: Path):
    """Generates accurate .txt manifest files based on the final JSON content."""
    if not json_path.exists():
        return
    print("Updating .txt files from the final JSON dataset...")
    with open(json_path, 'r', encoding='utf-8') as f:
        # We need to handle potentially incomplete JSON
        content = f.read().strip()
        if content.startswith('[') and not content.endswith(']'):
            last_brace_pos = content.rfind('}')
            if last_brace_pos != -1:
                content = content[:last_brace_pos + 1] + ']'
        try:
            final_data = json.loads(content)
        except json.JSONDecodeError:
            print("Could not parse JSON. Skipping.")
            return

    task_files = {}
    for item in final_data:
        task_a = Path(item['taskA_input']).parts[0]
        task_b = Path(item['taskB_input']).parts[0]

        if task_a not in task_files: task_files[task_a] = set()
        if task_b not in task_files: task_files[task_b] = set()

        task_files[task_a].add(Path(item['taskA_input']).name)
        task_files[task_b].add(Path(item['taskB_input']).name)

    for task_name, filenames in task_files.items():
        output_txt_path = output_dir / f"train_list_{task_name}.txt"
        with open(output_txt_path, 'w') as f:
            for filename in sorted(list(filenames)):
                f.write(f"{filename}\n")
        print(f"Updated train list for '{task_name}' with {len(filenames)} entries.")


def prepare_dataset_file_for_append(json_path: Path) -> bool:
    """Prepares the JSON file for appending."""
    # If file is new or effectively empty, write the opening bracket
    if not json_path.exists() or os.path.getsize(json_path) <= 2:
        json_path.parent.mkdir(exist_ok=True, parents=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write('[')
        return True

    # If the file exists, we need to ensure it's in a ready-to-append state
    with open(json_path, 'r+', encoding='utf-8') as f:
        content = f.read().rstrip()

        # Read the last few characters to check the state
        if content.endswith(']'):
            trimmed = content[:-1].rstrip()
        else:
            trimmed = content
        if trimmed.endswith(','):
            trimmed = trimmed[:-1].rstrip()
        if trimmed == '[':
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write('[')
            return True
        # Otherwise, write back the trimmed content (should end with '}')
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(trimmed)
        return False


def main():
    """Main execution function to generate the comparative dataset."""
    parser = argparse.ArgumentParser(
        description="Generate VLM dataset by comparing two specified tasks.")
    parser.add_argument("task_a", type=str,
                        help="Name of the first task folder (e.g., 'deraining').")
    parser.add_argument("task_b", type=str,
                        help="Name of the second task folder (e.g., 'shadow_removal').")
    args = parser.parse_args()

    # Step 1: Load existing data to determine the state.
    output_json_path = OUTPUT_DIR / "train_dataset.json"
    processed_data = load_processed_data(output_json_path)

    all_tasks = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    if len(all_tasks) < 2:
        print("Error: Could not find at least two task directories.")
        return

    # Decide whether to resume or use given args.
    resume_mode = False
    if not processed_data:
        # CASE 1: No existing data. This is a fresh start.
        print("No existing data found. Starting new tasks specified by command.")
        taskA_name, taskB_name = args.task_a, args.task_b
    else:
        # CASE 2: Resuming from existing data. Infer tasks from the LAST entry.
        last_entry = processed_data[-1]
        last_taskA = Path(last_entry['taskA_input']).parts[0]
        last_taskB = Path(last_entry['taskB_input']).parts[0]

        # Count how many samples were completed for the last task pair.
        last_task_processed_count = sum(1 for item in processed_data
                                        if item['taskA_input'].startswith(last_taskA)
                                        and item['taskB_input'].startswith(last_taskB))
        # Calculate the total possible combinations for that last task pair.
        last_trainA_pairs = sample_task_data(last_taskA, DATA_ROOT / last_taskA,
                                             TRAIN_RATIO, OUTPUT_DIR, resume=True)
        last_trainB_pairs = sample_task_data(last_taskB, DATA_ROOT / last_taskB,
                                             TRAIN_RATIO, OUTPUT_DIR, resume=True)
        # The target number for completion is the smaller of the two values.
        target_for_last = min(NUM_SAMPLES_TO_PROCESS,
                              len(last_trainA_pairs) * len(last_trainB_pairs))

        if not last_trainA_pairs or not last_trainB_pairs:
            print(f"Warning: No existing data lists for task pair ('{last_taskA}', '{last_taskB}').")
            taskA_name = args.task_a
            taskB_name = args.task_b
        else:
            if last_task_processed_count < target_for_last:
                # The last task was NOT finished. Force resume.
                print(f"Warning: Incomplete task pair ('{last_taskA}', '{last_taskB}').")
                print("Ignoring command-line arguments.")
                taskA_name = last_taskA
                taskB_name = last_taskB
                resume_mode = True
            else:
                # The last task was finished. Use the new command-line args.
                print(f"Previous task pair ('{last_taskA}', '{last_taskB}') is complete.")
                print("Starting new tasks from command line.")
                taskA_name = args.task_a
                taskB_name = args.task_b

    trainA_pairs = sample_task_data(taskA_name, DATA_ROOT / taskA_name,
                                    TRAIN_RATIO, OUTPUT_DIR, resume=resume_mode)
    trainB_pairs = sample_task_data(taskB_name, DATA_ROOT / taskB_name,
                                    TRAIN_RATIO, OUTPUT_DIR, resume=resume_mode)
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
        print(f"Target of {NUM_SAMPLES_TO_PROCESS} samples for this task pair is already met.")
        print("Starting a new pair on the next run.")
        # Ensure the file is properly closed with ']'
        with open(output_json_path, 'r+', encoding='utf-8') as f:
            if not f.read().rstrip().endswith(']'):
                f.write('\n]')
        # Now update the manifest files with the completed data.
        update_train_list(output_json_path, OUTPUT_DIR)
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
    is_first_new_write = prepare_dataset_file_for_append(output_json_path)

    with open(output_json_path, 'a', encoding='utf-8') as f:
        for (a_in, a_out, b_in, b_out) in tqdm(combos_to_process, desc="Generating Descriptions"):
            prompt = (
                "You are an expert in analyzing image processing tasks. Below are two tasks, A and B, each with an input and an output image."
                "The first two images belong to Task A, and the next two images belong to Task B."
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
                f.write(',\n')
            else:
                f.write('\n')

            json.dump(result, f, indent=2, ensure_ascii=False)
            is_first_new_write = False

    with open(output_json_path, 'a', encoding='utf-8') as f:
        f.write('\n]')

    update_train_list(output_json_path, OUTPUT_DIR)
    print(f"Data generation complete! The dataset is saved to: {output_json_path}")


if __name__ == "__main__":
    main()
