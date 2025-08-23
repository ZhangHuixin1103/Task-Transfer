import json
import os
import random
import time
from pathlib import Path

import google.generativeai as genai
from google import genai
from google.genai import types
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Data and output path configuration
DATA_ROOT = Path("./data/tasks")
OUTPUT_DIR = Path("./data/dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# Task and model configuration
TRAIN_RATIO = 0.3
NUM_REPEATS = 10  # Number of API calls per image pair
MODEL_NAME = "gemini-1.5-flash-latest"

# Load the SentenceTransformer model globally
print("Loading the semantic similarity model...")
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")


# --- 2. Function Definitions ---

def sample_task_data(task_name: str, task_path: Path, train_ratio: float, output_dir: Path) -> list[tuple[Path, Path]]:
    """
    Samples data for a single task and writes the list of sampled files to a dedicated .txt file.
    Returns a list of Path objects, not strings.
    """
    print(f"Sampling data for task '{task_name}'...")
    input_dir = task_path / "input"
    output_dir_task = task_path / "output"

    if not input_dir.is_dir() or not output_dir_task.is_dir():
        print(
            f"Warning: Input or output directory for task '{task_name}' not found. Skipping.")
        return []

    # Use Path objects for more robust file operations
    files = sorted([p.name for p in input_dir.glob('*') if p.is_file()])

    pairs = [
        (input_dir / f, output_dir_task / f)
        for f in files if (output_dir_task / f).exists()
    ]

    if not pairs:
        print(f"Warning: No valid input/output image pairs found in '{task_name}'.")
        return []

    random.shuffle(pairs)
    train_size = int(len(pairs) * train_ratio)
    train_pairs = pairs[:train_size]

    # Save a separate sample list for each task for better tracking
    output_txt_path = output_dir / f"train_list_{task_name}.txt"
    with open(output_txt_path, "w") as f:
        for inp_path, out_path in train_pairs:
            # Saving just the filename is cleaner
            f.write(f"{inp_path.name}\n")

    print(f"Sampled {len(train_pairs)} pairs out of {len(pairs)}.")
    print(f"Training list saved to: {output_txt_path}")
    return train_pairs


# Helper function to read image files into the required format.
def read_image_bytes(path: Path) -> types.Part:
    """Reads an image file and returns it as a types.Part object."""
    # Infer MIME type from file extension for better accuracy
    suffix = path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif suffix == ".png":
        mime_type = "image/png"
    else:
        # Default fallback
        mime_type = "image/jpeg"

    with open(path, 'rb') as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime_type)

# MODIFIED FUNCTION
def query_gemini(image_paths: list[Path], prompt: str, num_repeats: int) -> list[str]:
    """
    Queries the Gemini API using the client-based approach from the example file.
    """
    # Initialize the client directly, as shown in your gemini.py file.
    client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
    responses = []

    # Prepare the content for the API request
    try:
        # Load images as byte parts using the helper function
        image_parts = [read_image_bytes(p) for p in image_paths]
        contents = [types.Part(text=prompt)] + image_parts
    except FileNotFoundError as e:
        print(f"Error: Could not load image {e}. Skipping this combination.")
        return []

    for i in range(num_repeats):
        try:
            # Use the client.models.generate_content method
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
            )
            # Extract text from the response parts, as the response might be chunked
            full_text_response = "".join(part.text for part in response.candidates[0].content.parts if part.text)
            if full_text_response:
                responses.append(full_text_response)
        except Exception as e:
            print(f"API call failed (Attempt {i+1}/{num_repeats}): {e}")
            # If an error occurs, wait longer before retrying
            time.sleep(3)
            continue

    return responses


def pick_best_response(responses: list[str]) -> str:
    """
    Selects the most representative response using Centroid Similarity.
    This method is more stable and efficient than pairwise comparisons.
    """
    if not responses:
        return "Error: No valid responses from API."
    if len(responses) == 1:
        return responses[0]

    embeddings = EMBEDDER.encode(responses, convert_to_tensor=True)

    # 1. Calculate the mean of all vectors (the centroid)
    centroid = embeddings.mean(dim=0)

    # 2. Compute the cosine similarity between each vector and the centroid
    similarities = util.cos_sim(embeddings, centroid)

    # 3. Return the response with the highest similarity
    best_idx = similarities.argmax()
    return responses[best_idx].strip()


def main():
    """Main execution function"""
    all_tasks = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    if len(all_tasks) < 2:
        print("Error: Could not find at least two task directories under the data root.")
        return

    taskA_path, taskB_path = all_tasks[:2]
    taskA_name, taskB_name = taskA_path.name, taskB_path.name

    trainA_pairs = sample_task_data(
        taskA_name, taskA_path, TRAIN_RATIO, OUTPUT_DIR)
    trainB_pairs = sample_task_data(
        taskB_name, taskB_path, TRAIN_RATIO, OUTPUT_DIR)

    if not trainA_pairs or not trainB_pairs:
        print("Sampling list for one or both tasks is empty. Terminating program.")
        return

    # A list comprehension is clear for this scale, though itertools.product is more memory-efficient
    combos = [(*pairA, *pairB)
              for pairA in trainA_pairs for pairB in trainB_pairs]
    results = []

    print(f"Found {len(combos)} total image combinations to process. Starting generation...")

    # Use tqdm to create a progress bar
    for (a_in, a_out, b_in, b_out) in tqdm(combos, desc="Generating Descriptions"):
        prompt = (
            "You are given two image processing tasks. For each task, an input and its corresponding output are provided."
            "Task A is shown in the first pair, and Task B in the second."
            "Please analyze and describe the key differences between these two tasks. Focus on dimensions like: "
            "1. **Target Goal**: What is each task trying to achieve (e.g., remove blur, add color)?"
            "2. **Degradation Type**: What kind of problem is present in the input images (e.g., motion blur, noise, low light)?"
            "3. **Visual Characteristics**: Describe the changes in visual properties like sharpness, clarity, and color from input to output."
        )

        responses = query_gemini(
            [a_in, a_out, b_in, b_out], prompt, NUM_REPEATS)
        best_description = pick_best_response(responses)

        results.append({
            "taskA_input": str(a_in.relative_to(DATA_ROOT)),
            "taskA_output": str(a_out.relative_to(DATA_ROOT)),
            "taskB_input": str(b_in.relative_to(DATA_ROOT)),
            "taskB_output": str(b_out.relative_to(DATA_ROOT)),
            "description": best_description
        })

    # Save the final results to a JSON file
    output_json_path = OUTPUT_DIR / "train_dataset.json"
    with open(output_json_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Data generation complete! The dataset is saved to: {output_json_path}")


if __name__ == "__main__":
    main()
