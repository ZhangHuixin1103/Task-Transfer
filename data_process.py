import json

# Define input and output file paths for clarity
INPUT_FILE = "data/dataset/train_dataset.json"
OUTPUT_FILE = "data/dataset/converted_dataset.json"

# The core instruction prompt for the user's message.
# Using a constant for this makes the code cleaner and easier to modify.
INSTRUCTION_PROMPT = (
    "You are an expert in analyzing image processing tasks. Below are two vision tasks, A and B.\n"
    "The Picture 1 and 2 belong to Task A, 1 is input and 2 is output; the third image Picture 3 is input of Task B.\n"
    "Please simply describe the input images, focus on the visual changes from input to output, and analyze the key differences between them.\n"
    "Don't give me long descriptions or explanations; keep it concise and to the point.\n"
    "Don't tell me exactly what the tasks are (e.g., denoising, colorization, or shadow removal); instead, use implicit words and highlight how they differ in their objectives and effects.\n"
    "Fit your answer into 3 sentences: 1) input image descriptions (what need to be done); 2) visual changes (what task A and B did); 3) differences of task A and B.\n"
    "I know you can't see output of task B, but you can guess what task it is based on shortcoming of input."
)


def convert_data(raw_data: list) -> list:
    """
    Converts a raw dataset into the Qwen-VL conversation training format.

    Args:
        raw_data (list): A list of dictionaries, where each dictionary
                         contains details for two image processing tasks.
                         Expected fields: 'taskA_input', 'taskA_output',
                         'taskB_input', and 'description'.

    Returns:
        list: The converted dataset in a list of dictionaries,
              formatted for Qwen-VL training.
    """
    converted_dataset = []
    # Loop through each item in the raw dataset.
    for item in raw_data:
        # Collect all three image paths into a list for the 'image' key.
        image_paths = [
            item['taskA_input'],
            item['taskA_output'],
            item['taskB_input'],
        ]

        # Construct the user's message by embedding image paths and the instruction prompt.
        user_message_value = (
            f"Picture 1: <image>\n"
            f"Picture 2: <image>\n"
            f"Picture 3: <image>\n"
            f"{INSTRUCTION_PROMPT}"
        )

        # The assistant's response is the pre-written description from the raw data.
        assistant_message_value = item["description"]

        # Create the conversation turn, as required by the Qwen-VL format.
        conversation = [
            {"from": "human", "value": user_message_value},
            {"from": "gpt", "value": assistant_message_value},
        ]

        # Assemble the final dictionary for the current data sample.
        formatted_sample = {
            "image": image_paths,
            "conversations": conversation
        }
        converted_dataset.append(formatted_sample)

    return converted_dataset


if __name__ == "__main__":
    try:
        # Load the raw data from the specified input file.
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Call the conversion function.
        converted_data = convert_data(raw_data)

        # Save the result to the output file with proper JSON formatting.
        # `indent=2` makes the output file human-readable.
        # `ensure_ascii=False` handles non-ASCII characters correctly.
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)

        # Provide a success message to the user.
        print(f"✅ Conversion successful! {len(converted_data)} samples saved to {OUTPUT_FILE}.")

    except FileNotFoundError:
        # Handle the case where the input file does not exist.
        print(f"❌ Error: The input file '{INPUT_FILE}' was not found.")
    except json.JSONDecodeError:
        # Handle the case where the input file is not a valid JSON.
        print(f"❌ Error: Failed to parse '{INPUT_FILE}'. Please ensure it is a valid JSON file.")
    except Exception as e:
        # Catch any other unexpected errors.
        print(f"❌ An unexpected error occurred: {e}")
