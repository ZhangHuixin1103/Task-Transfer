import json

# Define input and output file paths for clarity
INPUT_FILE = "data/dataset/dataset.json"
OUTPUT_FILE = "data/dataset/converted_data.json"
IMAGE_BASE_PATH = "/mnt/data/huixin/Task-Transfer/data/tasks/"

# The core instruction prompt for the user's message.
# Using a constant for this makes the code cleaner and easier to modify.
INSTRUCTION_PROMPT = (
    "You are an expert in analyzing image processing tasks. "
    "Below are two vision tasks, A and B.\n"
    "The Picture 1 and 2 belong to Task A, 1 is input and 2 is output; "
    "the third image Picture 3 is input of Task B.\n"
    "Please analyze and describe the key differences between the two tasks. "
    "Focus on the target goal, the type of degradation in the input, "
    "and the visual changes from input to output.\n"
    "I know you can't see output of task B, but you can guess what task it is "
    "based on shortcoming of input."
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
    # Using enumerate to automatically generate a unique ID for each sample
    for i, item in enumerate(raw_data):
        # Construct the user's message by embedding image paths and the instruction prompt.
        # The f-string syntax is clean and efficient for this purpose.
        user_message_value = (
            f"Picture 1: <img>{IMAGE_BASE_PATH}{item['taskA_input']}</img>\n"
            f"Picture 2: <img>{IMAGE_BASE_PATH}{item['taskA_output']}</img>\n"
            f"Picture 3: <img>{IMAGE_BASE_PATH}{item['taskB_input']}</img>\n"
            f"{INSTRUCTION_PROMPT}"
        )

        # The assistant's response is the pre-written description from the raw data.
        assistant_message_value = item["description"]

        # Create the conversation turn, as required by the Qwen-VL format.
        conversation = [
            {"from": "user", "value": user_message_value},
            {"from": "assistant", "value": assistant_message_value},
        ]

        # Assemble the final dictionary for the current data sample.
        # The 'id' is crucial for distinguishing each training example.
        formatted_sample = {
            "id": f"identity_{i+1}",
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
