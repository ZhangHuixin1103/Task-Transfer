import os

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ----------------------------
# 1) Load finetuned model + processor
# ----------------------------
BASE_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
CHECKPOINT_PATH = "Qwen2.5-VL/qwen-vl-finetune/output/checkpoint-12000"

# Two common cases:
#   (A) CHECKPOINT_PATH is a LoRA adapter dir (it contains adapter_config.json)
#   (B) CHECKPOINT_PATH is a fully merged model directory
if os.path.exists(os.path.join(CHECKPOINT_PATH, "adapter_config.json")):
    # Case A: load base, then apply LoRA
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype="auto",     # or torch.bfloat16 if your GPU supports bf16
        device_map="auto",      # let HF/accelerate shard the model across devices if needed
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    # Optional: merge LoRA into base for faster & lighter inference
    try:
        model = model.merge_and_unload()
    except Exception:
        pass
else:
    # Case B: already merged weights — load directly
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype="auto",
        device_map="auto",
    )

model.eval()  # inference mode for stable outputs

processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)

# ----------------------------
# 2) Build input (multi-image + one text prompt)
# ----------------------------
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "data/tasks/shadow_removal/input/75-14.png",
                "min_pixels": 12544,
                "max_pixels": 401408,
            },
            {
                "type": "image",
                "image": "data/tasks/shadow_removal/output/75-14.png",
                "min_pixels": 12544,
                "max_pixels": 401408,
            },
            {
                "type": "image",
                "image": "data/tasks/deraining/input/12294.jpg",
                "min_pixels": 12544,
                "max_pixels": 401408,
            },
            {
                "type": "text",
                "text": "You are an expert in analyzing image processing tasks. Below are two vision tasks, A and B.\nThe Picture 1 and 2 belong to Task A, 1 is input and 2 is output; the third image Picture 3 is input of Task B.\nPlease analyze and describe the key differences between the two tasks. Focus on the target goal, the type of degradation in the input, and the visual changes from input to output.\nI know you can't see output of task B, but you can guess what task it is based on shortcoming of input.",
            },
        ],
    },
]

# Prepare the prompt and visual inputs
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
# For multi-image input; video_inputs will be None here
image_inputs, video_inputs = process_vision_info(messages)

# Tokenize and prepare inputs for the model
inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Only move inputs to a single device if the model is NOT sharded.
hf_device_map = getattr(model, "hf_device_map", None)
if not hf_device_map:
    try:
        model_device = next(model.parameters()).device
        inputs = inputs.to(model_device)
    except StopIteration:
        pass  # extremely rare case if model has no parameters

# ----------------------------
# 3) Generate output
# ----------------------------
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

# Trim the prompt part and decode the generated tokens
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

# Print the generated text
print("Generated Text:")
for text in output_text:
    print(text)
