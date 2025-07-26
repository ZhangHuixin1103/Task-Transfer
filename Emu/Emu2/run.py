import os
import torch
from PIL import Image
from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor

device = "cuda:0"
cache_path = "/mnt/data/huixin/Task-Transfer/.cache/BAAI/Emu2-Gen"

multimodal_encoder = AutoModelForCausalLM.from_pretrained(
    os.path.join(cache_path, "multimodal_encoder"),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16",
)
tokenizer = AutoTokenizer.from_pretrained(os.path.join(cache_path, "tokenizer"))

pipe = DiffusionPipeline.from_pretrained(
    cache_path,
    custom_pipeline="pipeline_emu2_gen",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16",
    multimodal_encoder=multimodal_encoder,
    tokenizer=tokenizer,
)
pipe.enable_attention_slicing()
pipe.to(device)
print("✅ Emu2 pipeline loaded on", device)

A_in  = Image.open("../../data/demo/deraining/2.jpg").convert("RGB")
A_out = Image.open("../../data/demo/deraining/2-derain.jpg").convert("RGB")
B_in  = Image.open("../../data/demo/removal/1.png").convert("RGB")

prompt = [
    "<grounding>",
    "The first two images show how a type of visual interference was removed — "
    "specifically, elongated streaks caused by external weather. "
    "The transformation result — the second image — restores the clean appearance of the scene. "
    "Now, please apply a similar transformation to the third image.",
    "<phrase>the first image</phrase>",
    A_in,
    "<phrase>the second image</phrase>",
    A_out,
    "<phrase>the third image</phrase>",
    B_in,
]

print("🚀 Running Emu2 Gen for visual in-context learning...")
outputs = pipe(prompt)

result_img = outputs.images[0]
out_path = "../../data/demo/removal/output.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
result_img.save(out_path)
print(f"✅ Generated Task B result saved to: {out_path}")
