import os
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor, AutoModel, AutoModelForCausalLM,
    AutoTokenizer, GenerationConfig, LogitsProcessorList,
    PrefixConstrainedLogitsProcessor
)
from emu3.mllm.processing_emu3 import Emu3Processor

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda:0"

CHAT_HUB = "BAAI/Emu3-Chat"
VQ_HUB   = "BAAI/Emu3-VisionTokenizer"
GEN_HUB  = "BAAI/Emu3-Gen"

chat_tokenizer       = AutoTokenizer.from_pretrained(CHAT_HUB, trust_remote_code=True, padding_side="left")
chat_image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
chat_image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="auto", trust_remote_code=True).eval()
chat_processor       = Emu3Processor(chat_image_processor, chat_image_tokenizer, chat_tokenizer)
chat_model           = AutoModelForCausalLM.from_pretrained(
    CHAT_HUB,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval().to(device)

gen_tokenizer = chat_tokenizer
gen_processor = chat_processor
gen_model     = AutoModelForCausalLM.from_pretrained(
    GEN_HUB,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval().to(device)

A_in  = Image.open("../../data/demo/deraining/2.jpg").convert("RGB")
A_out = Image.open("../../data/demo/deraining/2-derain.jpg").convert("RGB")
B_in  = Image.open("../../data/demo/removal/1.png").convert("RGB")

text_prompt = [
    "This is the first image, input of task A",
    "This is the second image, output of task A. You can see how a type of visual interference was removed.",
    "This is the third image, input of task B. Please apply a similar transformation to the third image.",
]

chat_inputs = chat_processor(
    text=text_prompt,
    image=[A_in, A_out, B_in],
    mode='U',
    image_area=chat_model.config.image_area,
    return_tensors="pt",
    padding="longest"
)
for k, v in chat_inputs.items():
    if isinstance(v, torch.Tensor):
        chat_inputs[k] = v.to(device)

with torch.no_grad():
    chat_out = chat_model(
        input_ids=chat_inputs.input_ids,
        attention_mask=chat_inputs.attention_mask,
        return_dict=True,
        output_hidden_states=True,
    )
prefix_embeds = chat_out.hidden_states[-1]  # (1, seq_len, hidden_dim)

gen_texts = ["<generate>"]
gen_inputs = gen_processor(
    text=gen_texts,
    mode='G',
    image_area=gen_model.config.image_area,
    return_tensors="pt",
    padding="longest"
)
gen_input_ids = gen_inputs.input_ids.to(device)
gen_attn_mask = gen_inputs.attention_mask.to(device)

gen_cfg = GenerationConfig(
    use_cache=True,
    eos_token_id=gen_model.config.eos_token_id,
    pad_token_id=gen_model.config.pad_token_id,
    max_new_tokens=4096,
    do_sample=True,
    top_k=2048,
    temperature=0.7,
)

w, h = B_in.size
constrained_fn = gen_processor.build_prefix_constrained_fn(h, w)
logits_proc = LogitsProcessorList([
    PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
])

gen_embeds   = gen_model.get_input_embeddings()(gen_input_ids)  # (1, L, dim)
inputs_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)
attn_mask     = torch.cat([torch.ones(prefix_embeds.size()[:-1], device=device), gen_attn_mask], dim=1)

gen_out = gen_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attn_mask,
    generation_config=gen_cfg,
    logits_processor=logits_proc,
)

B_imgs = gen_processor.decode(gen_out[0])

os.makedirs("../../data/demo/removal", exist_ok=True)
for idx, im in enumerate(B_imgs):
    im.save(f"../../data/demo/removal/output_{idx}.png")

print("✅ Result saved to ../../data/demo/removal/output_*.png")
