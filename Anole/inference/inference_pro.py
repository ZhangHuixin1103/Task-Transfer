import json
import os

import torch
from chameleon.inference.chameleon import ChameleonInferenceModel, Options

ckpt_path = "/data1/tzz/huixin/Task-Transfer/.cache/Anole-7b-v0.1/"
MODEL_7B_PATH = ckpt_path + "models/7b"
TOKENIZER_TEXT_PATH = ckpt_path + "tokenizer/text_tokenizer.json"
TOKENIZER_IMAGE_PATH = ckpt_path + "tokenizer/vqgan.ckpt"
TOKENIZER_IMAGE_CFG_PATH = ckpt_path + "tokenizer/vqgan.yaml"


def split_token_sequence(tokens, boi, eoi):
    # Get device and data type from the input tensor
    device = tokens.device
    dtype = tokens.dtype

    tokens = tokens[0]  # [seq] - remove batch dimension
    segments = []
    cur = []
    in_img = False
    for t in tokens:
        token_val = int(t.item())  # Convert torch element to Python int

        if token_val == boi:
            if cur:
                # Convert current text segment (list) to torch.LongTensor [1, seq_len]
                segments.append(("text_seg", torch.tensor(
                    cur, dtype=dtype, device=device).reshape(1, -1)))
                cur = []
            in_img = True
        elif token_val == eoi and in_img:
            # Convert current image segment (list) to torch.LongTensor [1, seq_len]
            segments.append(("image_seg", torch.tensor(
                cur, dtype=dtype, device=device).reshape(1, -1)))
            cur = []
            in_img = False
        else:
            # Append Python int to the current segment list
            cur.append(token_val)

    if cur:
        # Convert any remaining segment to torch.LongTensor
        tensor_segment = torch.tensor(
            cur, dtype=dtype, device=device).reshape(1, -1)
        segments.append(
            ("image_seg" if in_img else "text_seg", tensor_segment))

    return segments


def run_anole(jsonl_path, save_dir="output"):
    model = ChameleonInferenceModel(
        MODEL_7B_PATH,
        TOKENIZER_TEXT_PATH,
        TOKENIZER_IMAGE_CFG_PATH,
        TOKENIZER_IMAGE_PATH,
    )
    os.makedirs(save_dir, exist_ok=True)
    options = Options(txt=True, img=True)
    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            prompt = data["prompt"]
            images = data["images"]

            # Split the prompt around <image> placeholders
            parts = prompt.split("<image>")
            # One more part than input images (for the output <image>)
            expected_parts = len(images) + 1
            assert len(
                parts) == expected_parts, f"Prompt has {len(parts)-1} <image> placeholders, but {len(images)} images provided."

            # Build interleaved batch_prompt_ui
            ui = []
            for i in range(len(images)):
                if parts[i]:  # Only add non-empty text parts
                    ui.append({"type": "text", "value": parts[i]})
                ui.append({"type": "image", "value": f"file:{images[i]}"})

            # Add the final text part, including the output <image> as text (to cue generation)
            final_text = parts[-1]
            if final_text:
                ui.append({"type": "text", "value": "<image>" + final_text})
            else:
                ui.append({"type": "text", "value": "<image>"})

            batch_prompt_ui = [ui]
            print(f"Running sample {idx+1} ...")
            tokens = model.generate(
                batch_prompt_ui=batch_prompt_ui, options=options)
            boi = model.vocab.begin_image
            eoi = model.vocab.end_image
            segments = split_token_sequence(tokens, boi, eoi)
            img_count = 0
            for seg_i, (stype, seg) in enumerate(segments):
                if stype == "image_seg":
                    assert seg.shape[1] == 1024, "Image tokens must be exactly 1024"
                    img = model.decode_image(seg)[0]
                    out_path = os.path.join(
                        save_dir, f"sample{idx+1}_gen{img_count}.png")
                    img.save(out_path)
                    img_count += 1
                    print(f"✅ Image saved: {out_path}")
                else:
                    text_tokens_list = seg.tolist()[0]
                    text = model.decode_text([text_tokens_list])[0]
                    print("TEXT:", text)


if __name__ == "__main__":
    run_anole("prompts_test.jsonl",
              "/data1/tzz/huixin/Task-Transfer/data/output/anole_output")
