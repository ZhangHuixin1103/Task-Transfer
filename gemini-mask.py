import os
import sys
from io import BytesIO
from typing import List, Tuple

import numpy as np
from google import genai
from google.genai import types
from PIL import Image, ImageFilter

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from GrAInS.src.attribution.gradient.vlm_grad import get_token_attributions_contrastive
from GrAInS.src.utils.config import MODEL_NAME_MAP
from GrAInS.src.utils.model import load_vlm_model_and_processor

GEMINI_API_KEY = "AIzaSyA0UE4rh5PCyw_HEmHDeZ3aEVAx85TfmGA"
GEMINI_MODEL = "gemini-2.0-flash-preview-image-generation"
QWEN_MODEL = "qwen-2.5-vl-7b-instruct"
QWEN_MODEL_NAME = MODEL_NAME_MAP[QWEN_MODEL]

SRC_INPUT = "./data/demo/deraining/1.jpg"
SRC_OUTPUT = "./data/demo/deraining/1-derain.jpg"
DST_INPUT = "./data/demo/removal/2.png"

CONTRAST_PROMPT = "Identify and correct low-frequency illumination degradations (shadow-like, spatially coherent)."
POS_RESPONSE = "Shadows and uneven illumination are corrected smoothly and consistently."
NEG_RESPONSE = "Shadows remain and the illumination is not corrected."

OUT_DIR = "./data/demo/tmp"
os.makedirs(OUT_DIR, exist_ok=True)

TOKENS_TXT = os.path.join(OUT_DIR, "pos_neg_tokens.txt")
HEATMAP_PNG = os.path.join(OUT_DIR, "2_heatmap.png")
MASK_PNG = os.path.join(OUT_DIR, "2_mask.png")
OVERLAY_PNG = os.path.join(OUT_DIR, "2_overlay.png")


def decode_top_tokens(tokenizer, input_ids, scores: np.ndarray, top_k: int = 15, mode: str = "pos") -> List[Tuple[str, float, int]]:
    """
    从 token attribution 分数中取 Top-K 并 decode 成字符串（用于在 Gemini 提示里加入“语义引导”）。
    mode: "pos" 取分数高的；"neg" 取分数低的；"abs" 取绝对值高的。
    """
    ids = input_ids[0].tolist()
    if mode == "pos":
        idxs = list(np.argsort(scores)[::-1])[:top_k]
    elif mode == "neg":
        idxs = list(np.argsort(scores))[:top_k]
    else:
        idxs = list(np.argsort(np.abs(scores))[::-1])[:top_k]

    out = []
    for i in idxs:
        tok = tokenizer.decode([ids[i]], skip_special_tokens=True).strip()
        if tok:
            out.append((tok, float(scores[i]), i))
    return out


def save_tokens_text(path, pos_list, neg_list):
    with open(path, "w", encoding="utf-8") as f:
        f.write("[Top Positive Tokens]\n")
        for t, s, i in pos_list:
            f.write(f"{i:4d}\t{s:+.4f}\t{t}\n")
        f.write("\n[Top Negative Tokens]\n")
        for t, s, i in neg_list:
            f.write(f"{i:4d}\t{s:+.4f}\t{t}\n")


def make_lowfreq_mask_and_overlay(
    img_path: str,
    blur_radius: int = 31,
    threshold: float = 0.10,
    soften_radius: int = 9,
    out_mask_path: str = MASK_PNG,
    out_overlay_path: str = OVERLAY_PNG,
) -> Tuple[str, str]:
    """
    生成一个掩码（白=需要修复，黑=保持）。
    这一步是空间引导：若你将来有 GrAInS 的像素级热力图，直接替换本函数输出即可。
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    gray = np.array(img.convert("L")).astype(np.float32) / 255.0
    blur = np.array(img.filter(ImageFilter.GaussianBlur(
        radius=blur_radius)).convert("L")).astype(np.float32) / 255.0

    eps = 1e-6
    sal = (blur - gray) / (blur + eps)
    sal = np.clip(sal, 0.0, 1.0)

    mask = (sal > threshold).astype(np.float32)
    if soften_radius > 0:
        mimg = Image.fromarray((mask * 255).astype(np.uint8),
                               mode="L").filter(ImageFilter.GaussianBlur(radius=soften_radius))
        mask = np.array(mimg).astype(np.float32) / 255.0

    Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8),
                    mode="L").save(out_mask_path)

    base_rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (255, 0, 0, 0))
    alpha = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    overlay.putalpha(Image.fromarray(alpha, mode="L"))
    Image.alpha_composite(base_rgba, overlay).save(out_overlay_path)

    return out_mask_path, out_overlay_path


def read_image_part(path, mime):
    with open(path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)


def main():
    print(f"[GrAInS] Loading VLM: {QWEN_MODEL_NAME}")
    model, processor = load_vlm_model_and_processor(QWEN_MODEL_NAME)  # 官方函数
    tokenizer = processor.tokenizer

    print("[GrAInS] Running contrastive token attributions (official)...")
    image = Image.open(DST_INPUT).convert("RGB")
    attrib = get_token_attributions_contrastive(
        model=model,
        processor=processor,
        image=image,
        prompt=CONTRAST_PROMPT,
        pos_response=POS_RESPONSE,
        neg_response=NEG_RESPONSE,
        method="integrated_gradients",
    )

    pos_scores, pos_ids = attrib["pos"]
    neg_scores, neg_ids = attrib["neg"]
    top_pos = decode_top_tokens(
        tokenizer, pos_ids, pos_scores, top_k=15, mode="pos")
    top_neg = decode_top_tokens(
        tokenizer, neg_ids, neg_scores, top_k=15, mode="neg")
    save_tokens_text(TOKENS_TXT, top_pos, top_neg)
    print(f"[GrAInS] Saved attribution tokens → {TOKENS_TXT}")

    mask_path, overlay_path = make_lowfreq_mask_and_overlay(DST_INPUT)
    print(f"[Mask] Saved mask: {mask_path}")
    print(f"[Mask] Saved overlay: {overlay_path}")
    heatmap_path = HEATMAP_PNG if os.path.exists(HEATMAP_PNG) else None

    client = genai.Client(api_key=GEMINI_API_KEY)

    img1_part = read_image_part(SRC_INPUT, "image/jpeg")
    img2_part = read_image_part(SRC_OUTPUT, "image/jpeg")
    target_part = read_image_part(DST_INPUT, "image/png")
    mask_part = read_image_part(mask_path, "image/png")
    overlay_part = read_image_part(overlay_path, "image/png")

    pos_vocab = ", ".join([t for t, _, _ in top_pos[:10]]
                          ) or "illumination, shadow, dark region"
    neg_vocab = ", ".join([t for t, _, _ in top_neg[:10]]
                          ) or "keep, unchanged, unmodified"

    prompt_text = f"""
You are given 3 input images and 2 guidance images:
- Image #1 and #2 form a visual in-context example of *deraining* (remove localized high-frequency degradations).
- Image #3 is the target image containing *low-frequency illumination degradations* (shadow-like, spatially coherent darkening).
- Image #4 is an **attention mask**: WHITE = regions that should be corrected; BLACK = regions that must remain unchanged.
- Image #5 is a visualization overlay for #4.

Semantic steering (from an external attribution model):
- Positive tokens to guide restoration: {pos_vocab}
- Negative tokens indicating what to avoid or keep unchanged: {neg_vocab}

Task:
(1) Learn by analogy from (#1 -> #2) and apply a *shadow-removal-like* restoration to Image #3,
    focusing ONLY on white regions in Image #4.
(2) Keep textures, edges, colors and structures in BLACK regions strictly unchanged.
(3) Avoid hallucinating new objects or changing geometry; do not sharpen or denoise non-white regions.
(4) Output a single image that looks like #3 but with low-frequency illumination degradations corrected in WHITE regions.

Important details:
- Treat the problem as restoring consistent lighting, not deraining per se.
- Prefer smooth illumination transitions; avoid ringing or over-bright patches.
- Maintain global color balance and scene consistency.
"""

    contents = [types.Part(text=prompt_text), img1_part,
                img2_part, target_part, mask_part, overlay_part]
    if heatmap_path:
        contents.append(read_image_part(heatmap_path, "image/png"))

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]),
    )

    saved = False
    for part in resp.candidates[0].content.parts:
        if getattr(part, "text", None):
            print(part.text)
        elif getattr(part, "inline_data", None):
            out = Image.open(BytesIO(part.inline_data.data))
            out.save(os.path.join(OUT_DIR, "output.png"))
            print(
                f"[Gemini] Saved output → {os.path.join(OUT_DIR, 'output.png')}")
            saved = True
    if not saved:
        print("[Gemini] No image returned. Check model quota/inputs.")
    print("Done.")


if __name__ == "__main__":
    main()
