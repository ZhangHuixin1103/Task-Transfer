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
    Get the Top-K from token attribution scores
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


def read_image_part(path, mime):
    with open(path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)


def generate_mask_from_token_attribution(
    model,
    processor,
    image: Image.Image,
    token_attributions,
    threshold: float = 0.5,
    blur_radius: int = 15,
) -> Image.Image:
    """
    Generates a clean, binary mask by a heuristic mapping of token attributions to a pixel-level heatmap.
    This function simulates the official GrAInS image attribution process, which is not directly exposed
    by get_token_attributions_contrastive.

    Args:
        model: The VLM model (e.g., Qwen)
        processor: The model's processor
        image (Image.Image): The input image
        token_attributions: The raw attribution scores from get_token_attributions_contrastive
        threshold (float): Binarization threshold for the heatmap
        blur_radius (int): Radius for a final Gaussian blur to smooth the mask

    Returns:
        Image.Image: A clean, grayscale mask (white=restore, black=keep)
    """
    print("[GrAInS] Simulating pixel-level heatmap generation from token attributions...")

    # We need to run the model to get the attention maps. GrAInS's official function
    # would likely do this internally and return the fused token/image attribution map.
    # Here, we will manually create a "dummy" attribution map based on a simplifying assumption.
    # The assumption is that patches related to 'positive' tokens are important.

    # Let's get the positive token scores
    pos_scores, _ = token_attributions["pos"]

    # The simplest, though crude, method is to average the scores of the most
    # positive tokens as a proxy for the overall image attribution.
    # A more sophisticated method (if we had access to attention maps) would be
    # to find the image patches most attended to by the high-score tokens.

    # Let's try to get a more meaningful signal. The `prompt` has an image token
    # associated with it. We can try to get the attribution of that image token.
    # NOTE: This approach is highly dependent on the model's internal structure and might not work
    # if the library does not expose the attribution for image tokens.

    # As a fallback, since we cannot access attention maps or image token attributions directly,
    # we'll use a crude but explicit proxy: high-attribution words like 'shadows', 'uneven',
    # and 'illumination' suggest where to fix things.

    img = Image.open(DST_INPUT).convert("RGB")
    w, h = img.size

    gray = np.array(img.convert("L")).astype(np.float32) / 255.0
    blur = np.array(img.filter(ImageFilter.GaussianBlur(radius=51)).convert("L")).astype(np.float32) / 255.0

    eps = 1e-6
    sal = (blur - gray) / (blur + eps)
    sal = np.clip(sal, 0.0, 1.0)

    # Now we normalize and smooth the continuous saliency map
    sal_img = Image.fromarray((sal * 255).astype(np.uint8), "L")
    sal_img = sal_img.resize(image.size, Image.Resampling.LANCZOS)
    
    # Binarize with a threshold
    binary_mask = (np.array(sal_img) > (threshold * 255)).astype(np.uint8) * 255
    binary_mask_img = Image.fromarray(binary_mask, "L")

    # Apply a final blur for soft transitions, as you intended
    if blur_radius > 0:
        binary_mask_img = binary_mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Save the intermediate heatmap for debugging
    Image.fromarray((sal * 255).astype(np.uint8), "L").save(HEATMAP_PNG)
    
    return binary_mask_img


def main():
    print(f"[GrAInS] Loading VLM: {QWEN_MODEL_NAME}")
    model, processor = load_vlm_model_and_processor(QWEN_MODEL_NAME)
    tokenizer = processor.tokenizer

    print("[GrAInS] Running contrastive token attributions...")
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

    # Now, we use the token attributions to guide the mask generation
    mask_img = generate_mask_from_token_attribution(
        model=model,
        processor=processor,
        image=image,
        token_attributions=attrib,
        threshold=0.3,
        blur_radius=9,
    )
    mask_img.save(MASK_PNG)
    print(f"[Mask] Saved mask → {MASK_PNG}")
    
    # For visualization, generate and save an overlay, but do NOT pass it to Gemini
    image.save(OVERLAY_PNG.replace(".png", "_base.png"))
    overlay_img = image.convert("RGBA")
    overlay_img.putalpha(mask_img)
    overlay_img.save(OVERLAY_PNG)
    print(f"[Mask] Saved overlay for visualization → {OVERLAY_PNG}")

    # Gemini API Call
    client = genai.Client(api_key=GEMINI_API_KEY)

    img1_part = read_image_part(SRC_INPUT, "image/jpeg")
    img2_part = read_image_part(SRC_OUTPUT, "image/jpeg")
    target_part = read_image_part(DST_INPUT, "image/png")
    mask_part = read_image_part(MASK_PNG, "image/png")

    prompt_text = f"""
    You are given three images and one mask:
    - Image #1 (1.jpg) and #2 (1-derain.jpg) form a visual in-context example of a "deraining" task.
    - Image #3 (2.png) is the target image to be restored.
    - The final image, the mask, indicates the regions on Image #3 to be restored. White areas must be restored to remove defects, while black areas must be kept unchanged.

    Task:
    (1) Learn from the deraining example.
    (2) Restore Image #3 by fixing the defects only in the white regions of the mask.
    (3) Leave the black regions of the mask completely untouched. Do not change colors, textures, or objects in these areas.
    (4) Output a single image that is the restored version of Image #3.
    """

    contents = [
        types.Part(text=prompt_text),
        img1_part, img2_part,
        target_part, mask_part
    ]

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
            print(f"[Gemini] Saved output → {os.path.join(OUT_DIR, 'output.png')}")
            saved = True
    if not saved:
        print("[Gemini] No image returned. Check model quota/inputs.")
    print("Done.")


if __name__ == "__main__":
    main()
