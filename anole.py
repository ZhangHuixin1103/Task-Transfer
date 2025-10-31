import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
from PIL import Image

# -------------------------
# Anole / Chameleon imports
# -------------------------

# Add Anole-specific transformers path
anole_transformers = "/data1/tzz/huixin/Task-Transfer/Anole/transformers/src"
if anole_transformers not in sys.path:
    sys.path.insert(0, anole_transformers)

try:
    from Anole.transformers.src.transformers.models.chameleon import (
        ChameleonConfig, ChameleonForConditionalGenerationWithCFG,
        ChameleonProcessor)
except Exception as e:
    ChameleonInferenceModel = None
    Options = None
    logging.critical(
        "Could not import Anole/Chameleon models. "
        "Make sure this script is run in the correct conda env ('anole_env') "
        "and the path '%s' is correct. Import error: %s", anole_transformers, e
    )
    sys.exit(1)

from transformers import StoppingCriteria, StoppingCriteriaList

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# -------------------------
# Anole / Chameleon integration
# -------------------------

class StopAtSpecificTokenCriteria(StoppingCriteria):
    """Stop generation when a specific token is generated"""

    def __init__(self, stop_token_id, device):
        self.stop_token_id = stop_token_id
        self.device = device

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last generated token is our stop token
        return (input_ids[0, -1] == self.stop_token_id).item()


class InterleavedGenerator:
    """Handles interleaved text-image generation with CFG switching"""

    def __init__(self, model_name: str, device: str = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load model and processor
        self.config = ChameleonConfig.from_pretrained(model_name)
        self.config.attn_implementation = "flash_attention_2"

        self.processor = ChameleonProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"

        self.model = ChameleonForConditionalGenerationWithCFG.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=torch.bfloat16
        ).to(self.device)

        # Get special tokens
        self.boi_token_id = self.config.boi_token_id
        self.eoi_token_id = self.config.eoi_token_id
        self.eos_token_id = self.config.eos_token_id
        self.pad_token_id = 1

        # Image vocabulary for filtering
        self.image_conditioned_allowed = set([i for i in range(4, 8196)]) | {
            self.config.bos_token_id,
            self.boi_token_id,
            self.eoi_token_id,
        }

        # Setup initial CFG (disabled for text)
        self.model.setup_cfg(
            guidance_scale_full=2.0,
            guidance_scale_image=1.2,
            guidance_scale_negative=0.0,
            guidance_scale_original_prompt=5.0,
            config=self.config,
            cfg_config="no"  # Start with no CFG for text
        )

        # Store original prompt tokens for CFG
        self.original_prompt_tokens = None

    def _prepare_cfg_batch(self, token_ids, cfg_type="normal"):
        """
        Prepare batch for CFG by creating multiple conditions
        """
        # Negative prompt for image generation
        negative_prompt = "text in the image, text, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
        negative_tokens = self.processor.tokenizer.encode(
            negative_prompt, add_special_tokens=False)

        batch_token_ids = []

        if cfg_type == "normal":
            # For normal CFG (3 conditions)
            # 1. Full condition
            batch_token_ids.append(token_ids)

            # 2. Unconditional (just BOI)
            batch_token_ids.append([self.boi_token_id])

            # 3. Image-conditioned tokens only
            image_only_tokens = [tok for tok in token_ids if tok in self.image_conditioned_allowed]
            if not image_only_tokens or image_only_tokens[-1] != self.boi_token_id:
                image_only_tokens.append(self.boi_token_id)
            batch_token_ids.append(image_only_tokens)

        elif cfg_type == "obj":
            # For object-focused generation (3 conditions)
            # 1. Full condition
            batch_token_ids.append(token_ids)
            # 2. Unconditional (just BOI)
            batch_token_ids.append([self.boi_token_id])
            # 3. Negative condition
            batch_token_ids.append(negative_tokens + [self.boi_token_id])

        elif cfg_type == "full":
            # For full CFG (5 conditions)
            # 1. Full condition
            batch_token_ids.append(token_ids)
            # 2. Image-conditioned tokens only
            image_only_tokens = [tok for tok in token_ids if tok in self.image_conditioned_allowed]
            if not image_only_tokens or image_only_tokens[-1] != self.boi_token_id:
                image_only_tokens.append(self.boi_token_id)
            batch_token_ids.append(image_only_tokens)
            # 3. Unconditional
            batch_token_ids.append([self.boi_token_id])
            # 4. Negative condition
            batch_token_ids.append(negative_tokens + [self.boi_token_id])
            # 5. Original prompt condition
            if self.original_prompt_tokens:
                orig_tokens = self.original_prompt_tokens.copy()
                if not orig_tokens or orig_tokens[-1] != self.boi_token_id:
                    orig_tokens.append(self.boi_token_id)
                batch_token_ids.append(orig_tokens)
            else:
                batch_token_ids.append([self.boi_token_id])

        # Pad sequences to same length
        max_len = max(len(seq) for seq in batch_token_ids)
        attention_masks = []

        for i, seq in enumerate(batch_token_ids):
            # Pad sequence
            padding_length = max_len - len(seq)
            if padding_length > 0:
                # Pad on the left (since padding_side="left")
                batch_token_ids[i] = [self.pad_token_id] * padding_length + seq
                attention_masks.append([0] * padding_length + [1] * len(seq))
            else:
                attention_masks.append([1] * len(seq))

        # Convert to tensors
        input_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self.device)

        return input_ids, attention_mask

    def generate_interleaved(
        self,
        prompt_tokens: list,
        original_prompt_tokens: list = None,
        max_length: int = 5000,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_images: int = 4,
        cfg_type: str = "normal",
        mode: str = "general"
    ):
        """
        Generate interleaved text and images by alternating between modes
        """
        # Store original prompt tokens for CFG
        self.original_prompt_tokens = original_prompt_tokens if cfg_type == "full" else None

        # Keep track of all generated tokens
        all_tokens = prompt_tokens.copy()

        # Statistics
        num_images_generated = 0
        generation_segments = []

        # Continue generation until we hit EOS or max length
        while len(all_tokens) < max_length and num_images_generated < max_images:
            current_input_ids = torch.tensor([all_tokens], device=self.device)

            # Setup stopping criteria for BOI token
            stop_at_boi = StopAtSpecificTokenCriteria(self.boi_token_id, self.device)
            stop_at_eos = StopAtSpecificTokenCriteria(self.eos_token_id, self.device)

            # Disable CFG for text generation
            self.model.cfg_config = "no"

            if mode == "image_critique" and num_images_generated == 0:
                new_tokens = []
            else:
                text_output = self.model.generate(
                    input_ids=current_input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    stopping_criteria=StoppingCriteriaList([stop_at_boi, stop_at_eos]),
                    multimodal_generation_mode="interleaved-text-image",
                    pad_token_id=self.pad_token_id
                )
                # Extract newly generated tokens
                new_tokens = text_output[0][len(all_tokens):].tolist()

            all_tokens.extend(new_tokens)

            # Check if we stopped at EOS (end of generation)
            if all_tokens[-1] == self.eos_token_id:
                print("Reached end of sequence token.")
                break

            # Check if we stopped at BOI (need to generate image)
            if all_tokens[-1] == self.boi_token_id:
                print(f"Generating image {num_images_generated + 1}...")

                # Generate image with CFG
                actual_cfg_type = cfg_type
                if mode == "object_thoughts":
                    if num_images_generated < 2:
                        actual_cfg_type = "obj"
                    else:
                        actual_cfg_type = "full"

                self.model.cfg_config = actual_cfg_type

                # Prepare CFG batch
                cfg_input_ids, cfg_attention_mask = self._prepare_cfg_batch(
                    all_tokens, cfg_type=actual_cfg_type
                )

                image_output = self.model.generate(
                    input_ids=cfg_input_ids,
                    attention_mask=cfg_attention_mask,
                    max_new_tokens=1026,
                    temperature=temperature,
                    do_sample=True,
                    multimodal_generation_mode="image-only",
                    pad_token_id=self.pad_token_id
                )

                # Extract only the first condition's output
                new_image_tokens = image_output[0][len(cfg_input_ids[0]):].tolist()[:1025]
                all_tokens.extend(new_image_tokens)
                num_images_generated += 1
                self.model.cfg_config = "no"

        return {
            "tokens": all_tokens,
            "num_images": num_images_generated,
            "total_length": len(all_tokens)
        }

    # Helper to decode image
    def split_token_sequence(self, tokens_tensor, boi_token_id, eoi_token_id):
        """Helper to split tokens into text/image segments."""
        tokens = tokens_tensor[0].tolist()
        segments = []
        current_segment = []
        is_image_seg = False

        for token in tokens:
            if token == boi_token_id:
                if current_segment:
                    segments.append(("text_seg", torch.tensor([current_segment], device=self.device)))
                current_segment = [token]
                is_image_seg = True
            elif token == eoi_token_id:
                if is_image_seg:
                    current_segment.append(token)
                    segments.append(("image_seg", torch.tensor([current_segment], device=self.device)))
                    current_segment = []
                    is_image_seg = False
            else:
                if token not in [self.processor.tokenizer.pad_token_id, self.processor.tokenizer.eos_token_id]:
                    current_segment.append(token)

        if current_segment and not is_image_seg:
            segments.append(("text_seg", torch.tensor([current_segment], device=self.device)))

        return segments


def generate_image_with_anole(
    generator: InterleavedGenerator,
    taskA_input_path: str,
    taskA_output_path: str,
    taskB_input_path: str,
    text_prompt: str,
    combo_tmp_dir: str,
    use_mask: bool = False,
    mask_path: Optional[str] = None,
):
    """
    Generate image for Task B using Anole generator.
    This is a MODIFIED version of the function from mask_pro.py.
    It ONLY contains Anole-related code.
    All GrAInS/Qwen logic has been removed.
    """
    os.makedirs(combo_tmp_dir, exist_ok=True)

    # Build Anole prompt_tokens
    generation_prompt = f"Task A input: <image> Task A output: <image> Task B input: <image> {text_prompt} Generate Task B output:"

    images = [Image.open(taskA_input_path).convert("RGB"),
              Image.open(taskA_output_path).convert("RGB"),
              Image.open(taskB_input_path).convert("RGB")]

    if use_mask and mask_path:
        logging.info(f"[Anole Gen] Mask is requested, but Anole integration does not currently use it. "
                     f"Mask path: {mask_path}")

    inputs = generator.processor(generation_prompt, images=images, padding=False, return_tensors="pt",
                                 return_for_text_completion=True).to(generator.device, dtype=torch.bfloat16)

    input_ids = inputs['input_ids']
    if 'pixel_values' in inputs:
        pixel_values = inputs["pixel_values"]
        image_tokens = generator.model.get_image_tokens(pixel_values)
        special_image_mask = input_ids == 8711  # Image token ID
        image_tokens = image_tokens.to(input_ids.device, input_ids.dtype)
        input_ids = input_ids.masked_scatter(special_image_mask, image_tokens)

    prompt_tokens = input_ids[0].tolist() + [generator.boi_token_id]

    original_prompt = f"Generate the output image for Task B based on the example."
    original_tokens = generator.processor.tokenizer.encode(original_prompt, add_special_tokens=False)

    # Generate with Anole
    logging.info("[Anole Gen] Starting Anole image generation...")
    result = generator.generate_interleaved(
        prompt_tokens=prompt_tokens,
        original_prompt_tokens=original_tokens,
        max_length=6144,
        temperature=1.0,
        max_images=1,
        cfg_type="normal",
        mode="general"
    )

    # Extract the generated image
    tokens_tensor = torch.tensor([result["tokens"]], device=generator.device)
    segments = generator.split_token_sequence(tokens_tensor, generator.boi_token_id, generator.eoi_token_id)

    generated_image = None
    for seg_type, seg_tokens in segments[::-1]:
        if seg_type == "image_seg":
            generated_image = generator.model.decode_image(seg_tokens)[0]
            break

    if generated_image:
        gen_image_path = os.path.join(combo_tmp_dir, "generated_anole.png")
        generated_image.save(gen_image_path)
        logging.info(f"[Anole Gen] Saved generated image to {gen_image_path}")
        return gen_image_path
    else:
        logging.warning("[Anole Gen] No generated image found in segments.")
        return None


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anole Image Generation Sub-script")
    parser.add_argument("--input-json", type=str, required=True,
                        help="Path to the input JSON file containing paths and prompt.")
    parser.add_argument("--output-json", type=str, required=True,
                        help="Path to the output JSON file to write the generated image path.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to Anole model")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Temporary directory to save the generated image.")
    # Add data_tasks_dir to resolve relative paths
    parser.add_argument("--data-tasks-dir", type=str, required=True,
                        help="Base directory for task data (e.g., 'data/tasks')")

    args = parser.parse_args()

    # Load inputs from the JSON file
    try:
        with open(args.input_json, 'r') as f:
            inputs = json.load(f)

        taskA_input = inputs['taskA_input']
        taskA_output = inputs['taskA_output']
        taskB_input = inputs['taskB_input']
        text_prompt = inputs['text_prompt']
        use_mask = inputs['use_mask']
        mask_path = inputs['mask_path']

        # Resolve relative paths to absolute paths
        taskA_input_path = os.path.join(args.data_tasks_dir, taskA_input)
        taskA_output_path = os.path.join(args.data_tasks_dir, taskA_output)
        taskB_input_path = os.path.join(args.data_tasks_dir, taskB_input)

    except Exception as e:
        logging.critical(f"Failed to load or parse {args.input_json}: {e}")
        sys.exit(1)

    output_data = {"generated_image_path": None}
    try:
        # Initialize Anole generator
        logging.info(f"Loading Anole model from {args.model_path}...")
        anole_generator = InterleavedGenerator(args.model_path)
        logging.info("Anole model loaded.")

        # Run generation
        gen_path = generate_image_with_anole(
            generator=anole_generator,
            taskA_input_path=taskA_input_path,
            taskA_output_path=taskA_output_path,
            taskB_input_path=taskB_input_path,
            text_prompt=text_prompt,
            combo_tmp_dir=args.save_dir,
            use_mask=use_mask,
            mask_path=mask_path
        )

        output_data["generated_image_path"] = gen_path

    except Exception as e:
        logging.error(f"Anole generation failed: {e}", exc_info=True)
        # We still write to the output file so the main process doesn't hang

    finally:
        # Write the output path (or None on failure) to the output JSON
        try:
            with open(args.output_json, 'w') as f:
                json.dump(output_data, f)
        except Exception as e:
            logging.error(f"Failed to write output JSON {args.output_json}: {e}")
