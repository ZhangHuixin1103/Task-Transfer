import argparse
import os
from PIL import Image

from models.pipeline import EmuGenerationPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruct", action='store_true',
                        default=False, help="Load Emu-I")
    parser.add_argument("--ckpt-path", type=str, default="../../.cache/BAAI/Emu/pretrain",
                        help="Path to Emu1 pretrained checkpoint directory")
    parser.add_argument("--output-path", type=str, default="output/output.png")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert args.instruct is False, "Image Generation currently do not support instruct tuning model"

    pipeline = EmuGenerationPipeline.from_pretrained(
        path=args.ckpt_path,
        args=args,
    )
    pipeline = pipeline.bfloat16().cuda()

    # in-context generation
    image_1 = Image.open("../../data/demo/deraining/2.jpg")
    image_2 = Image.open("../../data/demo/deraining/2-derain.jpg")
    image_3 = Image.open("../../data/demo/removal/1.png")
    width, height = image_2.size

    image, _ = pipeline(
        [
            "This is the first image: ",
            image_1,
            "This is the second image: ",
            image_2,
            "This is the third image: ",
            image_3,
            "The first two images, image_1 and image_2, show how a type of visual interference was removed — \
            specifically, elongated streaks caused by external weather. The transformation restores the clean appearance of the scene. \
            The third image suffers from a different but related problem: it contains large, dark patches caused by uneven lighting. \
            Although the form of interference differs, both aim to recover the original look of the scene by eliminating external visual distortions. \
            Please apply a similar transformation to the third image.",
        ],
        height=height,
        width=width,
        guidance_scale=10.,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    image.save(args.output_path)
    print(f"✅ Saved generated result to {args.output_path}")
