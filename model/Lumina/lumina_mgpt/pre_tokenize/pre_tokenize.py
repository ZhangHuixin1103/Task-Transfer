import os
import sys

sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])

from argparse import ArgumentParser
import json
import math
import pickle
from PIL import Image
import random
import pandas as pd

from xllmx.data.data_reader import read_general
from data.convertsation import Conversation
from data.item_processor import FlexARItemProcessor
from data.item_processor import var_center_crop


class ItemProcessor(FlexARItemProcessor):
    def __init__(
        self,
        tokenizer="Alpha-VLLM/Lumina-mGPT-2.0",
        conv_template=Conversation,
        target_size=512,
    ):
        super().__init__(tokenizer, conv_template, target_size)
        print(self.crop_size_list)

    def process_item(self, raw_item, training_mode=False, out_flatten=True):

        # Add custom codes here to convert raw_item to the standard format
        # The standard format contains the "conversations" and "image" keys
        # The data format contains the "image_path" and "prompt" keys

        # ********* <start>  Add your custom codes here *******
        if "image_path" in raw_item:
            image = Image.open(read_general(raw_item["image_path"]))
            img_path = raw_item["image_path"]
        else:
            raise ValueError(f"No 'image_path' key found in {raw_item}, please replace it with your own image path key.")
        
        image = var_center_crop(image, crop_size_list=self.crop_size_list)

        if "prompt" in raw_item:
            caption = raw_item["prompt"]
        else:
            raise ValueError(f"No 'prompt' key found in {raw_item}, please replace it with your own prompt key.")
       
            
            

        if random.random() < 0.9:
            prompt = f"Generate an image of {image.size[0]}x{image.size[1]} according to the following prompt:\n{caption}"  # noqa
        else:
            prompt = f"Generate an image according to the following prompt:\n{caption}"

        conversations = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": "<|image|>"},
        ]
        raw_item["conversations"] = conversations
        # *********  <end>   Add your custom codes here *******

        item = {
            "conversations": raw_item["conversations"],
            "image": img_path,
        }

        return super(ItemProcessor, self).process_item(item, training_mode, out_flatten)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--splits",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--in_filename",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
    )
    parser.add_argument("--target_size", type=int, default=512)
    args = parser.parse_args()

    item_processor = ItemProcessor(target_size=args.target_size)

    if args.in_filename.endswith("jsonl"):
        ori_contents = []
        with open(args.in_filename, 'r') as file:
            for line in file.readlines():
                dic = json.loads(line, strict=False)
                ori_contents.append(dic)
    elif args.in_filename.endswith("json"):
        with open(args.in_filename, 'r') as json_file:
            f = json_file.read()
            ori_contents = json.loads(f)
    else:
        raise ValueError("Input file must be either .json or .jsonl format")


    num = len(ori_contents)

    splits = args.splits
    rank = args.rank
    output_dir = args.out_dir
    save_dir = os.path.join(output_dir, "files")
    os.makedirs(save_dir, exist_ok=True)

    num_per_rank = math.ceil(num / splits)

    try:
        with open(os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt"), "r") as f:
            start_idx = int(f.read()) + 1
        print(f"resume from {start_idx}")
    except:
        start_idx = num_per_rank * rank
        print(f"start from {start_idx}")

    end_idx = min(num_per_rank * (rank + 1), len(ori_contents))
    for i in range(start_idx, end_idx):
        if i % 10 == 0:
            print(f"{i}/{end_idx}")
        record = None
        pkl_path = os.path.join(save_dir, f"{i}.pkl")
        try:
            tokens, labels = item_processor.process_item(ori_contents[i], training_mode=True)
            new_item = {"token": tokens, "label": labels, "id": i}
            with open(pkl_path, "wb") as f:
                pickle.dump(new_item, f)

            record = {"file": pkl_path, "len": len(tokens), "id": i}

        except Exception as e:
            from traceback import format_exc

            print(f"item {i} error: \n{ori_contents[i]}")
            print(format_exc())

        if record is not None:
            with open(os.path.join(output_dir, f"{rank}-of-{splits}-record.jsonl"), "a") as f:
                record_str = json.dumps(record) + "\n"
                f.write(record_str)

        with open(os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt"), "w") as f:
            if i == end_idx - 1:
                f.write("finished")
            else:
                f.write(f"{i}")
