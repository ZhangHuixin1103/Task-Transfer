import pickle
from typing import List, Tuple
import random
from accelerate import init_empty_weights
import torch

from model import ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGeneration
import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit("/", 3)[0])
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.finetune import FinetuneSolverBase
from xllmx.data.data_reader import read_general


class ItemProcessor(ItemProcessorBase):
    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        assert training_mode

        if "token" in data_item and "label" in data_item:
            data_item = data_item
        else:
            assert "file" in data_item
            # print(data_item["file"])
            path = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/gaopeng/public/qinqi/MoVQGAN_new' + data_item["file"][1:]
            with open(path, "rb") as f:
                data_item = pickle.load(f)
        tokens = data_item["token"]
        labels = data_item["label"]
        assert len(tokens) == len(labels)
        if tokens[-2] == labels[-2] == 151666 and tokens.count(151666) == 1:
            if random.random() < 0.1:
                tokens = labels = [_ for _ in labels[:-1] if _ != -100]
        return tokens, labels

    def predict_item_token_length(self, data_item: dict) -> int:
        # breakpoint()
        if "token" in data_item:
            return len(data_item["token"])
        elif "len" in data_item:
            return data_item["len"]
        else:
            raise ValueError()


class Solver(FinetuneSolverBase):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=1024, type=int, help="max token length")
        parser.add_argument("--mask_image_logits", default=False)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.05)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "34B"])
        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (ChameleonXLLMXForConditionalGeneration, None):

        # Only instantiate the model on rank0
        # Other ranks will receive the model weights from rank0 during FSDP wrapping (through `sync_module_states`)
        # See https://github.com/pytorch/pytorch/issues/105840
        if self.global_rank == 0:
            model = ChameleonXLLMXForConditionalGeneration.from_pretrained(
                init_from,
                max_position_embeddings=self.args.max_seq_len,
                mask_image_logits=self.args.mask_image_logits,
                dropout=self.args.dropout,
                z_loss_weight=self.args.z_loss_weight,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
        else:
            with init_empty_weights():
                config = ChameleonXLLMXConfig.from_pretrained(
                    init_from,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    torch_dtype=torch.bfloat16,
                )
                model = ChameleonXLLMXForConditionalGeneration(config)
        del model.model.vqmodel

        return model, None

    def _item_processor_func(self) -> ItemProcessorBase:
        return ItemProcessor()

    def _make_and_save_starting_point(self, save_path: str) -> None:
        # 7B model
        config = ChameleonXLLMXConfig.from_pretrained(
                    './config_new.json',
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    torch_dtype=torch.bfloat16,
                )
        model = ChameleonXLLMXForConditionalGeneration(config)
        image_tokens = model.model.vocabulary_mapping.image_tokens
        model.lm_head.weight.data[image_tokens] = torch.zeros_like(model.lm_head.weight.data[image_tokens])

        model.save_pretrained(save_path, max_shard_size="10GB")


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run()
