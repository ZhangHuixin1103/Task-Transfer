"""
Reference:
    https://github.com/tyshiwo1/Accelerating-T2I-AR-with-SJD/blob/main/scheduler/jacobi_iteration_lumina_mgpt.py
"""

from functools import cached_property
import math
from typing import Optional, Tuple, Union, List, Dict, Sequence, Any
import random
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
# from transformers.utils import (
#     add_code_sample_docstrings,
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     is_flash_attn_2_available,
#     is_flash_attn_greater_or_equal_2_10,
#     logging,
#     replace_return_docstrings,
# )

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
import json
import copy
import numpy as np

from transformers import GenerationConfig, DynamicCache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.utils import ModelOutput, is_torchdynamo_compiling
from transformers.generation.utils import (
    GenerateNonBeamOutput, 
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList#, LogitsWarper

from logit_processor_3dim_static import (
    MultiTokensVLLogitsProcessor, 
    MultiTokensInterleavedTopKLogitsWarper,
    TopnsigmaLogitsWarper3d,
    get_double_cfg_input_ids, 
    gather_from_split_tensors,
)

from absl import logging
import time


from sjds import SpeculativeSampler

from static_utils import AuxiliarySuffixStaticCausalAttentionBlockMask, PointerStaticCache


def debug_attn_mask(attention_mask, name='emu3attention_mask.png'):
    from matplotlib import pyplot as plt
    while len(attention_mask.shape) > 2:
        attention_mask = attention_mask[0]
    
    attention_mask = attention_mask.exp()

    attention_mask = attention_mask.float().cpu().numpy()
    plt.imshow(attention_mask, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Attention Mask')
    plt.savefig(name)
    plt.close()

def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def delete_false_key_value(
    self,
    num_of_false_tokens,
):
    for layer_idx in range(len(self.key_cache)):
        self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]

def postprocess_cfg_decode(
    model_inputs,
    cfg_half_name_list=['inputs_embeds', 'input_ids', 'pixel_values', ],
):
    # conditional_logits, unconditional_logits = logits.chunk(2, dim=0)

    # logits = guidance_scale * (conditional_logits - unconditional_logits) + unconditional_logits

    cfg_half_name_list = cfg_half_name_list
    def cfg_half(x):
        return x[:x.shape[0]//2]
    
    for name in cfg_half_name_list:
        if (name in model_inputs) and (model_inputs[name] is not None):
            model_inputs[name] = cfg_half(model_inputs[name])
    
    return model_inputs

def check_is_force_no_cfg(
    input_ids, 
    image_start_token_id=None, 
    image_end_token_id=None, 
    guidance_scale=3., 
    do_cfg=True,
    non_cfg_start_steps=0,
    current_step=0,
):
    if (image_start_token_id is None) or (image_end_token_id is None):
        return False
    
    num_image_start_tokens = (input_ids[0] == image_start_token_id).sum()
    num_image_end_tokens = (input_ids[0] == image_end_token_id).sum()

    if num_image_start_tokens == num_image_end_tokens:
        return True
    elif current_step < non_cfg_start_steps:
        return True
    else:
        return False

from logit_processor_3dim_static import multinomial_token_sample as get_prob

def sampling_logits2tokens(
    logits,
    all_collected_input_ids,
    unfinished_sequences, pad_token_id,
    output_token_num = 1,
    logits_processor=None, logits_warper=None,
    do_sample=True,
    has_eos_stopping_criteria=True,
    do_cfg=False,
    guidance_scale=3.,
    generator=None, #token_sampler = None,
    is_force_no_cfg = False,
    is_doing_speculative=False,
):
    # logits = logits / 0.8
    # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
    # (the clone itself is always small)
    next_token_logits = logits[ :, -output_token_num:, : ].clone()
    if do_cfg:
        conditional_logits, unconditional_logits = next_token_logits.chunk(2, dim=0)
        if is_force_no_cfg:
            next_token_logits = conditional_logits
        else:
            next_token_logits = guidance_scale * (conditional_logits - unconditional_logits) + unconditional_logits
    next_token_scores = next_token_logits
    next_token_scores = logits_processor(all_collected_input_ids, next_token_logits)
    # if do_sample and (logits_warper is not None):
    if (logits_warper is not None):
        next_token_scores = logits_warper(all_collected_input_ids, next_token_scores)

    if isinstance(do_sample, torch.Tensor):
        sampled_next_tokens, next_token_scores = get_prob(next_token_scores, generator=generator)
        greedy_next_tokens = torch.argmax(next_token_scores, dim=-1)
        next_tokens = torch.where(do_sample, sampled_next_tokens, greedy_next_tokens)
    elif do_sample:
        next_tokens, next_token_scores = get_prob(next_token_scores, generator=generator)
    else:
        next_tokens = torch.argmax(next_token_scores, dim=-1)
        next_token_scores = nn.functional.softmax(next_token_scores, dim=-1)
    
    # finished sentences should have their next token be a padding token
    if has_eos_stopping_criteria:
        while len(unfinished_sequences.shape) < len(next_tokens.shape):
            unfinished_sequences = unfinished_sequences.unsqueeze(-1)
        
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)    
    return next_tokens, next_token_scores


def find_first_misaligned_token_inds(
    input_ids, next_tokens,
):
    # input_ids: [B, L], next_tokens: [B, L]
    first_misaligned_token_inds = []
    for b in range(input_ids.shape[0]):
        first_misaligned_token_index = input_ids.shape[1] #- 1 # keep at least one token left
        for i in range(1, input_ids.shape[1]):
            if input_ids[b, i] == next_tokens[b, i-1]:
                pass
            else:
                first_misaligned_token_index = i
                break
    
        first_misaligned_token_inds.append(first_misaligned_token_index)
    
    return first_misaligned_token_inds

def prefix_matching_next_tokens(
    model_input_ids, next_tokens, next_token_scores,
    is_prefilling_phase=False,
    input_token_scores = None,
    prefix_token_sampler=None,
    **kwargs,
):
    # all_collected_input_ids should only include the first element of model_inputs['input_ids']

    if is_prefilling_phase:
        matched_num = model_input_ids.shape[1]
        matched_next_tokens = next_tokens[:, -1:]
        unmatched_next_tokens = next_tokens[:, next_tokens.shape[1]:]

        matched_next_scores = next_token_scores[:, -1:]
        unmatched_next_scores = next_token_scores[:, next_token_scores.shape[1]:]
    else:

        if prefix_token_sampler is not None:
            # SpeculativeSampler.__call__
            # print(f"input_token_scores {input_token_scores.shape}")
            first_misaligned_input_token_inds, next_tokens, next_token_scores = prefix_token_sampler(
                draft_tokens = model_input_ids,
                advanced_tokens = next_tokens,
                draft_prob = input_token_scores,
                advanced_prob = next_token_scores,
                **kwargs,
            )
            min_first_misaligned_input_token_index = min(first_misaligned_input_token_inds)
        else:
            first_misaligned_input_token_inds = find_first_misaligned_token_inds(
                model_input_ids, next_tokens,
            )
            min_first_misaligned_input_token_index = min(first_misaligned_input_token_inds)

        matched_num = min_first_misaligned_input_token_index
        matched_next_tokens = next_tokens[:, :matched_num]
        unmatched_next_tokens = next_tokens[:, matched_num:]

        matched_next_scores = next_token_scores[:, :matched_num]
        unmatched_next_scores = next_token_scores[:, matched_num:]
    
    # NOTE: is_cfg is True
    past_key_values = kwargs.get("past_key_values", None)
    if past_key_values is not None:
        if hasattr(prefix_token_sampler, "largest_prefix_batch_index"):
            prefix_token_sampler.kvcache_repeat(
                past_key_values, 
                selected_batch_index=prefix_token_sampler.largest_prefix_batch_index, 
                is_cfg=True,
            )

    return matched_num, matched_next_tokens, unmatched_next_tokens, matched_next_scores, unmatched_next_scores

def push_forward_model_kwargs_and_inputs(
    model_kwargs,
    all_collected_input_ids, 
    model_input_ids, output_token_num,
    num_matched_tokens, matched_next_tokens, unmatched_next_tokens,
    temporary_collected_scores=None,
    matched_next_scores=None,
    unmatched_next_scores=None,
):

    updated_input_ids = torch.cat([all_collected_input_ids, matched_next_tokens], dim=-1)
    if temporary_collected_scores is not None:
        # all_collected_scores = torch.cat([all_collected_scores, matched_next_scores], dim=-2)
        temporary_collected_scores = torch.cat([
            temporary_collected_scores[:, -1:], 
            matched_next_scores,
        ], dim=-2)
    
    past_key_values = model_kwargs["past_key_values"]
    attention_mask = model_kwargs["attention_mask"]
    cache_position = model_kwargs["cache_position"]

    new_model_inputs = None

    seq_len = cache_position.shape[-1]
    remaining_tokens_num = seq_len - num_matched_tokens
    if remaining_tokens_num > 0:
        # roll back

        # delete_false_key_value(past_key_values, remaining_tokens_num)
        if hasattr(past_key_values, "delete_false_key_value"):
            past_key_values.delete_false_key_value(remaining_tokens_num)
        else:
            delete_false_key_value(past_key_values, remaining_tokens_num)
        
        # no position_ids in model_kwargs

        attention_mask = attention_mask[..., :-remaining_tokens_num, :-remaining_tokens_num]
        cache_position = cache_position[..., :-remaining_tokens_num]

        new_model_inputs = {
            "past_key_values": past_key_values,
            # "position_ids": position_ids,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
        }

        model_kwargs.update(new_model_inputs)

        # updated_model_input_ids = model_input_ids.clone()
        # assert remaining_tokens_num == unmatched_next_tokens.shape[1]
        # updated_model_input_ids[:, -remaining_tokens_num:] = unmatched_next_tokens
        # model_inputs["input_ids"] = updated_model_input_ids

        additional_tokens = unmatched_next_tokens
        additional_scores = unmatched_next_scores

        nonoverlap_output_token_num = output_token_num #max(1, output_token_num - num_matched_tokens)
    else:

        # attention_mask is all useful
        
        # cache_position is updated by +1 in _update model kwargs for generation

        nonoverlap_output_token_num = output_token_num
        additional_tokens = None
        additional_scores = None
    
    return model_kwargs, updated_input_ids, nonoverlap_output_token_num, additional_tokens, additional_scores, temporary_collected_scores

def renew_pipeline(model_class):
    class JacobiPipeline(model_class):

        def _init_new_params(self, guidance_scale=3.0, image_top_k=2000, text_top_k=10, **kwargs):
            self.cfg = guidance_scale
            self.image_top_k = image_top_k
            self.text_top_k = text_top_k

        def create_logits_processor(self, cfg=3.0, image_top_k=2000, text_top_k=10):
            cfg = self.cfg if hasattr(self, 'cfg') else cfg
            image_top_k = self.image_top_k if hasattr(self, 'image_top_k') else image_top_k
            text_top_k = self.text_top_k if hasattr(self, 'text_top_k') else text_top_k

            logits_processor = LogitsProcessorList()

            candidate_processor = MultiTokensVLLogitsProcessor(
                image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
                image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
                image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
                patch_size=32,
                voc_size=self.model.config.vocab_size,
                device='cuda',
            )

            topk_processor = MultiTokensInterleavedTopKLogitsWarper(
                image_top_k=image_top_k,
                text_top_k=text_top_k,
                image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
                image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            )

            logits_processor.append(candidate_processor)
            logits_processor.append(topk_processor)
            logits_processor.append(
                TopnsigmaLogitsWarper3d(
                    n = 1,
                )
            )

            return logits_processor
    
    return JacobiPipeline

def get_multi_token_for_preparation(
    img_vocab, 
    rand_token_num, 
    input_ids, temporary_collected_scores, device, 
    multi_token_init_scheme=None,
    last_input_tokens=None, last_input_scores=None,
    generator = None,
    extrapolation_guidance_weight = 1.5,
    eps = 1e-7,
    prefill_num = 3,
    additional_tokens_len = 0,
    **kwargs,
):

    def random_multinomial_sample_from_logits(rand_logits):
        # probs = F.softmax(rand_logits, dim=-1)
        logits = rand_logits
        probs_shape = None
        if len(logits.shape) >= 3:
            probs_shape = logits.shape
            logits = logits.flatten(0, len(probs_shape)-2)

        topk_logits, topk_cls_indices = torch.topk(logits, k=1, dim=-1)
        logits = torch.full_like(logits, -float("Inf")).scatter(-1, topk_cls_indices, topk_logits)
        probs = F.softmax(logits, dim=-1)

        rand_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        if probs_shape is not None:
            rand_tokens = rand_tokens.reshape(probs_shape[:-1])
            probs = probs.reshape(probs_shape)
        
        return rand_tokens, probs

    if multi_token_init_scheme == 'random':
        img_vocab = img_vocab.to(device)
        img_vocab_size = len(img_vocab) #self.config.vocab_size
        rand_tokens = torch.randint(
            0, img_vocab_size, 
            (*input_ids.shape[:-1], rand_token_num)
        ).to(device)
        rand_tokens = img_vocab[rand_tokens] #self.chameleon_ori_translation._vocab.image_tokens

        scores_of_rand_tokens = temporary_collected_scores.new_zeros(
            (*temporary_collected_scores.shape[:-2], rand_token_num, temporary_collected_scores.shape[-1])
        )
        scores_of_rand_tokens = torch.scatter(scores_of_rand_tokens, -1, rand_tokens.unsqueeze(-1), 1.0)
    
    else:
        img_vocab = img_vocab.to(device)
        img_vocab_size = len(img_vocab) #self.config.vocab_size
        rand_tokens = torch.randint(
            0, img_vocab_size, 
            (*input_ids.shape[:-1], rand_token_num)
        ).to(device)
        rand_tokens = img_vocab[rand_tokens]

        scores_of_rand_tokens = temporary_collected_scores.new_zeros(
            (*temporary_collected_scores.shape[:-2], rand_token_num, temporary_collected_scores.shape[-1])
        )
        scores_of_rand_tokens = torch.scatter(scores_of_rand_tokens, -1, rand_tokens.unsqueeze(-1), 1.0)


        img_width = kwargs.get("img_width", None)
        pad_len = 1
        img_width = img_width + pad_len if img_width is not None else int(512 // 16)# TRICK adatping Anole #0
        input_ids_len = input_ids.shape[1]
        
        prefill_num = prefill_num + 3 #TRICK adatping Anole: 0 #TRICK adatping mgpt: 3 # mgpt <start, h, w>
        if (img_width > 0) and (input_ids_len + additional_tokens_len >= prefill_num) and (rand_token_num > 0):
            
            horizon_indices = (torch.arange(
                input_ids_len + additional_tokens_len, 
                input_ids_len + additional_tokens_len + rand_token_num, 
                device=device, dtype=torch.long
            ) - prefill_num) % img_width

            vertical_indices = (torch.arange(
                input_ids_len + additional_tokens_len, 
                input_ids_len + additional_tokens_len + rand_token_num, 
                device=device, dtype=torch.long
            ) - prefill_num) // img_width

            if 'horizon' in multi_token_init_scheme:
                valid_indices = (horizon_indices - 1 >= 0)
                last_vertical_indices = vertical_indices
                last_horizon_indices = horizon_indices - 1
            elif 'vertical' in multi_token_init_scheme:
                valid_indices = (vertical_indices - 1 >= 0)
                last_vertical_indices = vertical_indices - 1
                last_horizon_indices = horizon_indices
            else:
                assert False, f"multi_token_init_scheme should be 'horizon' or 'vertical', but got {multi_token_init_scheme}"
            
            last_input_tokens = torch.cat(
                [input_ids, last_input_tokens], dim=1
            ) if last_input_tokens is not None else input_ids
            last_input_scores = torch.cat(
                [temporary_collected_scores, last_input_scores], dim=1
            ) if last_input_scores is not None else temporary_collected_scores
            last_input_logits = (last_input_scores.float() + eps).log()

            last_flatten_indices = last_vertical_indices[valid_indices] * img_width + last_horizon_indices[valid_indices] + prefill_num
            
            # e.g., last indices [100, 101, 102], but the current indices up to 100, 
            # and 101, 102 depends on the values from 100 (but 100 has not been appended to input-ids yet)
            last_flatten_indices = last_flatten_indices.clamp(min=0, max = last_input_tokens.shape[1]-1) 
            
            last_resampled_input_tokens = last_input_tokens[:, last_flatten_indices]

            if 'sample' in multi_token_init_scheme:
                
                last_flatten_indices = (
                    last_flatten_indices - (
                        last_input_tokens.shape[1] - input_ids.shape[1]
                    )
                ).clamp(min=0, max = last_input_logits.shape[1]-1)

                last_resampled_input_logits = last_input_logits[:, last_flatten_indices]
                resampled_rand_tokens, resampled_scores_of_rand_tokens = random_multinomial_sample_from_logits(
                    last_resampled_input_logits
                )
                # scores_of_rand_tokens[:, valid_indices] = resampled_scores_of_rand_tokens
                scores_of_rand_tokens[:, valid_indices] = 0
                scores_of_rand_tokens[:, valid_indices] = torch.scatter(
                    scores_of_rand_tokens[:, valid_indices], -1, resampled_rand_tokens.unsqueeze(-1), 1.0)
            elif 'repeat' in multi_token_init_scheme:
                resampled_rand_tokens = last_resampled_input_tokens
                scores_of_rand_tokens[:, valid_indices] = 0
                scores_of_rand_tokens[:, valid_indices] = torch.scatter(
                    scores_of_rand_tokens[:, valid_indices], -1, resampled_rand_tokens.unsqueeze(-1), 1.0)
            else:
                assert False, f"multi_token_init_scheme should be 'sample' or 'repeat', but got {multi_token_init_scheme}"
            
            rand_tokens[:, valid_indices] = resampled_rand_tokens
    
    return rand_tokens, scores_of_rand_tokens

def renew_sampler(model_class):
    
    class JacobiSampler(model_class, nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_new_params()
        
        def prepare_inputs_for_generation_jacobi(
            self,
            input_ids,
            pixel_values=None,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            prefill_num = 3,
            **kwargs,
        ):
            # filling random tokens for multi-next-token prediction
            all_collected_length = input_ids.shape[1]
            is_append_random_tokens = kwargs.get("is_append_random_tokens", False) ###!!!
            num_fill_rand_tokens = kwargs.get("num_fill_rand_tokens", 0)
            additional_tokens = kwargs.get("additional_tokens", None)
            additional_scores = kwargs.get("additional_scores", None)
            temporary_collected_scores = kwargs.get("temporary_collected_scores", None)

            generator = kwargs.get("generator", None)

            # input_ids_queries = kwargs.get("input_ids_queries", None)
            input_ids_accept_len_ptr = kwargs.get("input_ids_accept_len_ptr", 0)

            if is_append_random_tokens:
                
                if additional_tokens is not None:
                    kept_unconverged_token_num = num_fill_rand_tokens - additional_tokens.shape[-1]
                    rand_token_num = kept_unconverged_token_num if (kept_unconverged_token_num >= 0) else 0
                    additional_tokens_len = additional_tokens.shape[1]
                else:
                    kept_unconverged_token_num = 0
                    rand_token_num = num_fill_rand_tokens
                    additional_tokens_len = 0
                
                last_input_tokens = additional_tokens
                last_input_scores = additional_scores

                rand_tokens, scores_of_rand_tokens = get_multi_token_for_preparation(
                    img_vocab=self.img_vocab, rand_token_num=rand_token_num, 
                    input_ids=input_ids, temporary_collected_scores = temporary_collected_scores,
                    device=input_ids.device,
                    multi_token_init_scheme=self.multi_token_init_scheme,
                    last_input_tokens = last_input_tokens, last_input_scores = last_input_scores,
                    additional_tokens_len = additional_tokens_len,
                    img_width = kwargs.get("img_width", None),
                    generator = generator,
                    prefill_num = prefill_num,
                )

                if additional_tokens is not None:
                    input_ids = torch.cat([ 
                        input_ids, 
                        additional_tokens[:, : additional_tokens.shape[1] + kept_unconverged_token_num], 
                        rand_tokens,
                    ], dim=-1)
                else:
                    input_ids = torch.cat([ 
                        input_ids, 
                        rand_tokens,
                    ], dim=-1)

                input_token_scores = None
            
            # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
            # Exception 1: when passing input_embeds, input_ids may be missing entries
            # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
            if past_key_values is not None:
                # print(f" cache_position {cache_position} {cache_position.shape}")
                if isinstance(cache_position, torch.Tensor) and len(cache_position.shape) >= 2: ###!!!
                    if inputs_embeds is not None:  # Exception 1
                        input_ids = input_ids[cache_position]
                    elif input_ids.shape[1] != cache_position.shape[-1]:
                        input_ids = input_ids[cache_position]


                else:
                    if inputs_embeds is not None:  # Exception 1
                        input_ids = input_ids[:, -cache_position.shape[0] :]
                    elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                        input_ids = input_ids[:, cache_position]
                    

                recycled_scores = additional_scores[
                    :, : additional_scores.shape[1] + kept_unconverged_token_num
                ] if additional_scores is not None else temporary_collected_scores[:, -1:-1]
                input_token_scores = gather_from_split_tensors(
                    tensor_list = [
                        temporary_collected_scores[:, -1:], 
                        recycled_scores, 
                        scores_of_rand_tokens,
                    ],
                    indexes = cache_position,
                    dim=1,
                    prefilled_length = all_collected_length - 1, # temporary_collected_scores.shape[1]-1
                    device=input_ids.device,
                )
                # print(f"old input_token_scores {input_token_scores.shape} {temporary_collected_scores.shape} {recycled_scores.shape} {scores_of_rand_tokens.shape}")

            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

                if len(position_ids.shape) == 3: ###!!!
                    position_ids = position_ids[:, -1, :]
                
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]
            
            # print('position_ids', position_ids)
            # print('attention_mask', attention_mask)
            # assert False

            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases


            if cache_position is not None and cache_position[0] == 0:
                # If we're in cached decoding stage, pixel values should be `None` because input ids do not contain special image token anymore
                # Otherwise we need pixel values to be passed to model
                model_inputs["pixel_values"] = pixel_values
            
            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                }
            )

            model_inputs['input_token_scores'] = input_token_scores
            # model_inputs['input_ids_queries'] = input_ids_queries
            model_inputs['input_ids_accept_len_ptr'] = input_ids_accept_len_ptr
            model_inputs['rand_tokens_shape'] = rand_tokens.shape[-1]

            return model_inputs

        def prepare_cfg_input(
            self, 
            model_inputs, 
            cfg_repeat_name_list, 
            prefill_num=None,
            neg_input_ids = None,
        ):

            def cfg_repeat(x):
                return x.repeat(2, *([1] * (len(x.shape) - 1)))

            for name in cfg_repeat_name_list:
                if (name in model_inputs) and (model_inputs[name] is not None):

                    if name == 'attention_mask':
                        model_inputs[name] = cfg_repeat(model_inputs[name])
                        B = model_inputs[name].shape[0]
                        model_inputs[name][B//2:, :prefill_num] = 0
                    elif name == 'input_ids' and neg_input_ids is not None:
                        input_ids = model_inputs[name]
                        neg_input_ids = neg_input_ids
                        model_inputs[name] = get_double_cfg_input_ids(
                            input_ids, 
                            neg_input_ids,
                            pad_category = self.config.pad_token_id,
                        )
                    else:
                        model_inputs[name] = cfg_repeat(model_inputs[name])
            
            return model_inputs

        def _get_initial_cache_position(self, input_ids, model_kwargs):
            """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
            # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
            if "inputs_embeds" in model_kwargs:
                cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            else:
                cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

            past_length = 0
            if model_kwargs.get("past_key_values") is not None:
                cache = model_kwargs["past_key_values"]
                past_length = 0
                if not isinstance(cache, Cache):
                    past_length = cache[0][0].shape[2]
                elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                    past_length = cache.get_seq_length()

                # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
                # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
                if not is_torchdynamo_compiling():
                    cache_position = cache_position[past_length:]

            model_kwargs["cache_position"] = cache_position

            return model_kwargs
        
        def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            num_new_tokens: int = 1,
        ) -> Dict[str, Any]:
            # update past_key_values keeping its naming used in model code
            cache_name, cache = self._extract_past_from_model_output(outputs)
            model_kwargs[cache_name] = cache
            if getattr(outputs, "state", None) is not None:
                model_kwargs["state"] = outputs.state

            # update token_type_ids with last value
            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs["token_type_ids"]
                model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

            if not is_encoder_decoder:
                # update attention mask
                if "attention_mask" in model_kwargs:
                    
                    attention_mask = model_kwargs["attention_mask"]

                    if hasattr(self, "static_masker"):
                        model_kwargs["attention_mask"] = self.static_masker.get_causal_decoding_attention_mask(
                            num_new_tokens=num_new_tokens, pre_mask=attention_mask,
                        )
                    else:
                        while len(attention_mask.shape) < 3:
                            attention_mask = attention_mask.unsqueeze(1)
                        
                        attention_mask = attention_mask[..., -1:, :] #
                        
                        if attention_mask.shape[-2] < num_new_tokens:
                            attention_mask = attention_mask.expand(
                                (attention_mask.shape[0], num_new_tokens, attention_mask.shape[-1])
                            )
                        
                        model_kwargs["attention_mask"] = torch.ones(
                            ( attention_mask.shape[0], num_new_tokens, num_new_tokens + attention_mask.shape[-1] ),
                            device=attention_mask.device, dtype=attention_mask.dtype,
                        )
                        model_kwargs["attention_mask"][..., :, :attention_mask.shape[-1]] = attention_mask[..., -1:, :]
                        model_kwargs["attention_mask"][..., :, attention_mask.shape[-1]:] = torch.tril(
                            model_kwargs["attention_mask"][0, :, attention_mask.shape[-1]:]
                        )
            else:
                # update decoder attention mask
                if "decoder_attention_mask" in model_kwargs:
                    decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                    model_kwargs["decoder_attention_mask"] = torch.cat(
                        [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                        dim=-1,
                    )

            if model_kwargs.get("use_cache", True):
                # if num_new_tokens <= 1:
                #     model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
                # else: ###!!!
                past_positions = model_kwargs.pop("cache_position")
                new_positions = torch.arange(
                    past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
                ).to(past_positions.device)
                model_kwargs["cache_position"] = new_positions
            else:
                past_positions = model_kwargs.pop("cache_position")
                new_positions = torch.arange(
                    past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
                ).to(past_positions.device)
                model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
            
            return model_kwargs

        def _init_new_params(
            self, 
            jacobi_loop_interval_l = 1,  #768 // 16
            jacobi_loop_interval_r = (768 // 8)**2 + 768 // 8, # This should be determined by the image size ###!!!
            max_num_new_tokens = 8, #16,
            guidance_scale = 3.0,
            seed = 42,
            multi_token_init_scheme = 'random',
            do_cfg = True,
            prefix_token_sampler_scheme = 'speculative_jacobi',
            use_chameleon_tokenizer = True,
            _init_doubled_attn_mask_cfg = False,
            speculative_window_size = None, 
            is_compile=False,
            **kwargs,
        ):
            if speculative_window_size is None:
                speculative_window_size = max_num_new_tokens
            
            self.speculative_window_size = speculative_window_size

            self.static_masker = AuxiliarySuffixStaticCausalAttentionBlockMask(
                max_query_length = max_num_new_tokens,
                max_decoding_length = jacobi_loop_interval_r * 2,
            )

            self.sjd_class = SpeculativeSampler
            
            if use_chameleon_tokenizer:
                # import model.chameleon_vae_ori as chameleon_vae_ori
                # chameleon_ori_vocab = chameleon_vae_ori.VocabInfo(
                #     json.load(open("./ckpts/chameleon/tokenizer/text_tokenizer.json"))["model"]["vocab"]
                # )
                # chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(chameleon_ori_vocab)
                # img_vocab = chameleon_ori_translation._vocab.image_tokens
                img_vocab=self.model.vocabulary_mapping.get_image_tokens
                self.register_buffer("img_vocab", torch.tensor(img_vocab, dtype=torch.long))

            self.cfg_repeat_name_list = [
                'inputs_embeds', 'input_ids', 'pixel_values', 'input_token_scores', 
            ]
            self.cfg_half_name_list = [
                'inputs_embeds', 'input_ids', 'pixel_values', 'input_token_scores', 
            ]
            self.jacobi_loop_interval_l = jacobi_loop_interval_l
            self.jacobi_loop_interval_r = jacobi_loop_interval_r
            self.max_num_new_tokens = max_num_new_tokens
            self.max_jacobi_iter_num = min(200, self.max_num_new_tokens+1) ###!!!
            self.guidance_scale = guidance_scale

            self.seed = seed
            self.generator = None

            self.multi_token_init_scheme = multi_token_init_scheme
            self.do_cfg = do_cfg

            self.prefix_token_sampler_scheme = prefix_token_sampler_scheme
            self._init_doubled_attn_mask_cfg = _init_doubled_attn_mask_cfg
            
            self.output_file_name = None

            self.is_compile = is_compile
            if self.is_compile:
                # torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
                # self.compiled_self = torch.compile(
                #     self,
                #     mode="reduce-overhead"
                # ) 

                pass
                # self.compiled_self = torch.compile(
                #     self,
                #     mode="max-autotune", #fullgraph=True,
                #     backend="inductor",
                # )
        
        def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            streamer,
            logits_warper: Optional[LogitsProcessorList] = None,
            **model_kwargs,
        ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
            r"""
            Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
            can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

            Parameters:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    The sequence used as a prompt for the generation.
                logits_processor (`LogitsProcessorList`):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                    used to modify the prediction scores of the language modeling head applied at each generation step.
                stopping_criteria (`StoppingCriteriaList`):
                    An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                    used to tell if the generation loop should stop.
                generation_config ([`~generation.GenerationConfig`]):
                    The generation configuration to be used as parametrization of the decoding method.
                synced_gpus (`bool`):
                    Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
                streamer (`BaseStreamer`, *optional*):
                    Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                    through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
                logits_warper (`LogitsProcessorList`, *optional*):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                    to warp the prediction score distribution of the language modeling head applied before multinomial
                    sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                    `generation_config`)
                model_kwargs:
                    Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                    an encoder-decoder model the kwargs should include `encoder_outputs`.

            Return:
                [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
                A `torch.LongTensor` containing the generated tokens (default behaviour) or a
                [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
                `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
                `model.config.is_encoder_decoder=True`.
            """
            # init values
            pad_token_id = generation_config._pad_token_tensor
            # assert False, f"pad_token_id: {pad_token_id}"
            output_attentions = generation_config.output_attentions
            output_hidden_states = generation_config.output_hidden_states
            output_scores = generation_config.output_scores
            output_logits = generation_config.output_logits
            return_dict_in_generate = generation_config.return_dict_in_generate
            max_length = generation_config.max_length
            has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
            do_sample = generation_config.do_sample
            # print(f"do_sample {do_sample}")
            # if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            #     raise ValueError(
            #         "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
            #         f"{logits_warper})."
            #     )

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            raw_logits = () if (return_dict_in_generate and output_logits) else None
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

            # keep track of which sequences are already finished
            batch_size, cur_len = input_ids.shape
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            
            temporary_collected_scores = input_ids.new_zeros((batch_size, cur_len, self.config.vocab_size))
            temporary_collected_scores = torch.scatter(temporary_collected_scores, 2, input_ids.unsqueeze(-1), 1.0)

            # init: attn mask, cache_position, cfg, 
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
            prefill_num = model_kwargs['attention_mask'].shape[1] - 1

            do_cfg = self.do_cfg if hasattr(self, 'do_cfg') else False

            guidance_scale = self.guidance_scale if hasattr(self, 'guidance_scale') else 3.0
            do_cfg = (do_cfg & (guidance_scale != 1))

            device = input_ids.device
            dtype = input_ids.dtype
            if self.seed is not None:
                set_seed(self.seed)
                self.generator = torch.Generator(input_ids.device).manual_seed(self.seed)

            if self.prefix_token_sampler_scheme == 'speculative_jacobi':
                prefix_token_sampler = self.sjd_class(
                    generator=self.generator,
                    speculative_window_size = self.speculative_window_size,
                    logits_processor = logits_processor, logits_warper = logits_warper,
                )
            elif self.prefix_token_sampler_scheme == 'jacobi':
                prefix_token_sampler = None
            else:
                raise ValueError(f"prefix_token_sampler_scheme: {self.prefix_token_sampler_scheme}")

            
            if do_cfg:
                model_kwargs = self.prepare_cfg_input(
                    model_kwargs, 
                    cfg_repeat_name_list = ['attention_mask', ] if (
                        not self._init_doubled_attn_mask_cfg
                    ) else [],
                    prefill_num = prefill_num ,
                )

            # prefilling tokens always output 1 next token
            output_token_num = 1
            additional_tokens = None
            additional_scores = None

            jacobi_loop_interval_lr = (cur_len + self.jacobi_loop_interval_l, cur_len + self.jacobi_loop_interval_r)
            gen_loop_num = 0
            max_num_new_tokens = self.max_num_new_tokens ###!!!

            count_time = True
            if count_time:
                # t1 = torch.cuda.Event(enable_timing=True)
                # t2 = torch.cuda.Event(enable_timing=True)
                # torch.cuda.synchronize()
                # t1.record()
                t1 = time.time()
            
            num_matched_tokens = max_num_new_tokens
            
            if self.is_compile: #and (model_kwargs.get("past_key_values", None) is not None):
                past_key_values = model_kwargs["past_key_values"]
                # print(f"past_key_values {past_key_values}")
                past_key_values = PointerStaticCache(
                    config = self.config,
                    batch_size = batch_size * 2 if do_cfg else batch_size,
                    max_cache_len = jacobi_loop_interval_lr[-1] + 100,
                    device=device,
                    dtype= torch.bfloat16,
                )
                # from pdb import set_trace; set_trace()
                model_kwargs["past_key_values"] = past_key_values
                # print(f"past_key_values 2 {past_key_values}")

            do_sample_list = do_sample #torch.full((batch_size, max_num_new_tokens), do_sample, dtype=torch.bool, device=device)
            while self._has_unfinished_sequences(
                this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
            ):

                # prepare model inputs
                # place the last `len(cache_position)` elements of `input_ids` into `model_kwargs` 
                model_inputs = self.prepare_inputs_for_generation_jacobi(
                    input_ids, 
                    num_fill_rand_tokens = output_token_num - 1,
                    is_append_random_tokens = True,
                    additional_tokens = additional_tokens,
                    additional_scores = additional_scores,
                    temporary_collected_scores = temporary_collected_scores, 
                    # input_ids_queries = input_ids_queries, input_ids_accept_len_ptr = input_ids_accept_len_ptr,
                    img_width = logits_processor[0].w_latent_dim if hasattr(logits_processor[0], 'w_latent_dim') else None,
                    generator = self.generator,
                    prefill_num = prefill_num,
                    **model_kwargs,
                )
                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                # the first element of model_inputs['input_ids'] is in all_collected_input_ids
                all_collected_input_ids = input_ids
                model_input_ids = model_inputs['input_ids']

                input_ids_accept_len_ptr = model_inputs.pop('input_ids_accept_len_ptr')
                rand_tokens_shape = model_inputs.pop('rand_tokens_shape')

                is_force_no_cfg = check_is_force_no_cfg(
                    input_ids, 
                    image_start_token_id = logits_processor[0].image_start_token_id if hasattr(
                        logits_processor[0], 'image_start_token_id'
                    ) else None,
                    image_end_token_id = logits_processor[0].image_end_token_id if hasattr(
                        logits_processor[0], 'image_end_token_id'
                    ) else None,
                    guidance_scale=guidance_scale,
                    do_cfg = do_cfg,
                    non_cfg_start_steps=3,
                    current_step=gen_loop_num,
                ) # to adapt the bs=2 kv cache

                #NOTE new here
                input_token_scores = model_inputs.pop('input_token_scores')
                # print(f"1 input_token_scores {input_token_scores.shape}")

                if do_cfg:
                    model_inputs = self.prepare_cfg_input(
                        model_inputs,
                        cfg_repeat_name_list = self.cfg_repeat_name_list,
                        neg_input_ids = model_kwargs.get(
                            'neg_input_ids', None
                        ) if (gen_loop_num == 0) else None,
                    )
                # if gen_loop_num % 1000 == 0:
                #     print(gen_loop_num, cur_len, 'output_token_num', output_token_num)


                # forward pass to get next token
                do_sample_list = do_sample #torch.full((max_num_new_tokens, ), do_sample, dtype=torch.bool, device=device)

                # if self.is_compile and output_token_num == max_num_new_tokens and (cur_len >= jacobi_loop_interval_lr[0] + 1) and (cur_len < jacobi_loop_interval_lr[-1] ):
                #     outputs = self.compiled_self(**model_inputs, return_dict=True)
                # else:
                #     outputs = self(**model_inputs, return_dict=True)
                outputs = self(**model_inputs, return_dict=True)

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                logits = outputs.logits


                next_tokens, next_token_scores = sampling_logits2tokens(
                    logits,
                    all_collected_input_ids,
                    unfinished_sequences, pad_token_id,
                    output_token_num = output_token_num,
                    logits_processor=logits_processor, logits_warper=logits_warper,
                    do_sample=do_sample_list,
                    has_eos_stopping_criteria=has_eos_stopping_criteria,
                    do_cfg=do_cfg,
                    generator=self.generator, # token_sampler = prefix_token_sampler,
                    guidance_scale=guidance_scale,
                    is_force_no_cfg=is_force_no_cfg,
                    is_doing_speculative = (output_token_num > 1),
                )

                if do_cfg:
                    model_inputs = postprocess_cfg_decode(
                        model_inputs, 
                        cfg_half_name_list=self.cfg_half_name_list, #NOTE: new
                    )

                num_matched_tokens, matched_next_tokens, unmatched_next_tokens, \
                matched_next_scores, unmatched_next_scores = prefix_matching_next_tokens(
                    model_input_ids=model_input_ids, 
                    input_token_scores = input_token_scores,
                    next_tokens=next_tokens, 
                    next_token_scores=next_token_scores,
                    is_prefilling_phase = (output_token_num <= 1),
                    prefix_token_sampler = prefix_token_sampler, #logits_processor = logits_processor, logits_warper = logits_warper,
                    all_collected_input_ids = all_collected_input_ids,
                    past_key_values = model_kwargs["past_key_values"],
                )


                if (cur_len >= jacobi_loop_interval_lr[0]) and (cur_len < jacobi_loop_interval_lr[-1]):

                    output_token_num = max(1, min(max_num_new_tokens, jacobi_loop_interval_lr[-1] - cur_len)) #BUG fixed
                    
                else:
                    output_token_num = 1

                model_kwargs, updated_input_ids, \
                nonoverlap_output_token_num, \
                additional_tokens, additional_scores, \
                temporary_collected_scores = push_forward_model_kwargs_and_inputs(
                    model_kwargs=model_kwargs, 
                    all_collected_input_ids=all_collected_input_ids, 
                    model_input_ids=model_input_ids, 
                    output_token_num=output_token_num,
                    num_matched_tokens=num_matched_tokens,
                    matched_next_tokens=matched_next_tokens,
                    unmatched_next_tokens=unmatched_next_tokens,
                    temporary_collected_scores=temporary_collected_scores,
                    matched_next_scores=matched_next_scores,
                    unmatched_next_scores=unmatched_next_scores,
                )
                input_ids = updated_input_ids

                # Store scores, attentions and hidden_states when required
                assert (not return_dict_in_generate) # TODO: too many codes to collect the prefixes in outputs

                if streamer is not None:
                    if len(next_tokens.shape) == 1:
                        streamer.put(next_tokens.cpu())
                    else:
                        for j in range(next_tokens.shape[1]):
                            streamer.put(next_tokens[:, j].cpu())
                
                # No input_ids here, only the cache_position
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    num_new_tokens = nonoverlap_output_token_num,
                )

                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0

                cur_len = input_ids.shape[1]
                gen_loop_num += 1

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs

            if streamer is not None:
                streamer.end()
            
            if count_time:
                # t2.record()
                # torch.cuda.synchronize()
                t2 = time.time()

                # t = t1.elapsed_time(t2) / 1000
                t = (t2 - t1)
                # print("Time elapsed inner: ", t)
                # print("gen loop num (NFE): ", gen_loop_num)
                # print("tokens length: ", cur_len)
                logging.info(f"Time elapsed inner: {t}")
                logging.info(f"gen loop num (NFE): {gen_loop_num}")
                logging.info(f"tokens length: {cur_len}")
            
            if prefix_token_sampler is not None:
                prefix_token_sampler.reset()
                
            if hasattr(self, "static_masker"):
                self.static_masker.reset()

            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    return GenerateEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
                else:
                    return GenerateDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
            else:
                return input_ids
    
    return JacobiSampler

def renew_backbone(model_class):
    class JacobiBackbone(model_class): # ChameleonForConditionalGeneration (NOT) -> ChameleonModel

        def _init_new_params(
            self,
            **kwargs,
        ):
            pass

        def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
        ):
            # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
            # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
            # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
            # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
            if not isinstance(attention_mask, torch.Tensor):
                return attention_mask

            if self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None

            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
            # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
            # to infer the attention mask.
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            using_static_cache = isinstance(past_key_values, StaticCache)

            if self.training:
                return attention_mask

            # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
            if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
                ):
                    return None

            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = torch.finfo(dtype).min
            sequence_length = input_tensor.shape[1]
            if using_static_cache:
                target_length = past_key_values.get_max_length()
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else past_seen_tokens + sequence_length + 1
                )

            if attention_mask is not None and attention_mask.dim() == 4:
                # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
                if attention_mask.max() != 0:
                    raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
                causal_mask = attention_mask
            else:
                causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
                if sequence_length != 1:
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
                causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
                if attention_mask is not None:
                    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                    mask_length = attention_mask.shape[-1]

                    while attention_mask.dim() < 4:
                        attention_mask = attention_mask.unsqueeze(1)

                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask # [:, None, None, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )
            
            if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
            ):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

            return causal_mask
        
        def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, BaseModelOutputWithPast]:
            
            if self.training:

                if pixel_values is not None:
                    image_tokens = self.get_image_tokens(pixel_values)
                    special_image_mask = input_ids == self.vocabulary_mapping.image_token_id
                    image_tokens = image_tokens.to(input_ids.device, input_ids.dtype)
                    input_ids = input_ids.masked_scatter(special_image_mask, image_tokens)

                    pixel_values = None


            outputs = super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            
            return outputs
    
    return JacobiBackbone

def renew_pipeline_sampler(pipe_line, **kwargs):
    pipe_line.__class__ = renew_pipeline(pipe_line.__class__)
    pipe_line._init_new_params(**kwargs)
    pipe_line.model.__class__ = renew_sampler(pipe_line.model.__class__)
    pipe_line.model._init_new_params(**kwargs)

    # from train_utils.flexattns import equip_flexattn # inference very slow
    # if equip_flexattn is not None:
    #     equip_flexattn(pipe_line.model.model) # ChameleonModel

    pipe_line.model.model.__class__ = renew_backbone(pipe_line.model.model.__class__)
    pipe_line.model.model._init_new_params(**kwargs)

    return pipe_line

