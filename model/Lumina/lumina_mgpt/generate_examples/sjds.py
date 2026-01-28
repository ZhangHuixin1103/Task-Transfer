"""
Reference:
    https://github.com/tyshiwo1/Accelerating-T2I-AR-with-SJD/blob/main/scheduler/jacobi_iteration_lumina_mgpt.py
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn
import copy

from logit_processor_3dim_static import multinomial_token_sample


class SpeculativeSampler:

    def __init__(
        self,
        generator=None,
        draft_type = 'jacobian_states',
        speculative_window_size = 128, #16,
        is_rej_sampling = True,
        logits_processor = None, logits_warper = None,
        **kwargs,
    ):
        self.logits_processor = logits_processor
        self.logits_warper = logits_warper

        self.is_rej_sampling = is_rej_sampling
        self.speculative_window_size = speculative_window_size

        self.draft_token_index_selector = lambda x: x
        if draft_type == 'jacobian_states':
            self.advanced_token_index_selector = lambda x: x - 1
        else:
            self.advanced_token_index_selector = lambda x: x

        self.generator = generator

        self.called_times = 0
        self._init_reject_sampling_params()
    
    def get_acc_criterion_mask(
        self,
        advanced_prob, draft_prob, draft_tokens, 
        rs=None, generator=None,
        rand_times=1,
        is_return_all=False,
    ):
        cls_indices = draft_tokens[:, 1:]
        sampled_advanced_probs = torch.gather(advanced_prob[:, :-1, :], 2, cls_indices.unsqueeze(-1)).squeeze(-1)
        sampled_draft_probs = torch.gather(draft_prob[:, 1:, :], 2, cls_indices.unsqueeze(-1)).squeeze(-1)
        if rs is None:
            rs = torch.rand(sampled_advanced_probs.shape, device=advanced_prob.device, generator=generator).float()
        elif len(rs.shape) >= 3:
            rs = torch.gather(rs[:, 1:, :], 2, cls_indices.unsqueeze(-1)).squeeze(-1)
        
        criterion_mask = (sampled_advanced_probs / sampled_draft_probs.clamp(min=1e-7)).clamp(max=1) # B, L-1

        if rand_times > 1:
            new_rs = torch.rand(
                (*rs.shape, rand_times), 
                device=advanced_prob.device, generator=generator
            ) # B, L-1, N
            new_rs[..., :1].copy_(rs.unsqueeze(-1))
            rs = new_rs
            acc_criterion_mask = (
                (rs < criterion_mask.unsqueeze(-1).repeat(1, 1, rand_times)) * 1.
            ).sum(dim=-1)
            acc_criterion_mask = (acc_criterion_mask >= rand_times / 2)

        else:
            acc_criterion_mask = (rs < criterion_mask)

        p_res = self.get_reject_sampling_logits(advanced_prob[:, :-1, :], draft_prob[:, 1:, :])
        if is_return_all:
            return acc_criterion_mask, criterion_mask, p_res, sampled_advanced_probs, sampled_draft_probs
        else:
            return acc_criterion_mask, criterion_mask, p_res
    
    def reset(self,):

        self.collected_draft_logits = []
        self.collected_advanced_logits = []
        self._init_reject_sampling_params()
    
    def get_reject_sampling_logits(self, token_advanced_prob, token_draft_prob, eps=1e-7):
        delta_prob = (
            token_advanced_prob.float() - token_draft_prob.float().clamp(max=0.99)
        ).clamp(min=0)

        pos_delta_prob = delta_prob / delta_prob.sum(-1, keepdim=True).clamp(min=1e-10)

        valid_mask = (delta_prob.sum(-1, keepdim=False) + eps > 1.)
        pos_delta_prob[~valid_mask] = token_advanced_prob[~valid_mask]

        pos_delta_logits = pos_delta_prob.log()

        return pos_delta_logits
    
    def reject_sampling_single_token(
        self, token_advanced_prob, token_draft_prob,
        all_collected_input_ids=None,
        pos_delta_logits = None,
    ):
        logits_processor = self.logits_processor
        logits_warper = self.logits_processor

        if pos_delta_logits is None:
            # pos_delta_logits = token_advanced_prob.clamp(min=1e-10).log().clamp(min=-1e5)
            pos_delta_logits = token_advanced_prob.log()

        shape_pos_delta_logits = pos_delta_logits.shape

        if (logits_processor is not None) or (logits_warper is not None):
            while len(all_collected_input_ids.shape) < 2:
                all_collected_input_ids = all_collected_input_ids.unsqueeze(0)
            
            while len(pos_delta_logits.shape) < 3:
                pos_delta_logits = pos_delta_logits.unsqueeze(0)
        
        if logits_processor is not None:
            pos_delta_logits = logits_processor(all_collected_input_ids, pos_delta_logits)
        
        elif logits_warper is not None: #NOTE: if logits_warper is not None:
            pos_delta_logits = logits_warper(all_collected_input_ids, pos_delta_logits)

        resampled_tokens, resampled_scores = multinomial_token_sample(
            pos_delta_logits, generator=self.generator, is_input_score=False,
        )

        if len(shape_pos_delta_logits) == 1:
            resampled_tokens = resampled_tokens.reshape(-1).squeeze(-1)
        else:
            resampled_tokens = resampled_tokens.view(shape_pos_delta_logits[:-1])
        
        resampled_scores = resampled_scores.view(shape_pos_delta_logits)

        return resampled_tokens, resampled_scores
        
    def _init_reject_sampling_params(self,):
        # self.reject_sampling_relative_ids.fill_(-1)
        # self.reject_sampling_draft_token_logits.fill_(0)
        self.called_times = 0

    def __call__(
        self, draft_tokens, advanced_tokens, draft_prob, advanced_prob,
        all_collected_input_ids = None,
        **kwargs,
    ): 
        B, L = draft_tokens.shape
        dtype = draft_prob.dtype

        # draft_prob = draft_prob.clone().float()
        # advanced_prob = advanced_prob.clone().float()
        proxy_draft_prob = draft_prob #.clone().float()
        proxy_advanced_prob = advanced_prob #.clone().float()

        rs = torch.rand(advanced_prob.shape, device=advanced_prob.device, generator=self.generator).float()

        draft_token_index_selector = self.draft_token_index_selector
        advanced_token_index_selector = self.advanced_token_index_selector

        resampled_target_tokens = advanced_tokens.clone()
        resampled_target_scores = advanced_prob.clone().float()

        acc_criterion_mask, criterion_mask, p_res, \
        sampled_hard_advanced_probs, sampled_draft_probs = self.get_acc_criterion_mask(
            advanced_prob, draft_prob, draft_tokens, 
            rs=rs, generator=self.generator,
            is_return_all=True,
        )

        first_misaligned_token_inds = []
        for b in range(B):
            first_misaligned_token_index = L # keep at least one token left

            for i in range(1, L):

                draft_token_index = draft_token_index_selector(i)
                target_token_index = advanced_token_index_selector(i)

                cls_idx = draft_tokens[b, draft_token_index]

                # acc_criterion = (r < (sampled_advanced_prob / sampled_draft_prob.clamp(min=1e-7)).clamp(max=1))
                acc_criterion = acc_criterion_mask[b, target_token_index]

                if acc_criterion: 
                    # accept sampling
                    resampled_target_tokens[b, target_token_index] = cls_idx
                    resampled_target_scores[b, target_token_index, :] = draft_prob[b, draft_token_index, :]
                else:
                    if self.is_rej_sampling:

                        # we perform reject sampling in the backbone model's prediction loop out of the this sampler
                        token_advanced_prob = proxy_advanced_prob[b, target_token_index, :]
                        token_draft_prob = proxy_draft_prob[b, draft_token_index, :]

                        pos_delta_logits = p_res[b, target_token_index, :]

                        resampled_tokens, resampled_scores = self.reject_sampling_single_token(
                            token_advanced_prob = token_advanced_prob, 
                            token_draft_prob = None, # NOTE: decrepte token_draft_prob,
                            all_collected_input_ids = torch.cat([
                                all_collected_input_ids[b, :],
                                resampled_target_tokens[b, :target_token_index], # advanced_tokens[b, :target_token_index],
                            ], dim=-1),
                            pos_delta_logits=pos_delta_logits,
                        )
                        resampled_target_tokens[b, target_token_index] = resampled_tokens
                        # resampled_target_scores[b, target_token_index, :] = resampled_scores # the score (probability) is kept, so not to update this

                    
                    first_misaligned_token_index = target_token_index + 1 # NOTE: `+1` means the calibrated token directly ac

                    break

            first_misaligned_token_inds.append(first_misaligned_token_index)


        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores.to(dtype)