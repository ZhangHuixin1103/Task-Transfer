import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Tuple, Any

from transformers.cache_utils import Cache, StaticCache

class StaticCausalAttentionMask:
    def __init__(self, max_decoding_length=90*91+10, **kwargs):
        self.max_decoding_length = max_decoding_length
        self.query_pointer = None
        self.key_pointer = None
        self.static_decoding_attention_mask = None
    
    def _init_causal_decoding_attention_mask(self, pre_mask, **kwargs):
        max_decoding_length = self.max_decoding_length
        attention_mask = pre_mask # prefilling_mask

        while len(attention_mask.shape) < 3:
            attention_mask = attention_mask.unsqueeze(1)
        
        attention_mask = attention_mask[..., -1:, :] # B, 1, L

        B, _, L_prefilling = attention_mask.shape
        
        # attention_mask = attention_mask.expand(
        #     (attention_mask.shape[0], num_new_tokens, attention_mask.shape[-1])
        # )
        static_decoding_attention_mask = torch.full(
            (B, max_decoding_length, max_decoding_length + L_prefilling),
            fill_value = 1,
            device=attention_mask.device, dtype=attention_mask.dtype,
        )

        static_decoding_attention_mask[..., :, :L_prefilling] = attention_mask #.repeat(1, max_decoding_length, 1)
        static_decoding_attention_mask[..., :, L_prefilling:] = self._make_tril_mask(
            static_decoding_attention_mask[0, :, L_prefilling:],
            max_val=1, #attention_mask.max().item(),
            min_val=0, #attention_mask.min().item(),
        )
        # torch.tril(
        #     static_decoding_attention_mask[0, :, L_prefilling:]
        # )

        self.static_decoding_attention_mask = static_decoding_attention_mask
        self.query_pointer = 0
        self.key_pointer = L_prefilling

        return static_decoding_attention_mask
    
    def _make_tril_mask(self, attention_mask, max_val, min_val):
        """
            attention_mask must 1;
            min_val != 1
        """
        attention_mask = torch.tril(
            attention_mask
        )
        if min_val == 1:
            attention_mask = torch.where(
                attention_mask == 0, min_val, max_val
            )
        else:
            attention_mask.masked_fill_(attention_mask == 0, min_val)
            attention_mask.masked_fill_(attention_mask == 1, max_val)
        
        return attention_mask

    
    def _extend_static_mask(self, extend_key_length, extend_query_length=0):
        static_decoding_attention_mask = self.static_decoding_attention_mask
        B, q_len, k_len = static_decoding_attention_mask.shape
        new_static_decoding_attention_mask = torch.full(
            (B, q_len + extend_query_length, k_len + extend_key_length),
            fill_value = 1,
            device=static_decoding_attention_mask.device, dtype=static_decoding_attention_mask.dtype,
        )
        new_static_decoding_attention_mask[..., :q_len, :k_len] = static_decoding_attention_mask
        new_static_decoding_attention_mask[..., :q_len, k_len:] = 0 #static_decoding_attention_mask.min().item()
        if extend_query_length != 0 and extend_key_length != 0:
            new_static_decoding_attention_mask[..., q_len:, k_len:] = self._make_tril_mask(
                new_static_decoding_attention_mask[0, q_len:, k_len:],
                max_val=1, #static_decoding_attention_mask.max().item(),
                min_val=0, #static_decoding_attention_mask.min().item(),
            )
            # torch.tril(
            #     new_static_decoding_attention_mask[0, q_len:, k_len:]
            # )
        
        new_static_decoding_attention_mask[..., q_len:, :k_len] = static_decoding_attention_mask[..., -1:, :]
        self.static_decoding_attention_mask = new_static_decoding_attention_mask
        return new_static_decoding_attention_mask
        

    def get_causal_decoding_attention_mask(self, num_new_tokens, is_gradual_remove=False, **kwargs):
        """
        
            attention_mask = model_kwargs["attention_mask"]

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
        """

        if self.static_decoding_attention_mask is None:
            self._init_causal_decoding_attention_mask(**kwargs)

        
        attention_mask = self.static_decoding_attention_mask
        q_pointer = self.query_pointer
        k_pointer = self.key_pointer

        if q_pointer+num_new_tokens > attention_mask.shape[-2]:
            attention_mask = self._extend_static_mask(num_new_tokens, num_new_tokens)

        if not is_gradual_remove:
            attention_mask = attention_mask[..., q_pointer:q_pointer+num_new_tokens, :k_pointer+num_new_tokens]
            self.query_pointer = q_pointer + num_new_tokens
            self.key_pointer = k_pointer + num_new_tokens

        else:
            attention_mask = attention_mask[..., q_pointer:q_pointer+num_new_tokens, :k_pointer+num_new_tokens]
            self.query_pointer = q_pointer
            self.key_pointer = k_pointer + num_new_tokens
            self.static_decoding_attention_mask = attention_mask[..., q_pointer+num_new_tokens:, :]

        return attention_mask
    
    def reset(self):
        self.query_pointer = None
        self.key_pointer = None
        self.static_decoding_attention_mask = None
    

class StaticCausalAttentionBlockMask(StaticCausalAttentionMask):
    def __init__(self, max_query_length=128 + 16, is_static_kvcache=False, **kwargs):
        super().__init__(**kwargs)
        self.max_query_length = max_query_length
        self.is_static_kvcache = is_static_kvcache

    def _init_causal_decoding_attention_mask(self, pre_mask):
        max_decoding_length = self.max_decoding_length
        attention_mask = pre_mask #prefilling_mask

        while len(attention_mask.shape) < 3:
            attention_mask = attention_mask.unsqueeze(1)
        
        attention_mask = attention_mask[..., -1:, :] # B, 1, L

        B, _, L_prefilling = attention_mask.shape
        
        max_query_length = self.max_query_length
        static_decoding_attention_mask = torch.full(
            (B, max_query_length, max_decoding_length + L_prefilling),
            fill_value = 1,
            device=attention_mask.device, dtype=attention_mask.dtype,
        )

        static_little_full_tril_mask = self._make_tril_mask(
            static_decoding_attention_mask[0, :, L_prefilling:L_prefilling+max_query_length],
            max_val=1, #attention_mask.max().item(),
            min_val=0, #attention_mask.min().item(),
        )

        static_decoding_attention_mask[..., :, :L_prefilling] = attention_mask #.repeat(1, max_decoding_length, 1)
        static_decoding_attention_mask[..., :, L_prefilling:L_prefilling+max_query_length] = static_little_full_tril_mask
        static_decoding_attention_mask[..., :, L_prefilling+max_query_length:] = 0 #attention_mask.min().item()

        self.static_little_full_tril_mask = static_little_full_tril_mask
        self.static_decoding_attention_mask = static_decoding_attention_mask
        self.query_pointer = 0
        self.key_pointer = L_prefilling
        self.prefilling_length = L_prefilling

        self.prefilling_mask = attention_mask

        return static_decoding_attention_mask
    
    def get_causal_decoding_attention_mask(self, num_new_tokens, pre_mask, kv_cache_pointer=None, **kwargs):

        if self.static_decoding_attention_mask is None:
            self._init_causal_decoding_attention_mask(pre_mask=pre_mask, **kwargs)
        
        # consider the rollback mask
        if not self.is_static_kvcache:
            rollback_q_len, rollback_k_len = pre_mask.shape[-2:]
            if rollback_k_len < self.key_pointer:
                self.key_pointer = rollback_k_len
        else:
            assert kv_cache_pointer is not None
            self.key_pointer = kv_cache_pointer

        attention_mask = self.static_decoding_attention_mask
        k_pointer = self.key_pointer
        prefilling_length = self.prefilling_length
        max_query_length = self.max_query_length

        max_val = 1 #attention_mask.max().item()
        min_val = 0 # attention_mask.min().item()

        if k_pointer+num_new_tokens > attention_mask.shape[-1]:
            attention_mask = self._extend_static_mask(num_new_tokens, 0)
            self.static_decoding_attention_mask = attention_mask

        right_lim_len = min(k_pointer+max_query_length, attention_mask.shape[-1]) - k_pointer
        
        attention_mask[..., :, :prefilling_length] = self.prefilling_mask

        attention_mask[..., :, prefilling_length:k_pointer] = max_val
        attention_mask[..., :, k_pointer:k_pointer+max_query_length].copy_(
            self.static_little_full_tril_mask[..., :right_lim_len]
        )
        attention_mask[..., :, k_pointer+max_query_length:] = min_val
        if not self.is_static_kvcache:
            attention_mask = attention_mask[..., :num_new_tokens, :k_pointer+num_new_tokens]
        else:
            attention_mask = attention_mask[..., :num_new_tokens, :]
            # attention_mask = attention_mask[..., num_new_tokens:, :]

        self.query_pointer = num_new_tokens
        self.key_pointer = k_pointer + num_new_tokens

        return attention_mask
    

class AuxiliarySuffixStaticCausalAttentionBlockMask(StaticCausalAttentionBlockMask):

    def _init_causal_decoding_attention_mask(self, **kwargs):
        attention_mask = super()._init_causal_decoding_attention_mask(**kwargs)
        self.last_key_pointer = self.key_pointer
        return attention_mask
    
    def get_causal_decoding_attention_mask(self, num_new_tokens, **kwargs):
        self.last_key_pointer = self.key_pointer
        attention_mask = super().get_causal_decoding_attention_mask(num_new_tokens, **kwargs)
        return attention_mask
    
    def get_auxiliary_suffix_causal_decoding_attention_mask(self, auxiliary_suffix_len, **kwargs):
        # find the original memory
        attention_mask = self.static_decoding_attention_mask
        k_pointer = self.key_pointer
        query_pointer = self.query_pointer
        # last_k_pointer = self.last_key_pointer
        # print(f"query_pointer: {query_pointer}, k_pointer: {k_pointer}")
        if not self.is_static_kvcache:
            attention_mask = attention_mask[..., :query_pointer+auxiliary_suffix_len, :k_pointer+auxiliary_suffix_len]
        else:
            attention_mask = attention_mask[..., :query_pointer+auxiliary_suffix_len, :]

        return attention_mask
    


# https://github.com/huggingface/transformers/blob/8c1b5d37827a6691fef4b2d926f2d04fb6f5a9e3/src/transformers/cache_utils.py#L1057C1-L1231C1
class PointerStaticCache(StaticCache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used. If you are manually setting the batch size, make sure to take into account the number of beams if you are running beam search
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = StaticCache(config=model.config, batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        StaticCache()
        ```

        cache_shape (self.max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
    """

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config,
        batch_size: int = None,
        max_cache_len: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
        max_batch_size: Optional[int] = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__(
            config=config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            max_batch_size=max_batch_size,
            layer_device_map=layer_device_map,
        )

        self.position_pointers = dict()
    
    def reset(self):
        super().reset()
        self.position_pointers = dict()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        key_states = key_states.to(k_out.dtype)
        value_states = value_states.to(v_out.dtype)

        L = key_states.shape[-2]

        pointers = self.position_pointers
        if layer_idx not in pointers:
            pointers[layer_idx] = 0

        pointer = pointers[layer_idx]
        k_out[..., pointer:pointer+L, :].copy_(key_states)
        v_out[..., pointer:pointer+L, :].copy_(value_states)

        # if cache_position is None:
        #     k_out.copy_(key_states)
        #     v_out.copy_(value_states)
        # else:
        #     # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
        #     # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
        #     # operation, that avoids copies and uses less memory.
        #     try:
        #         k_out.index_copy_(2, cache_position, key_states)
        #         v_out.index_copy_(2, cache_position, value_states)
        #     except NotImplementedError:
        #         # The operator 'aten::index_copy.out' is not currently implemented for the MPS device.
        #         k_out[:, :, cache_position] = key_states
        #         v_out[:, :, cache_position] = value_states

        pointer = pointer + L
        self.position_pointers[layer_idx] = pointer
        return k_out, v_out


    def delete_false_key_value(
        self,
        num_of_false_tokens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            for layer_idx in range(len(self.key_cache)):
                self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]
        """
        for layer_idx, pointer in self.position_pointers.items():
            pointer = max(0, pointer - num_of_false_tokens)
            self.position_pointers[layer_idx] = pointer

    # def get_max_length(self) -> Optional[int]:
    #     return self.get_max_cache_shape()

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        # max_length = self.get_max_cache_shape()
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        # if max_length is not None and previous_seq_length > max_length - new_seq_length:
        #     return max_length - new_seq_length
        # return previous_seq_length
        max_length = max_length if max_length is not None else previous_seq_length + new_seq_length
        return min(previous_seq_length, max_length - new_seq_length)