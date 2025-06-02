from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
import einops
import math

from diffusers.utils import deprecate
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

class SeparateKVAttnProcessor(AttnProcessor2_0):
    r"""
    Allows keys and values to be computed based on different encodings
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        value_encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # breakpoint()
            # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            encoder_attention_mask = attention_mask
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if value_encoder_hidden_states is None:
            value_encoder_hidden_states = encoder_hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(value_encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        if attention_mask is not None or encoder_attention_mask is not None:
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            B, N_q, _ = hidden_states.shape
            _, N_k, _ = encoder_hidden_states.shape
            device = hidden_states.device

            if attention_mask is None:
                attention_mask = torch.ones((B, N_q), dtype=torch.bool, device=device)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones((B, N_k), dtype=torch.bool, device=device)
            
            attention_mask = einops.rearrange(attention_mask, 'b l -> b l 1')
            encoder_attention_mask = einops.rearrange(encoder_attention_mask, 'b s -> b 1 s')
            combined_attention_mask = torch.logical_and(attention_mask, encoder_attention_mask)

            attention_mask = einops.repeat(combined_attention_mask, 'B L S -> B H L S', H=attn.heads)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
