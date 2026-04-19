# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import triton.profiler as proton

from vexact.batch_invariant_ops.kv_cache_context import store_kvcache


@proton.scope("flash_attention_forward")
def flash_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    sliding_window: Optional[int] = None,
    use_cute: bool = False,
    **kwargs,
):
    """
    Flash Attention forward pass that supports both prefill and decode phases.

    This function matches the interface of flex_attention.py but uses flash attention
    varlen for both prefill and decode. It relies on a global context managed by
    `kv_cache_context` to access the KV cache and associated metadata.

    Args:
        module (nn.Module): The attention module (used to access config like num_key_value_groups).
        query (torch.Tensor): The query tensor.
            Shape: (batch_size, num_attention_heads, query_len, head_dim).
        key (torch.Tensor): The key tensor for the *current* tokens.
            - For prefill, this contains the full key sequence.
            - For decode, this contains only the new keys to be added to the cache.
            Shape: (batch_size, num_key_value_heads, key_len, head_dim).
        value (torch.Tensor): The value tensor for the *current* tokens. Similar to `key`.
            Shape: (batch_size, num_key_value_heads, value_len, head_dim).
        attention_mask (Optional[torch.Tensor]): An optional attention mask. Not actively used.
        scaling (float): The scaling factor for the attention scores.
        dropout (float): The dropout rate. Not currently used.

    Returns:
        Tuple[torch.Tensor, None]: A tuple containing the attention output and None.
            The attention output has a shape of (total_query_tokens, num_attention_heads, head_dim).
    """
    from .kv_cache_context import get_kv_cache_context

    # window_size = (-1, -1)
    # if sliding_window is not None:
    #     window_size = (sliding_window, -1)

    # Get KV cache context
    kv_context = get_kv_cache_context()
    if kv_context is None:
        raise RuntimeError("KV cache context not set. Call set_kv_cache_context() before forward pass.")

    # Transpose to (B, S, H, D) format
    query = query.transpose(1, 2).contiguous()  # (B, H, Q, D) -> (B, Q, H, D)
    key = key.transpose(1, 2).contiguous()  # (B, nKVH, K, D) -> (B, K, nKVH, D)
    value = value.transpose(1, 2).contiguous()  # (B, nKVH, V, D) -> (B, V, nKVH, D)

    _, q_len, num_attention_heads, head_dim = query.shape
    assert key.ndim == 4
    _, _, num_key_value_heads, _ = key.shape

    # num_key_value_groups = num_attention_heads // num_key_value_heads

    if not kv_context.is_paged_attn:
        raise ValueError(
            "flash_attention_forward only supports paged attention mode. Set is_paged_attn=True in kv_cache_context."
        )

    # Extract layer index from module to access the correct layer's cache
    assert hasattr(module, "layer_idx"), "Module must have 'layer_idx' attribute for paged attention"
    layer_idx = module.layer_idx
    # print(f"layer_idx: {layer_idx}")

    # Extract layer-specific cache
    assert kv_context.key_cache[layer_idx].ndim == 4, (
        f"Expected 4D cache (num_blocks, page_size, nKVH, D), got {kv_context.key_cache.ndim}D"
    )
    assert kv_context.value_cache[layer_idx].ndim == 4, (
        f"Expected 4D cache (num_blocks, page_size, nKVH, D), got {kv_context.value_cache.ndim}D"
    )

    # Get layer-specific cache: (num_blocks, page_size, nKVH, D)
    key_cache_blocks = kv_context.key_cache[layer_idx]
    value_cache_blocks = kv_context.value_cache[layer_idx]

    # Get metadata
    block_tables = kv_context.block_tables  # (B, max_num_blocks_per_seq)
    context_lens = kv_context.context_lens  # (B,)
    # past_lens = kv_context.past_lens
    slot_mapping = kv_context.slot_mapping  # (total_query_tokens,)
    query_start_loc = kv_context.query_start_loc  # (B+1,)
    # page_size = kv_context.page_size

    # Store new key-value pairs into cache if provided
    if key is not None and value is not None:
        # Flatten batch dimension: (B, K, nKVH, D) -> (total_tokens, nKVH, D)
        # Key and value may have different head_dim (e.g., MLA: qk_head_dim=192, v_head_dim=128)
        key_flat = key.reshape(-1, num_key_value_heads, head_dim)
        v_head_dim = value.shape[-1]
        value_flat = value.reshape(-1, num_key_value_heads, v_head_dim)

        store_kvcache(key_flat, value_flat, key_cache_blocks, value_cache_blocks, slot_mapping)

    # Prepare query: (B, Q, H, D) -> (total_query_tokens, H, D)
    query_flat = query.reshape(-1, num_attention_heads, head_dim)

    # Use flash_attn_varlen_func for both prefill and decode
    # batch_size = len(context_lens)

    # Compute cumulative sequence lengths for varlen interface
    cu_seqlens_q = query_start_loc.to(torch.int32)
    # cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=query.device)
    # cu_seqlens_k[1:] = context_lens.cumsum(0)

    max_seqlen_q = kv_context.max_seqlen_q
    assert max_seqlen_q is not None, "need max_seqlen_q precomputed in kv context"
    # Fallback for callers that haven't precomputed max_seqlen_q
    # max_seqlen_q = int((query_start_loc[1:] - query_start_loc[:-1]).max().item())
    # max_seqlen_k = context_lens.max().item()
    from vexact.utils.device import DEVICE_MAJOR

    if use_cute:
        assert DEVICE_MAJOR >= 9, f"FA4 (flash_attn.cute) requires SM90+, got SM{DEVICE_MAJOR}0"
        from flash_attn.cute import flash_attn_varlen_func

        attn_output, _ = flash_attn_varlen_func(
            q=query_flat,  # (total_q, nH, D)
            k=key_cache_blocks,  # (num_blocks, page_size, nKVH, D)
            v=value_cache_blocks,  # (num_blocks, page_size, nKVH, D_v)
            cu_seqlens_q=cu_seqlens_q,  # (B+1,) int32
            cu_seqlens_k=None,  # MUST be None when page_table is used
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=None,  # derived internally from num_pages * page_size
            seqused_k=context_lens,  # (B,) int32 — valid KV tokens per seq (replaces cache_seqlens)
            page_table=block_tables,  # (B, max_num_blocks_per_seq) int32
            softmax_scale=scaling,
            causal=True,
            num_splits=1,
        )
    else:
        assert DEVICE_MAJOR == 9, f"FA3 requires SM90, got SM{DEVICE_MAJOR}0"
        from flash_attn_interface import flash_attn_with_kvcache

        attn_output = flash_attn_with_kvcache(
            query_flat,
            key_cache_blocks,
            value_cache_blocks,
            cu_seqlens_q=cu_seqlens_q,
            # cu_seqlens_k=cu_seqlens_k,
            cache_seqlens=context_lens,
            max_seqlen_q=max_seqlen_q,
            # max_seqlen_k=max_seqlen_k,
            softmax_scale=scaling,
            causal=True,
            page_table=block_tables,
            # window_size=window_size
            num_splits=1,
        )
    # Output shape: (H, total_query_tokens, D)
    return attn_output, None


# FA4 (flash_attn.cute) variant — forces use_cute=True on both Hopper and Blackwell
flash_attention_forward_cute = partial(flash_attention_forward, use_cute=True)
