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

from typing import Optional

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from vexact.batch_invariant_ops.kv_cache_context import get_kv_cache_context, store_kvcache


# Global configuration for compilation
FLEX_ATTENTION_USE_COMPILE = False

# Compilation configuration presets for different modes
# Each preset is optimized for specific scenarios (decode vs prefill)
COMPILE_CONFIG_PRESETS = {
    # Decode mode: single token, fixed shapes, prioritize latency
    "decode": {
        "fullgraph": False,  # Stable shape -> full graph compilation
        "mode": None,  # Minimize overhead (latency critical)
        "backend": None,  # Let PyTorch choose optimal backend
        "dynamic": True,  # Shape is always (B, H, 1, D)
    },
    # Prefill mode: variable shapes, prioritize throughput
    "prefill": {
        "fullgraph": False,  # Variable shape -> avoid recompilation
        "mode": None,  # Autotune for best performance
        "backend": None,
        "dynamic": True,  # Shapes vary per step
    },
    # Alternative decode: with dynamic shapes (if needed)
    "decode_dynamic": {
        "fullgraph": False,
        "mode": "reduce-overhead",
        "backend": None,
        "dynamic": True,
    },
    # Conservative prefill: balance between perf and compilation time
    "prefill_conservative": {
        "fullgraph": False,
        "mode": "reduce-overhead",
        "backend": None,
        "dynamic": True,
    },
}


def get_kernel_options():
    """Get kernel options for FlexAttention with fixed block sizes.

    The kernel execution block size is always fixed at 16 for optimal performance,
    independent of the page_size used for KV cache organization.
    """
    kernel_options = {}
    # Fixed kernel block size for optimal performance
    block_size = 128
    kernel_options["BLOCK_M"] = block_size
    kernel_options["BLOCK_N"] = block_size
    assert kernel_options["BLOCK_M"] >= block_size and kernel_options["BLOCK_N"] >= block_size, (
        "BLOCK_M and BLOCK_N must be >= 128"
    )
    # kernel_options["IS_DIVISIBLE"] = False
    # TODO: add back IS_DIVISIBLE
    return kernel_options


def get_kernel_block_size():
    """Get the fixed kernel block size for FlexAttention execution.

    Returns:
        int: The kernel block size (always 16)
    """
    return 128


def get_mask_mod(
    *,
    context_lens: Optional[torch.Tensor] = None,
    past_lens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    q_len: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    """
    Returns a `mask_mod(b, h, q_idx, kv_idx) -> bool` compatible with:
    - causal attention (default)

    Notes:
    - `past_lens[b]` offsets q positions for decode (absolute_q = past_lens[b] + q_idx).
    - `context_lens[b]` caps kv_idx (kv_idx < context_lens[b]).
    """

    # Precompute chunk mapping for packed chunks
    chunk_mapping = None
    if query_start_loc is not None and past_lens is not None and query_start_loc.shape[0] > 2 and q_len is not None:
        # Optimized vectorized implementation:
        lengths = query_start_loc[1:] - query_start_loc[:-1]
        chunk_ids = torch.arange(lengths.shape[0], device=device, dtype=torch.int64)
        chunk_mapping = torch.repeat_interleave(chunk_ids, lengths)

    def mask_mod(b, h, q_idx, kv_idx):
        # All args are torch.Tensors (often int32/int64 scalars). Keep tensor ops only.
        if chunk_mapping is not None:
            chunk_idx = chunk_mapping[q_idx]
            p_len = past_lens[chunk_idx]
            q_start = query_start_loc[chunk_idx]
            abs_q = p_len + (q_idx - q_start)

            ok = kv_idx <= abs_q  # causal
            if context_lens is not None:
                ok = ok & (kv_idx < context_lens[chunk_idx])
        else:
            abs_q = q_idx if past_lens is None else (past_lens[b] + q_idx)

            ok = kv_idx <= abs_q  # causal
            if context_lens is not None:
                ok = ok & (kv_idx < context_lens[b])
        return ok

    return mask_mod


def get_compile_config(is_decode: bool, preset: str = "default") -> dict:
    """
    Get the compile configuration for the current attention mode.

    Args:
        is_decode (bool): Whether this is a decode step (q_len==1) or prefill.
        preset (str): Configuration preset to use. Options:
            - "default": Auto-select based on is_decode
            - "decode": Optimized for single-token decode (low latency)
            - "prefill": Optimized for multi-token prefill (high throughput)
            - "decode_dynamic": Decode with dynamic shapes
            - "prefill_conservative": Conservative prefill settings

    Returns:
        dict: Configuration dict with keys: dynamic, mode, fullgraph, backend

    Example:
        >>> cfg = get_compile_config(is_decode=True)
        >>> # Will return decode-optimized config: fullgraph=True, mode="reduce-overhead"
        >>> cfg = get_compile_config(is_decode=False, preset="prefill")
        >>> # Will return prefill config: fullgraph=False, mode="max-autotune"
    """
    if preset == "default":
        preset = "decode" if is_decode else "prefill"

    if preset not in COMPILE_CONFIG_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(COMPILE_CONFIG_PRESETS.keys())}")

    return COMPILE_CONFIG_PRESETS[preset].copy()


def _get_compiled_flex_attention(
    *,
    dynamic: bool,
    mode: Optional[str],
    fullgraph: bool,
    backend: Optional[str],
    page_flag: bool = False,
):
    """Return a cached `torch.compile(flex_attention, ...)` wrapper."""
    kwargs = {"dynamic": dynamic, "fullgraph": fullgraph}
    if mode is not None:
        kwargs["mode"] = mode
    if backend is not None:
        kwargs["backend"] = backend

    return torch.compile(flex_attention, **kwargs)


def _get_compiled_flex_attention_paged(
    *,
    dynamic: bool,
    mode: Optional[str],
    fullgraph: bool,
    backend: Optional[str],
    page_flag: bool = False,
):
    """Return a cached `torch.compile(flex_attention, ...)` wrapper."""
    kwargs = {"dynamic": dynamic, "fullgraph": fullgraph}
    if mode is not None:
        kwargs["mode"] = mode
    if backend is not None:
        kwargs["backend"] = backend

    return torch.compile(flex_paged_attention_core, **kwargs)


def _get_compiled_create_block_mask(
    *,
    dynamic: bool,
    mode: Optional[str],
    fullgraph: bool,
    backend: Optional[str],
):
    """
    Return a cached `torch.compile(create_block_mask, ...)` wrapper.

    Note: Only enable if your mask_mod is stable across calls; otherwise Dynamo may recompile often.
    """
    kwargs = {"dynamic": dynamic, "fullgraph": fullgraph}
    if mode is not None:
        kwargs["mode"] = mode
    if backend is not None:
        kwargs["backend"] = backend
    return torch.compile(create_block_mask, **kwargs)


def flex_paged_attention_core(ctx, query, key, value, layer_idx, scaling):
    """
    Core paged attention computation that can be compiled.

    Returns:
        query, k, v, block_mask
    """
    block_tables: torch.Tensor = ctx.block_tables
    context_lens: torch.Tensor = ctx.context_lens
    past_lens: torch.Tensor = ctx.past_lens
    slot_mapping: torch.Tensor = ctx.slot_mapping
    # query_start_loc: torch.Tensor = ctx.query_start_loc

    key_cache: torch.Tensor = ctx.key_cache
    value_cache: torch.Tensor = ctx.value_cache

    bsz, q_heads, q_len, head_dim = query.shape
    num_blocks, page_size, kv_heads, _ = key_cache[layer_idx].shape
    total_slots = num_blocks * page_size
    device = query.device

    if key is not None:
        _, kv_heads, s_new, _ = key.shape
        # Flatten batch dimension: (B, K, nKVH, D) -> (total_tokens, nKVH, D)
        key_flat = key.reshape(-1, kv_heads, head_dim)
        value_flat = value.reshape(-1, kv_heads, head_dim)
        store_kvcache(key_flat, value_flat, key_cache[layer_idx], value_cache[layer_idx], slot_mapping)

    # Reshape cache to (1, H, total_slots, D) for flex_attention
    # (num_blocks, page_size, H, D) -> (1, H, total_slots, D)
    k_reshaped = (
        key_cache[layer_idx]
        .view(num_blocks * page_size, kv_heads, head_dim)  # (total_slots, H, D)
        .transpose(0, 1)  # (H, total_slots, D)
        .unsqueeze(0)  # (1, H, total_slots, D)
    )
    v_reshaped = (
        value_cache[layer_idx]
        .view(num_blocks * page_size, kv_heads, head_dim)  # (total_slots, H, D)
        .transpose(0, 1)  # (H, total_slots, D)
        .unsqueeze(0)  # (1, H, total_slots, D)
    )

    # Build physical to logical block mapping
    max_num_blocks = block_tables.shape[1]
    num_physical_blocks = key_cache[layer_idx].shape[0]
    physical_to_logical = torch.full((bsz, num_physical_blocks), -1, dtype=torch.int64, device=device)
    logical_grid = torch.arange(max_num_blocks, device=device, dtype=torch.int64).unsqueeze(0).expand(bsz, -1)
    valid_mask = block_tables >= 0
    safe_physical_indices = torch.where(valid_mask, block_tables, torch.zeros_like(block_tables))
    physical_to_logical.scatter_(1, safe_physical_indices, logical_grid)
    mask_values = torch.where(valid_mask, logical_grid, torch.full_like(logical_grid, -1))
    physical_to_logical.scatter_(1, safe_physical_indices, mask_values)

    # Precompute chunk mapping for packed chunks
    # chunk_mapping = None
    # if query_start_loc is not None and past_lens is not None and query_start_loc.shape[0] > 2 and q_len > 1:
    #     # Optimized vectorized implementation:
    #     lengths = query_start_loc[1:] - query_start_loc[:-1]
    #     chunk_ids = torch.arange(lengths.shape[0], device=device, dtype=torch.int64)
    #     chunk_mapping = torch.repeat_interleave(chunk_ids, lengths)

    # Physical mask function
    def physical_mask_mod(b, h, q_idx, physical_kv_idx):
        """Mask function that works directly with physical KV indices in the cache"""
        # Convert physical index to logical position
        physical_block = physical_kv_idx // page_size
        block_offset = physical_kv_idx % page_size

        # Handle chunked queries
        # if chunk_mapping is not None:
        #     chunk_idx = chunk_mapping[q_idx]
        #     logical_block = physical_to_logical[chunk_idx, physical_block]
        #     logical_kv_idx = logical_block * page_size + block_offset

        #     p_len = past_lens[chunk_idx]
        #     q_start = query_start_loc[chunk_idx]
        #     abs_q = p_len + (q_idx - q_start)

        #     ok = logical_kv_idx <= abs_q
        #     if context_lens is not None:
        #         ok = ok & (logical_kv_idx < context_lens[chunk_idx])
        # else:
        logical_block = physical_to_logical[b, physical_block]
        logical_kv_idx = logical_block * page_size + block_offset

        abs_q = past_lens[b] + q_idx
        ok = logical_kv_idx <= abs_q

        if context_lens is not None:
            ok = ok & (logical_kv_idx < context_lens[b])

        return torch.where(logical_block >= 0, ok, False)

    # Create block mask with physical addressing
    block_mask = create_block_mask(
        physical_mask_mod,
        bsz,
        q_heads,
        q_len,
        total_slots,
        BLOCK_SIZE=get_kernel_block_size(),
    )

    # Run flex attention
    attn_output = flex_attention(
        query,
        k_reshaped,
        v_reshaped,
        block_mask=block_mask,
        scale=scaling,
        kernel_options=get_kernel_options(),
        enable_gqa=True,
    )

    return attn_output


def _flatten_attn_output(attn_output: torch.Tensor) -> torch.Tensor:
    """
    Flatten attention output from (B, H, Q, D) to (B*Q, H, D).
    """
    bsz, num_heads, q_len, head_dim = attn_output.shape
    return attn_output.permute(0, 2, 1, 3).reshape(bsz * q_len, num_heads, head_dim)


def flex_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    A custom attention function that leverages `torch.nn.attention.flex_attention`
    to support both standard (packed) and paged KV cache attention mechanisms.

    This function relies on a global context managed by `kv_cache_context` to access
    the appropriate KV cache and associated metadata. The behavior is determined by
    the `is_paged_attn` flag in the context.

    Args:
        module (nn.Module): The attention module (used to access config like num_key_value_groups).
        query (torch.Tensor): The query tensor.
            Shape: (batch_size, num_attention_heads, query_len, head_dim).
        key (torch.Tensor): The key tensor for the *current* tokens.
            - For packed attention, this contains the full key sequence.
            - For paged attention, this contains only the new keys to be added to the cache.
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
    # Input size: (B, H, S, D) for flex_attention
    _ = attention_mask  # this implementation relies on block_mask instead of attention_mask

    ctx = get_kv_cache_context()

    # Defaults: decode(q_len==1)->dynamic=True, prefill->dynamic=False.
    bsz, q_heads, q_len, head_dim = query.shape

    # ---------------------For compiling-------------------------#
    is_decode = q_len == 1
    use_compile = FLEX_ATTENTION_USE_COMPILE
    cfg = get_compile_config(is_decode=is_decode, preset="default")

    # Extract config parameters
    dynamic = cfg["dynamic"]
    mode = cfg["mode"]
    fullgraph = cfg["fullgraph"]
    backend = cfg["backend"]

    common = dict(dynamic=dynamic, mode=mode, fullgraph=fullgraph, backend=backend, page_flag=ctx.is_paged_attn)

    if use_compile:
        flex_attn_fn = _get_compiled_flex_attention(**common)
        flex_attn_fn_paged = _get_compiled_flex_attention_paged(**common)
        # create_block_mask_fn = _get_compiled_create_block_mask(**common)
        create_block_mask_fn = create_block_mask

    else:
        flex_attn_fn = flex_attention
        flex_attn_fn_paged = flex_paged_attention_core
        create_block_mask_fn = create_block_mask

    # ---------------------Start attention-----------------------#
    # for packed attention
    if not ctx.is_paged_attn:
        # Standard (non-paged) flex attention: directly use provided K/V.
        # Only consider block_mask (causal).
        if key is not None:
            kv_len = key.shape[2]
        else:
            kv_len = ctx.kv_len

        context_lens = ctx.context_lens
        past_lens = ctx.past_lens
        query_start_loc = ctx.query_start_loc

        block_mask = create_block_mask_fn(
            get_mask_mod(
                context_lens=context_lens,
                past_lens=past_lens,
                query_start_loc=query_start_loc,
                q_len=q_len,
                device=query.device,
            ),
            bsz,
            q_heads,
            q_len,
            kv_len,
            BLOCK_SIZE=get_kernel_block_size(),
        )

        # guarantee tensor shape (B.H,S,D)
        attn_output = flex_attn_fn(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=scaling,
            kernel_options=get_kernel_options(),
            enable_gqa=True,
        )

    # for paged attention
    else:
        if ctx is None:
            raise RuntimeError("Paged attention requested but kv_cache_context is not available.")

        layer_idx = kwargs.get("layer_idx") if "layer_idx" in kwargs else getattr(module, "layer_idx", 0)
        layer_idx = int(layer_idx)
        attn_output = flex_attn_fn_paged(ctx, query, key, value, layer_idx, scaling)

    attn_output = _flatten_attn_output(attn_output)
    return attn_output, None
