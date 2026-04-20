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
        "mode": None,  # Default mode for best performance
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


_KERNEL_BLOCK_SIZE = 64
_KERNEL_OPTIONS = {"BLOCK_M": _KERNEL_BLOCK_SIZE, "BLOCK_N": _KERNEL_BLOCK_SIZE}


def get_kernel_options():
    """Get kernel options for FlexAttention with fixed block sizes."""
    return _KERNEL_OPTIONS


def get_kernel_block_size():
    """Get the fixed kernel block size for FlexAttention execution."""
    return _KERNEL_BLOCK_SIZE


def get_mask_mod(
    *,
    context_lens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    q_len: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    """
    Returns a `mask_mod(b, h, q_idx, kv_idx) -> bool` compatible with:
    - causal attention (default)

    Notes:
    - `kv_offset[b]` (derived from context_lens and query_start_loc) offsets q positions
      for decode (absolute_q = kv_offset[b] + q_idx).
    - `context_lens[b]` caps kv_idx (kv_idx < context_lens[b]).
    """

    # Derive kv_offset (replaces past_lens): number of KV entries before this step's tokens
    kv_offset = None
    if query_start_loc is not None and context_lens is not None:
        tokens_per_seq = query_start_loc[1:] - query_start_loc[:-1]
        kv_offset = context_lens - tokens_per_seq  # shape: (num_seqs,)

    # Precompute chunk mapping for packed chunks
    chunk_mapping = None
    if query_start_loc is not None and kv_offset is not None and query_start_loc.shape[0] > 2 and q_len is not None:
        # Optimized vectorized implementation:
        lengths = query_start_loc[1:] - query_start_loc[:-1]
        chunk_ids = torch.arange(lengths.shape[0], device=device, dtype=torch.int64)
        chunk_mapping = torch.repeat_interleave(chunk_ids, lengths)

    def mask_mod(b, h, q_idx, kv_idx):
        # All args are torch.Tensors (often int32/int64 scalars). Keep tensor ops only.
        if chunk_mapping is not None:
            chunk_idx = chunk_mapping[q_idx]
            offset = kv_offset[chunk_idx]
            q_start = query_start_loc[chunk_idx]
            abs_q = offset + (q_idx - q_start)

            ok = kv_idx <= abs_q  # causal
            if context_lens is not None:
                ok = ok & (kv_idx < context_lens[chunk_idx])
        else:
            abs_q = q_idx if kv_offset is None else (kv_offset[b] + q_idx)

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


_compile_cache: dict = {}


def _get_compiled_flex_attention(
    *,
    dynamic: bool,
    mode: Optional[str],
    fullgraph: bool,
    backend: Optional[str],
    page_flag: bool = False,
):
    """Return a cached `torch.compile(flex_attention, ...)` wrapper."""
    cache_key = ("flex", dynamic, mode, fullgraph, backend, page_flag)
    if cache_key not in _compile_cache:
        kwargs = {"dynamic": dynamic, "fullgraph": fullgraph}
        if mode is not None:
            kwargs["mode"] = mode
        if backend is not None:
            kwargs["backend"] = backend
        _compile_cache[cache_key] = torch.compile(flex_attention, **kwargs)
    return _compile_cache[cache_key]


_paged_block_mask_cache: dict = {}


def _prepare_paged_kv(ctx, query, key, value, layer_idx):
    """
    Prepare KV tensors and block mask for paged attention.

    Uses zero-copy view of the physical cache and caches the block mask
    across layers within the same inference step.
    """
    block_tables: torch.Tensor = ctx.block_tables
    context_lens: torch.Tensor = ctx.context_lens
    slot_mapping: torch.Tensor = ctx.slot_mapping
    query_start_loc: Optional[torch.Tensor] = ctx.query_start_loc

    key_cache: torch.Tensor = ctx.key_cache
    value_cache: torch.Tensor = ctx.value_cache

    bsz, q_heads, q_len, head_dim = query.shape
    num_blocks, page_size, kv_heads, _ = key_cache[layer_idx].shape
    total_slots = num_blocks * page_size

    if key is not None:
        # key/value come in (B, H, L, D) layout from HF's attention transpose.
        # Collapse to (B*L, H, D) for slot-mapped cache store. Without the explicit
        # .transpose(1, 2), reshape(-1, H, D) on the non-contiguous transposed view
        # collapses B*H instead of B*L whenever L == H, giving wrong semantics.
        key_flat = key.transpose(1, 2).reshape(-1, kv_heads, head_dim)
        value_flat = value.transpose(1, 2).reshape(-1, kv_heads, head_dim)
        store_kvcache(key_flat, value_flat, key_cache[layer_idx], value_cache[layer_idx], slot_mapping)

    # Zero-copy view: (num_blocks, page_size, H, D) -> (1, H, total_slots, D)
    k_reshaped = key_cache[layer_idx].view(total_slots, kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
    v_reshaped = value_cache[layer_idx].view(total_slots, kv_heads, head_dim).transpose(0, 1).unsqueeze(0)

    # Cache block_mask per context (reusable across layers within same step)
    cache_key = (id(ctx), q_len)
    if cache_key not in _paged_block_mask_cache:
        _paged_block_mask_cache.clear()

        # block_tables/context_lens are keyed by per-seq index, but the flex
        # attention call's batch dim (bsz) may be packed to 1 across multiple
        # seqs. Use num_seqs for the per-seq mapping and derive seq_id from
        # q_idx via query_start_loc when packed.
        num_seqs = block_tables.shape[0]
        max_num_blocks = block_tables.shape[1]
        device = query.device
        physical_to_logical = torch.full((num_seqs, num_blocks), -1, dtype=torch.int64, device=device)
        logical_grid = torch.arange(max_num_blocks, device=device, dtype=torch.int64).unsqueeze(0).expand(num_seqs, -1)
        valid_mask = block_tables >= 0
        batch_idx = torch.arange(num_seqs, device=device).unsqueeze(1).expand_as(block_tables)
        physical_to_logical[batch_idx[valid_mask], block_tables[valid_mask]] = logical_grid[valid_mask]

        # chunk_mapping[q_idx] -> seq_id is only meaningful when queries from
        # multiple seqs are packed into bsz=1 (q_idx iterates across seqs).
        # When bsz > 1 each seq already has its own flex batch dim and `b` is
        # the seq id -- q_idx is per-batch-local and must not index chunk_mapping.
        chunk_mapping = None
        kv_offset = None
        if bsz == 1 and query_start_loc is not None and query_start_loc.shape[0] > 2:
            lengths = query_start_loc[1:] - query_start_loc[:-1]
            chunk_ids = torch.arange(lengths.shape[0], device=device, dtype=torch.int64)
            chunk_mapping = torch.repeat_interleave(chunk_ids, lengths)
            kv_offset = context_lens - lengths  # per-seq start of this step in the KV cache

        def physical_mask_mod(b, h, q_idx, physical_kv_idx):
            if chunk_mapping is not None:
                seq_id = chunk_mapping[q_idx]
                abs_q = kv_offset[seq_id] + (q_idx - query_start_loc[seq_id])
            else:
                seq_id = b
                abs_q = (context_lens[seq_id] - q_len) + q_idx

            physical_block = physical_kv_idx // page_size
            block_offset = physical_kv_idx % page_size
            logical_block = physical_to_logical[seq_id, physical_block]
            logical_kv_idx = logical_block * page_size + block_offset

            ok = logical_kv_idx <= abs_q
            if context_lens is not None:
                ok = ok & (logical_kv_idx < context_lens[seq_id])
            return torch.where(logical_block >= 0, ok, False)

        _paged_block_mask_cache[cache_key] = create_block_mask(
            physical_mask_mod,
            bsz,
            q_heads,
            q_len,
            total_slots,
            BLOCK_SIZE=get_kernel_block_size(),
        )

    return k_reshaped, v_reshaped, _paged_block_mask_cache[cache_key]


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
        create_block_mask_fn = create_block_mask

    else:
        flex_attn_fn = flex_attention
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
        query_start_loc = ctx.query_start_loc

        block_mask = create_block_mask_fn(
            get_mask_mod(
                context_lens=context_lens,
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

        # Gather KV and build mask outside compile, then run compiled flex_attention
        gathered_k, gathered_v, block_mask = _prepare_paged_kv(ctx, query, key, value, layer_idx)
        attn_output = flex_attn_fn(
            query,
            gathered_k,
            gathered_v,
            block_mask=block_mask,
            scale=scaling,
            kernel_options=get_kernel_options(),
            enable_gqa=True,
        )

    attn_output = _flatten_attn_output(attn_output)

    return attn_output, None
