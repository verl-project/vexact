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

"""
KV Cache Context for Flex Attention Integration

This module provides a global context system to pass KV cache information
to flex_attention, enabling efficient paged attention with block-based cache management.
"""

import threading
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import triton
import triton.language as tl


# Thread-local storage for context
_context_storage = threading.local()


@dataclass(frozen=True)
class KVCacheInfo:
    """Configuration for building a KV cache manager."""

    layer_idxs: Iterable[int]
    num_kv_heads: int
    head_dim: int
    page_size: int
    max_blocks: int
    device: torch.device
    dtype: torch.dtype = torch.bfloat16


@dataclass
class KVCacheContext:
    """
    Context holding KV cache information for flex_attention.

    This context determines whether to use paged attention or packed (non-paged) attention.
    """

    is_paged_attn: bool = False
    """If True, paged attention is used. Otherwise, packed attention is used."""

    key_cache: Optional[dict[int, torch.Tensor]] = None
    """The paged key cache.
       Shape: (num_blocks, page_size, num_kv_heads, head_dim)
       This layout ensures proper stride for flash_attn paged attention."""
    value_cache: Optional[dict[int, torch.Tensor]] = None
    """The paged value cache.
       Shape: (num_blocks, page_size, num_kv_heads, head_dim)
       This layout ensures proper stride for flash_attn paged attention."""
    block_tables: Optional[torch.Tensor] = None
    """Maps logical block indices to physical block indices for each sequence.
       The value is the index of key/value cache.
       Shape: (batch_size, max_num_blocks_per_seq)."""
    slot_mapping: Optional[torch.Tensor] = None
    """Maps each query token's position to a physical slot in the KV cache. We wil use it
       to store kv value into corresponding kv cache. -1 vlaue is invalid, since we must
       store kv value.
       TODO: support -1 value.
       Any example value is [5,3,2] indicates that the first kv maps to the 5th slots,
       the second kv maps to the 3rd slots, and the third kv maps to the 2nd slots.
       Shape: (total_num_query_tokens,)."""

    # --- Packed & Paged Attention Parameters ---
    context_lens: Optional[torch.Tensor] = None
    """The total length of each sequence (past_len + new_tokens).
       Used in paged attention to determine valid token ranges.
       Shape: (batch_size,)."""
    query_start_loc: Optional[torch.Tensor] = None
    """The start index of each sequence's query tokens in the flattened query tensor.
       Used for both paged and packed attention to associate queries with sequences.
       Its format is like [0,3,6], which indicates the first 3 tokens belong to the first sequence,
       and the next 3 tokens belong to the second sequence.
       Shape: (batch_size + 1,)."""
    max_seqlen_q: Optional[int] = None
    """Pre-computed maximum query length for the current batch.
       When set, avoids recomputing and synchronizing max on the forward path."""

    def __post_init__(self):
        """Validate context parameters"""
        if self.is_paged_attn:
            assert self.key_cache is not None, "key_cache required for paged attention"
            assert self.value_cache is not None, "value_cache required for paged attention"
            assert self.block_tables is not None, "block_tables required for paged attention"
            assert self.context_lens is not None, "context_lens required for paged attention"
        else:
            assert self.query_start_loc is not None, "query_start_loc required for packed attention"
            assert self.context_lens is not None, "context_lens required for packed attention"


def set_kv_cache_context(
    is_paged_attn: bool = False,
    key_cache: Optional[torch.Tensor] = None,
    value_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    context_lens: Optional[torch.Tensor] = None,
    slot_mapping: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
):
    """Set the global KV cache context for flex_attention"""
    context = KVCacheContext(
        is_paged_attn=is_paged_attn,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        slot_mapping=slot_mapping,
        query_start_loc=query_start_loc,
        max_seqlen_q=max_seqlen_q,
    )
    _context_storage.kv_cache_context = context


def get_kv_cache_context() -> Optional[KVCacheContext]:
    """Get the current KV cache context"""
    return getattr(_context_storage, "kv_cache_context", None)


def has_kv_cache_context() -> bool:
    """Check if KV cache context is set"""
    return hasattr(_context_storage, "kv_cache_context")


class KVCacheManager:
    """
    KV Cache Manager for continuous batching with flex_attention support

    This class manages block-based KV cache allocation and provides utilities
    for setting up the context needed by flex_attention.
    """

    def __init__(self, cache_config):
        """
        Initialize KV cache manager.

        Args:
            cache_config: CacheConfig from vexact.config
        """
        self.cache_config = cache_config

        # Block allocation tracking
        self.allocated_blocks = set()
        self.free_blocks_set = set(range(cache_config.max_cache_blocks))

    def _num_blocks_needed(self, num_tokens: int) -> int:
        """
        Calculate the number of KV cache blocks needed for a given number of tokens.

        Args:
            num_tokens: Total number of tokens (e.g., max_length from generation config)

        Returns:
            Number of blocks required
        """
        return (num_tokens + self.cache_config.page_size - 1) // self.cache_config.page_size

    def allocate_blocks(self, total_tokens: int, num_current_blocks: int = 0) -> list[int] | None:
        """
        Ensure KV cache coverage for total_tokens given num_current_blocks already allocated.

        Args:
            total_tokens: Total number of tokens that need block coverage
            num_current_blocks: Number of blocks already allocated for this request

        Returns:
            List of newly allocated block IDs (may be empty if already sufficient),
            or None if not enough free blocks (OOM).
        """
        num_needed = self._num_blocks_needed(total_tokens)
        delta = num_needed - num_current_blocks

        if delta <= 0:
            return []

        if len(self.free_blocks_set) < delta:
            return None

        allocated = []
        for _ in range(delta):
            block_id = min(self.free_blocks_set)
            self.free_blocks_set.remove(block_id)
            self.allocated_blocks.add(block_id)
            allocated.append(block_id)

        return allocated

    def num_free_blocks(self) -> int:
        """Return the number of free KV cache blocks."""
        return len(self.free_blocks_set)

    def num_allocated_blocks(self) -> int:
        """Return the number of allocated KV cache blocks."""
        return len(self.allocated_blocks)

    def free_blocks(self, block_ids: list[int]):
        """Free allocated blocks"""
        for block_id in block_ids:
            if block_id in self.allocated_blocks:
                self.allocated_blocks.remove(block_id)
                self.free_blocks_set.add(block_id)


class KVCacheStore:
    """Owns key/value cache tensors and provides helpers for attention contexts."""

    def __init__(self, hf_config, cache_config, device: torch.device):
        """
        Initialize KV cache store.

        Args:
            hf_config: HuggingFace model config
            cache_config: CacheConfig from vexact.config
            device: Device to allocate tensors on
        """
        self.hf_config = hf_config
        self.cache_config = cache_config
        self.device = device
        self.page_size = cache_config.page_size
        self.max_blocks = cache_config.max_cache_blocks

        num_kv_heads = hf_config.num_key_value_heads
        # For MLA models (e.g., DeepSeek V3), key uses qk_head_dim (192) while
        # value uses v_head_dim (128). FA4 natively supports different K/V head dims.
        default_head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        k_head_dim = getattr(hf_config, "qk_head_dim", default_head_dim)
        v_head_dim = getattr(hf_config, "v_head_dim", k_head_dim)

        k_shape = (cache_config.max_cache_blocks, cache_config.page_size, num_kv_heads, k_head_dim)
        v_shape = (cache_config.max_cache_blocks, cache_config.page_size, num_kv_heads, v_head_dim)
        dtype = getattr(hf_config, "torch_dtype", torch.bfloat16)

        self.key_cache: dict[int, torch.Tensor] = {}
        self.value_cache: dict[int, torch.Tensor] = {}
        for layer_idx in range(hf_config.num_hidden_layers):
            self.key_cache[layer_idx] = torch.zeros(k_shape, device=device, dtype=dtype)
            self.value_cache[layer_idx] = torch.zeros(v_shape, device=device, dtype=dtype)


# from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


@triton.jit
def _store_cache_kernel(
    src_ptr,
    src_stride,
    cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Triton kernel for storing a tensor into a paged cache."""
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    src = tl.load(src_ptr + idx * src_stride + offsets, mask=mask)
    tl.store(cache_ptr + slot * D + offsets, src, mask=mask)


def _store_single_cache(
    tensor: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Store a (N, num_heads, head_dim) tensor into paged cache."""
    N, num_heads, head_dim = tensor.shape
    D = num_heads * head_dim
    assert tensor.stride(-1) == 1
    assert tensor.stride(1) == head_dim
    assert cache.stride(1) == D, f"Cache stride mismatch: expected stride(1)={D}, got {cache.stride(1)}"
    BLOCK_D = triton.next_power_of_2(D)
    _store_cache_kernel[(N,)](tensor, tensor.stride(0), cache, slot_mapping, D, BLOCK_D)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    Store key-value pairs into the KV cache using Triton kernel.
    Supports different head_dim for key and value (e.g., MLA models).

    Args:
        key: Key tensor of shape (N, num_heads, k_head_dim)
        value: Value tensor of shape (N, num_heads, v_head_dim)
        k_cache: Key cache tensor of shape (num_blocks, block_size, num_heads, k_head_dim)
        v_cache: Value cache tensor of shape (num_blocks, block_size, num_heads, v_head_dim)
        slot_mapping: Slot mapping tensor of shape (N,) indicating which cache slot each token goes to
    """
    assert slot_mapping.numel() == key.shape[0]
    _store_single_cache(key, k_cache, slot_mapping)
    _store_single_cache(value, v_cache, slot_mapping)
