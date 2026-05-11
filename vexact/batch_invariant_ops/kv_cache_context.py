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
from collections import OrderedDict
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
    """KV cache manager with optional content-addressed prefix cache.

    Two responsibilities:
      1. Block lifetime via reference counting. Free blocks live in an LRU queue
         (least-recently-released first) and are reclaimed for new content when
         the pool runs out.
      2. Prefix cache via chained block-content hashing. When `prefix_cache_enabled`
         is True, the scheduler calls `plan_prefix_cache` per request to compute
         per-full-block chain hashes and the leading hit count. After prefill
         completes, `mark_blocks_filled` stamps the just-computed hashes onto the
         allocated blocks so future requests with the same prefix can hit them.

    Lifecycle of a cached block:
      A) plan_prefix_cache(token_ids) → (block_hashes, num_prefix_hit_blocks)
      B) commit_prefix_plan(block_hashes, num_prefix_hit_blocks, total_tokens) →
         incref hit blocks; take fresh blocks (LRU-oldest, evicting their old
         hash if any) for the remaining full blocks and the partial last block
      C) prefill completes → mark_blocks_filled(block_ids, block_hashes) stamps
         the full blocks (idempotent — safe to call every decode step)
      D) request finishes / preempts → free_blocks → decref; if 0 → push to free_lru
         (block KEEPS its hash association — still cache-eligible until evicted)
      E) future allocation runs out of fresh-tagged blocks → pops oldest from free_lru,
         drops its hash entry from the index → block becomes fresh again

    Invalidation: `clear_cache_index()` drops all hash entries (refcounts / free_lru
    untouched). It is the caller's responsibility to also preempt every in-flight
    request that owns blocks under the old KV state — those blocks' KV is stale
    and decode would read garbage. See `Scheduler.reset_for_state_change()`.

    For pp_size > 1 set prefix_cache_enabled=False — KV is replicated across PP
    ranks but only this driver-side manager hashes; coordinating cache state across
    ranks is out of scope.
    """

    # Constant seed for the chain hash. Doesn't need cryptographic strength —
    # the chain lives within a single process and Python's int hash is fine for
    # 1024-block scale (collision probability ~3e-14).
    _SEED: int = 0

    def __init__(self, cache_config, enable_prefix_cache: bool = True):
        """
        Args:
            cache_config: CacheConfig from vexact.config
            enable_prefix_cache: when True, full blocks are hashed for content-addressed
                reuse across requests. When False, behaves like the original allocator
                (always fresh blocks, no lookups). Set False for pp_size > 1.
        """
        self.cache_config = cache_config
        self.page_size: int = cache_config.page_size
        self.prefix_cache_enabled: bool = enable_prefix_cache
        self._max_blocks = cache_config.max_cache_blocks

        # refcount[bid] == 0 ⇔ bid in free_lru. Both kept in sync by _incref/_decref.
        self._refcount: dict[int, int] = {bid: 0 for bid in range(self._max_blocks)}
        self._free_lru: OrderedDict[int, None] = OrderedDict((bid, None) for bid in range(self._max_blocks))

        # Content-addressed prefix cache: chain hash → block_id. Only populated for
        # full blocks (covering exactly page_size tokens) after their request has
        # finished prefill. Both maps are cleared together (a block is either in
        # both or neither).
        self._block_hash_to_id: dict[int, int] = {}
        self._block_id_to_hash: dict[int, int] = {}

    # ---- diagnostics ----

    def num_free_blocks(self) -> int:
        return len(self._free_lru)

    def num_allocated_blocks(self) -> int:
        return self._max_blocks - len(self._free_lru)

    def num_cached_blocks(self) -> int:
        """Blocks currently registered in the prefix cache index (in-use OR free-but-cached)."""
        return len(self._block_hash_to_id)

    def _num_blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.page_size - 1) // self.page_size

    # ---- low-level ref / pool management ----

    def _incref(self, block_id: int) -> None:
        if self._refcount[block_id] == 0:
            self._free_lru.pop(block_id, None)
        self._refcount[block_id] += 1

    def _decref(self, block_id: int) -> None:
        self._refcount[block_id] -= 1
        if self._refcount[block_id] == 0:
            # Push as MOST recently freed (last). _take_free_block pops from front (oldest).
            self._free_lru[block_id] = None

    def _take_free_block(self) -> Optional[int]:
        """Pop the LRU-oldest free block; if it carried a cache hash, evict it first."""
        if not self._free_lru:
            return None
        bid = next(iter(self._free_lru))
        del self._free_lru[bid]
        old_hash = self._block_id_to_hash.pop(bid, None)
        if old_hash is not None:
            # Defensive: only delete if it still maps back to us (it should).
            if self._block_hash_to_id.get(old_hash) == bid:
                del self._block_hash_to_id[old_hash]
        return bid

    def _rollback(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            self._decref(bid)

    # ---- public allocation paths ----

    def allocate_blocks(self, total_tokens: int, num_current_blocks: int = 0) -> list[int] | None:
        """Incremental allocation for an already-active request.

        Used by the scheduler during decode and chunked-prefill steps to extend
        block coverage as num_computed_tokens grows. New blocks are taken fresh
        from the free pool — no prefix lookup, no stamping (decode tokens are
        unique to this request and would never hit the cache anyway).

        Returns newly allocated block IDs (possibly empty), or None on OOM.
        """
        num_needed = self._num_blocks_needed(total_tokens)
        delta = num_needed - num_current_blocks
        if delta <= 0:
            return []
        if len(self._free_lru) < delta:
            return None
        allocated: list[int] = []
        for _ in range(delta):
            bid = self._take_free_block()
            assert bid is not None  # checked above
            self._incref(bid)
            allocated.append(bid)
        return allocated

    def plan_prefix_cache(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Compute the prefix-cache plan for a token sequence. Stateless — no allocation.

        For each FULL block (last partial excluded) compute the chained hash and
        check whether it's currently in the cache index. The "hit" run stops at
        the first miss — a later "hit" would still require prefill to fill the
        gap, which would overwrite the cached blocks, so we don't try to exploit it.

        Returns (block_hashes, num_prefix_hit_blocks):
          - block_hashes: one chain hash per full block (`len(token_ids) // page_size`
            entries). Empty when prefix cache is disabled.
          - num_prefix_hit_blocks: length of the leading contiguous hit run.

        The scheduler reuses block_hashes verbatim in `commit_prefix_plan` and
        `mark_blocks_filled` — neither recomputes hashes.
        """
        if not self.prefix_cache_enabled:
            return [], 0

        page_size = self.page_size
        num_full = len(token_ids) // page_size
        block_hashes: list[int] = []
        num_prefix_hit_blocks = 0
        contiguous = True
        prev_hash = self._SEED

        for i in range(num_full):
            block_hash = hash((prev_hash, tuple(token_ids[i * page_size : (i + 1) * page_size])))
            block_hashes.append(block_hash)
            if contiguous and block_hash in self._block_hash_to_id:
                num_prefix_hit_blocks += 1
            else:
                contiguous = False
            prev_hash = block_hash

        return block_hashes, num_prefix_hit_blocks

    def commit_prefix_plan(
        self,
        block_hashes: list[int],
        num_prefix_hit_blocks: int,
        total_tokens: int,
    ) -> list[int] | None:
        """Commit a plan from `plan_prefix_cache`: incref hit blocks, take fresh for the rest.

        Single-threaded scheduler ⇒ cache state can't change between plan and commit,
        so the leading `num_prefix_hit_blocks` lookups via `_block_hash_to_id` are
        guaranteed to hit (the chain hash is unique to the content).

        Args:
            block_hashes: hashes from plan_prefix_cache (one per full block). May be
                empty when prefix cache is disabled or `total_tokens < page_size`.
            num_prefix_hit_blocks: leading hits from plan_prefix_cache.
            total_tokens: total tokens this request needs coverage for. Determines
                whether a partial last block is needed (always fresh).

        Returns the full block_ids list on success, or None on OOM (with full rollback).
        """
        num_blocks_needed = self._num_blocks_needed(total_tokens)
        if num_blocks_needed == 0:
            return []

        block_ids: list[int] = []

        # Cached portion: refcount existing blocks. Guaranteed to hit per the
        # single-threaded plan/commit invariant.
        for i in range(num_prefix_hit_blocks):
            cached_bid = self._block_hash_to_id[block_hashes[i]]
            self._incref(cached_bid)
            block_ids.append(cached_bid)

        # Remaining full blocks (cache misses) plus the partial last block: fresh from pool.
        for _ in range(num_prefix_hit_blocks, num_blocks_needed):
            bid = self._take_free_block()
            if bid is None:
                self._rollback(block_ids)
                return None
            self._incref(bid)
            block_ids.append(bid)

        return block_ids

    def free_blocks(self, block_ids: list[int]):
        """Decref the given blocks; those reaching zero rejoin free_lru (still hashed)."""
        for bid in block_ids:
            if self._refcount.get(bid, 0) > 0:
                self._decref(bid)

    # ---- prefix cache management ----

    def mark_blocks_filled(self, block_ids: list[int], block_hashes: list[int]) -> None:
        """Stamp full blocks with the chain hashes computed by `plan_prefix_cache`.

        Called every decode step (and once at prefill completion) — the fast-path
        check on the last full block (which is the last to be stamped, and refcounted
        by us so no one else can rewrite it) makes repeat calls O(1).

        Why the partial last block is never stamped (intentional, not just an omission):
          1. Chain hash is sensitive to tuple length: `hash((prev, tuple(partial)))`
             differs from `hash((prev, tuple(full)))`, so a future longer request
             couldn't hit a partial-stamped block anyway.
          2. The partial slot keeps being written by decode, so any stamp written
             at prefill completion would go stale the next decode step.

        Hash collisions and re-stamping of already-stamped blocks are both safe:
          - same content → same hash → no-op write
          - true collision → last-writer-wins; the displaced entry's defensive
            cleanup in `_take_free_block` keeps `_block_hash_to_id` consistent
            on eviction.
        """
        if not block_hashes:
            return
        # Fast path: stamping is atomic per request, so if the last full block
        # is already stamped with our hash, all earlier ones are too. The
        # request owns these blocks (refcount > 0), so eviction can't clobber.
        last_bid = block_ids[len(block_hashes) - 1]
        if self._block_id_to_hash.get(last_bid) == block_hashes[-1]:
            return
        for bid, block_hash in zip(block_ids, block_hashes):
            self._block_hash_to_id[block_hash] = bid
            self._block_id_to_hash[bid] = block_hash

    def clear_cache_index(self) -> None:
        """Drop all prefix-cache hash entries.

        Called on weight update or memory-saver sleep — the KV in cached blocks is
        no longer correct under the new weights / restored memory, so future requests
        must not hit them.

        DOES NOT preempt in-flight requests. Active requests still hold refcounts
        on blocks whose KV is now stale; their decode would read garbage if they
        continued. Caller is responsible for preempting them first — see
        `Scheduler.reset_for_state_change()`.
        """
        self._block_hash_to_id.clear()
        self._block_id_to_hash.clear()


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
