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

"""Unit tests for KVCacheManager: refcounting, LRU eviction, and prefix cache plan/commit/mark."""

import pytest

from vexact.batch_invariant_ops.kv_cache_context import KVCacheManager
from vexact.config import CacheConfig


@pytest.fixture
def mgr():
    def _build(page_size: int = 4, max_blocks: int = 8, enable_prefix_cache: bool = True):
        return KVCacheManager(
            CacheConfig(page_size=page_size, max_cache_blocks=max_blocks),
            enable_prefix_cache=enable_prefix_cache,
        )

    return _build


# ---------- construction & public surface ----------


def test_construction_exposes_public_attrs(mgr):
    m = mgr(page_size=8, max_blocks=16)
    assert m.prefix_cache_enabled is True
    assert m.page_size == 8
    assert m.num_free_blocks() == 16
    assert m.num_allocated_blocks() == 0
    assert m.num_cached_blocks() == 0


def test_construction_disabled(mgr):
    m = mgr(enable_prefix_cache=False)
    assert m.prefix_cache_enabled is False


# ---------- plan_prefix_cache ----------


def test_plan_empty_tokens(mgr):
    m = mgr()
    assert m.plan_prefix_cache([]) == ([], 0)


def test_plan_only_partial_block(mgr):
    # page_size=4, 3 tokens → 0 full blocks
    m = mgr()
    assert m.plan_prefix_cache([1, 2, 3]) == ([], 0)


def test_plan_when_disabled(mgr):
    m = mgr(enable_prefix_cache=False)
    assert m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8]) == ([], 0)


def test_plan_cold_cache_returns_hashes_but_zero_hits(mgr):
    m = mgr()
    hashes, n_cached = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    assert len(hashes) == 2
    assert n_cached == 0


def test_chain_hash_diverges_on_content_difference(mgr):
    m = mgr()
    h_a, _ = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    h_b, _ = m.plan_prefix_cache([1, 2, 3, 4, 9, 9, 9, 9])
    assert h_a[0] == h_b[0]  # same first block content
    assert h_a[1] != h_b[1]  # diverged from block 1 onward


# ---------- commit_prefix_plan ----------


def test_commit_cold_allocates_fresh(mgr):
    m = mgr()
    hashes, n_cached = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    bids = m.commit_prefix_plan(hashes, n_cached, 8)
    assert bids == [0, 1]
    assert m.num_allocated_blocks() == 2


def test_commit_partial_only_takes_one_block(mgr):
    m = mgr()
    hashes, n_cached = m.plan_prefix_cache([1, 2, 3])
    bids = m.commit_prefix_plan(hashes, n_cached, 3)
    assert bids == [0]
    assert m.num_allocated_blocks() == 1


def test_commit_full_plus_partial(mgr):
    # 1 full + 1 partial = 2 blocks total
    m = mgr()
    hashes, n_cached = m.plan_prefix_cache([1, 2, 3, 4, 5])
    assert len(hashes) == 1
    bids = m.commit_prefix_plan(hashes, n_cached, 5)
    assert len(bids) == 2


def test_commit_oom_rollback_keeps_pool_intact(mgr):
    m = mgr(max_blocks=2)
    hashes, n_cached = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert m.commit_prefix_plan(hashes, n_cached, 12) is None
    assert m.num_free_blocks() == 2
    assert m.num_allocated_blocks() == 0


def test_commit_oom_releases_cached_increfs(mgr):
    # Cached blocks incref'd during commit must be decref'd back on OOM, not stuck.
    m = mgr(max_blocks=2)
    # First request fills the cache index (2 full blocks).
    h1, n1 = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    bids1 = m.commit_prefix_plan(h1, n1, 8)
    m.mark_blocks_filled(bids1, h1)
    m.free_blocks(bids1)

    # Second request: same prefix (full hit) + a partial last block. OOM because
    # the partial needs a fresh block but the only 2 blocks just got refcounted.
    h2, n2 = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99])
    assert n2 == 2
    assert m.commit_prefix_plan(h2, n2, 11) is None
    assert m.num_free_blocks() == 2
    assert m.num_allocated_blocks() == 0
    # Cache index untouched by the failed commit
    assert m.num_cached_blocks() == 2


# ---------- mark_blocks_filled ----------


def test_mark_blocks_filled_records_full_blocks_only(mgr):
    m = mgr()
    hashes, n_cached = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7])  # 1 full + partial
    bids = m.commit_prefix_plan(hashes, n_cached, 7)
    m.mark_blocks_filled(bids, hashes)
    assert m.num_cached_blocks() == 1  # only the full block stamped


def test_mark_blocks_filled_idempotent(mgr):
    m = mgr()
    hashes, n_cached = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    bids = m.commit_prefix_plan(hashes, n_cached, 8)

    m.mark_blocks_filled(bids, hashes)
    state1 = (dict(m._block_hash_to_id), dict(m._block_id_to_hash))
    m.mark_blocks_filled(bids, hashes)
    m.mark_blocks_filled(bids, hashes)
    state2 = (dict(m._block_hash_to_id), dict(m._block_id_to_hash))
    assert state1 == state2


def test_mark_blocks_filled_stamps_misses_after_partial_hit(mgr):
    # Half-hit: first block hits cache, second block is fresh. The fresh block
    # must end up correctly stamped using the precomputed chain hash.
    m = mgr()
    h1, n1 = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    bids1 = m.commit_prefix_plan(h1, n1, 8)
    m.mark_blocks_filled(bids1, h1)
    m.free_blocks(bids1)

    h2, n2 = m.plan_prefix_cache([1, 2, 3, 4, 9, 9, 9, 9])
    assert n2 == 1  # only first block hits
    bids2 = m.commit_prefix_plan(h2, n2, 8)
    m.mark_blocks_filled(bids2, h2)

    assert m._block_id_to_hash[bids2[1]] == h2[1]
    assert m._block_hash_to_id[h2[1]] == bids2[1]


def test_mark_blocks_filled_noop_when_empty_hashes(mgr):
    m = mgr(enable_prefix_cache=False)
    bids = m.commit_prefix_plan([], 0, 8)
    m.mark_blocks_filled(bids, [])  # empty hashes path
    assert m.num_cached_blocks() == 0


# ---------- full cycle: miss → hit ----------


def test_full_cycle_miss_then_hit(mgr):
    m = mgr()
    toks = [1, 2, 3, 4, 5, 6, 7, 8]

    h1, n1 = m.plan_prefix_cache(toks)
    assert n1 == 0
    bids1 = m.commit_prefix_plan(h1, n1, len(toks))
    m.mark_blocks_filled(bids1, h1)
    m.free_blocks(bids1)
    assert m.num_free_blocks() == 8
    assert m.num_cached_blocks() == 2  # hashes survive free

    # Same content again — full hit on the same physical blocks.
    h2, n2 = m.plan_prefix_cache(toks)
    assert h2 == h1
    assert n2 == 2
    bids2 = m.commit_prefix_plan(h2, n2, len(toks))
    assert bids2 == bids1


def test_partial_last_block_does_not_block_prefix_hits(mgr):
    # Two requests share the same first full block but differ in the partial tail —
    # the full block should still hit.
    m = mgr()
    h1, n1 = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7])
    bids1 = m.commit_prefix_plan(h1, n1, 7)
    m.mark_blocks_filled(bids1, h1)
    m.free_blocks(bids1)

    h2, n2 = m.plan_prefix_cache([1, 2, 3, 4, 99, 99, 99])
    assert n2 == 1
    assert h2[0] == h1[0]


def test_contiguous_run_stops_at_first_miss(mgr):
    # Fill cache: blocks [1..4][5..8][9..12].
    m = mgr()
    h1, _ = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    bids1 = m.commit_prefix_plan(h1, 0, 12)
    m.mark_blocks_filled(bids1, h1)
    m.free_blocks(bids1)

    # New: [1..4] hits, [99..] misses, [9..12] hashes ALWAYS diverge because the
    # chain depends on the previous block's hash. So even the "same" 3rd block
    # is a different chain hash → reported miss.
    h2, n2 = m.plan_prefix_cache([1, 2, 3, 4, 99, 99, 99, 99, 9, 10, 11, 12])
    assert n2 == 1
    assert h2[0] == h1[0]
    assert h2[2] != h1[2]


def test_refcount_shared_between_concurrent_requests(mgr):
    m = mgr()
    h_a, n_a = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    bids_a = m.commit_prefix_plan(h_a, n_a, 8)
    m.mark_blocks_filled(bids_a, h_a)

    # Second concurrent request hits both blocks.
    h_b, n_b = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    bids_b = m.commit_prefix_plan(h_b, n_b, 8)
    assert bids_b == bids_a
    for bid in bids_a:
        assert m._refcount[bid] == 2

    m.free_blocks(bids_b)
    for bid in bids_a:
        assert m._refcount[bid] == 1
    assert m.num_free_blocks() == 6  # blocks still owned by A

    m.free_blocks(bids_a)
    assert m.num_free_blocks() == 8


# ---------- LRU & eviction ----------


def test_free_lru_oldest_first(mgr):
    m = mgr(max_blocks=4)
    a = m.allocate_blocks(total_tokens=4, num_current_blocks=0)
    b = m.allocate_blocks(total_tokens=4, num_current_blocks=0)
    c = m.allocate_blocks(total_tokens=4, num_current_blocks=0)
    assert a == [0] and b == [1] and c == [2]

    m.free_blocks(a)  # free order: 0
    m.free_blocks(b)  # then 1
    m.free_blocks(c)  # then 2
    # Pool currently: [3 (never used), 0, 1, 2]. Oldest first → 3, then 0, 1, 2.

    assert m.allocate_blocks(total_tokens=4, num_current_blocks=0) == [3]
    assert m.allocate_blocks(total_tokens=4, num_current_blocks=0) == [0]
    assert m.allocate_blocks(total_tokens=4, num_current_blocks=0) == [1]
    assert m.allocate_blocks(total_tokens=4, num_current_blocks=0) == [2]


def test_eviction_drops_hash_entry(mgr):
    m = mgr(max_blocks=2)
    # Fill cache with 2 hashed blocks.
    h1, _ = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    bids1 = m.commit_prefix_plan(h1, 0, 8)
    m.mark_blocks_filled(bids1, h1)
    m.free_blocks(bids1)
    assert m.num_cached_blocks() == 2

    # Allocate 2 fresh blocks for unrelated content → both old hashes get evicted.
    h2, _ = m.plan_prefix_cache([100, 101, 102, 103, 104, 105, 106, 107])
    bids2 = m.commit_prefix_plan(h2, 0, 8)
    assert bids2 is not None
    for old in h1:
        assert old not in m._block_hash_to_id


# ---------- clear_cache_index ----------


def test_clear_cache_index_drops_hashes_only(mgr):
    m = mgr()
    h, n = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    bids = m.commit_prefix_plan(h, n, 8)
    m.mark_blocks_filled(bids, h)
    assert m.num_cached_blocks() == 2
    assert m.num_allocated_blocks() == 2

    m.clear_cache_index()
    assert m.num_cached_blocks() == 0
    assert m.num_allocated_blocks() == 2  # refcounts untouched

    # New plan with same content sees a cold cache.
    _, n2 = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    assert n2 == 0


# ---------- disabled prefix cache ----------


def test_disabled_cache_behaves_like_plain_allocator(mgr):
    m = mgr(enable_prefix_cache=False)
    h, n = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    assert h == [] and n == 0
    bids = m.commit_prefix_plan(h, n, 8)
    assert len(bids) == 2

    m.mark_blocks_filled(bids, h)  # no-op
    assert m.num_cached_blocks() == 0

    # Second identical request still misses.
    m.free_blocks(bids)
    _, n2 = m.plan_prefix_cache([1, 2, 3, 4, 5, 6, 7, 8])
    assert n2 == 0


# ---------- allocate_blocks (decode-path) ----------


def test_allocate_blocks_incremental(mgr):
    m = mgr()
    bids = m.commit_prefix_plan([], 0, 4)  # 1 block via partial-only path
    assert m.allocate_blocks(total_tokens=16, num_current_blocks=1) == [1, 2, 3]
    # Already covers — no further allocation needed.
    assert m.allocate_blocks(total_tokens=16, num_current_blocks=4) == []
    del bids


def test_allocate_blocks_oom_returns_none(mgr):
    m = mgr(max_blocks=2)
    assert m.allocate_blocks(total_tokens=12, num_current_blocks=0) is None
    assert m.num_free_blocks() == 2
