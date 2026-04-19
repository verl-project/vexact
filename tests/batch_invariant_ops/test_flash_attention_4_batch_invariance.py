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

"""Batch invariance tests for Flash Attention 4 (Blackwell, flash_attn.cute)."""

import pytest
import torch


try:
    from flash_attn.cute import flash_attn_varlen_func

    from vexact.utils.device import DEVICE_MAJOR

    FA4_AVAILABLE = True if DEVICE_MAJOR == 10 else False
except ImportError:
    FA4_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FA4_AVAILABLE, reason="flash_attn.cute (FA4) not available or not on Blackwell GPU")

torch.set_default_device("cuda")


def _make_paged_kv_cache(k_seqs, v_seqs, page_size):
    """Pack per-sequence K/V tensors into paged cache + block table.

    Args:
        k_seqs: list of (kv_len, H, D) tensors.
        v_seqs: list of (kv_len, H, D) tensors.
        page_size: page size for the paged cache.

    Returns:
        k_cache: (total_pages, page_size, H, D)
        v_cache: (total_pages, page_size, H, D)
        block_table: (B, max_pages_per_seq) int32
    """
    batch_size = len(k_seqs)
    kv_lens = [k.shape[0] for k in k_seqs]
    pages_per_seq = [(slen + page_size - 1) // page_size for slen in kv_lens]
    max_pages = max(pages_per_seq)
    total_pages = sum(pages_per_seq)

    H, D = k_seqs[0].shape[1], k_seqs[0].shape[2]
    dtype = k_seqs[0].dtype

    k_cache = torch.zeros(total_pages, page_size, H, D, dtype=dtype)
    v_cache = torch.zeros(total_pages, page_size, H, D, dtype=dtype)
    block_table = torch.full((batch_size, max_pages), 0, dtype=torch.int32)

    page_idx = 0
    for i in range(batch_size):
        kv_len = kv_lens[i]
        n_pages = pages_per_seq[i]
        block_table[i, :n_pages] = torch.arange(page_idx, page_idx + n_pages, dtype=torch.int32)
        for p in range(n_pages):
            start = p * page_size
            end = min(start + page_size, kv_len)
            k_cache[page_idx + p, : end - start] = k_seqs[i][start:end]
            v_cache[page_idx + p, : end - start] = v_seqs[i][start:end]
        page_idx += n_pages

    return k_cache, v_cache, block_table


def test_fa4_decode_batch_invariance():
    """Decode-only: single-token queries with paged KV cache.

    Running one sequence alone vs. in a batch must produce identical results.
    """
    H, D = 8, 64
    page_size = 256
    kv_lens = [64, 112, 256, 280]
    batch_size = len(kv_lens)

    torch.manual_seed(42)
    k_seqs = [torch.randn(slen, H, D, dtype=torch.bfloat16) * 0.1 for slen in kv_lens]
    v_seqs = [torch.randn(slen, H, D, dtype=torch.bfloat16) * 0.1 for slen in kv_lens]
    q_seqs = [torch.randn(1, H, D, dtype=torch.bfloat16) * 0.1 for _ in kv_lens]

    k_cache, v_cache, block_table = _make_paged_kv_cache(k_seqs, v_seqs, page_size)

    q_all = torch.cat(q_seqs, dim=0)
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32)

    # Batched forward
    out_all, _ = flash_attn_varlen_func(
        q=q_all,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        max_seqlen_q=1,
        max_seqlen_k=None,
        seqused_k=seqused_k,
        page_table=block_table,
        softmax_scale=None,
        causal=True,
        num_splits=1,
    )

    # Per-sequence forward
    for i in range(batch_size):
        out_one, _ = flash_attn_varlen_func(
            q=q_seqs[i],
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            cu_seqlens_k=None,
            max_seqlen_q=1,
            max_seqlen_k=None,
            seqused_k=seqused_k[i : i + 1],
            page_table=block_table[i : i + 1],
            softmax_scale=None,
            causal=True,
            num_splits=1,
        )
        diff = (out_all[i] - out_one[0]).abs().max().item()
        assert diff < 1e-5, f"Batch invariance violated for seq {i}: max diff {diff}"


def test_fa4_prefill_batch_invariance():
    """Prefill: multi-token queries with paged KV cache.

    Running one prefill sequence alone vs. in a batch must produce identical results.
    """
    H, D = 8, 64
    page_size = 256
    seq_lens = [300, 220]
    batch_size = len(seq_lens)

    torch.manual_seed(42)
    q_seqs = [torch.randn(slen, H, D, dtype=torch.bfloat16) * 0.1 for slen in seq_lens]
    k_seqs = [torch.randn(slen, H, D, dtype=torch.bfloat16) * 0.1 for slen in seq_lens]
    v_seqs = [torch.randn(slen, H, D, dtype=torch.bfloat16) * 0.1 for slen in seq_lens]

    k_cache, v_cache, block_table = _make_paged_kv_cache(k_seqs, v_seqs, page_size)

    q_all = torch.cat(q_seqs, dim=0)
    cu_seqlens_q = torch.tensor(
        [0] + [sum(seq_lens[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
    )
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32)

    # Batched forward
    out_all, _ = flash_attn_varlen_func(
        q=q_all,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        max_seqlen_q=max(seq_lens),
        max_seqlen_k=None,
        seqused_k=seqused_k,
        page_table=block_table,
        softmax_scale=None,
        causal=True,
        num_splits=1,
    )

    # Per-sequence forward
    offset = 0
    for i in range(batch_size):
        out_one, _ = flash_attn_varlen_func(
            q=q_seqs[i],
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=torch.tensor([0, seq_lens[i]], dtype=torch.int32),
            cu_seqlens_k=None,
            max_seqlen_q=seq_lens[i],
            max_seqlen_k=None,
            seqused_k=seqused_k[i : i + 1],
            page_table=block_table[i : i + 1],
            softmax_scale=None,
            causal=True,
            num_splits=1,
        )
        diff = (out_all[offset : offset + seq_lens[i]] - out_one).abs().max().item()
        assert diff < 1e-5, f"Batch invariance violated for seq {i}: max diff {diff}"
        offset += seq_lens[i]


def test_fa4_mixed_prefill_decode_batch_invariance():
    """Mixed prefill + decode batch with paged KV cache.

    Individual sequence results must match when run in a combined batch.
    """
    H, D = 8, 64
    page_size = 256

    prefill_seq_lens = [300, 220]
    decode_seq_lens = [1, 1]
    decode_kv_lens = [64, 112]

    all_q_lens = prefill_seq_lens + decode_seq_lens
    all_kv_lens = prefill_seq_lens + decode_kv_lens  # prefill: q_len == kv_len
    batch_size = len(all_q_lens)

    torch.manual_seed(42)
    q_seqs = [torch.randn(slen, H, D, dtype=torch.bfloat16) * 0.1 for slen in all_q_lens]
    k_seqs = [torch.randn(slen, H, D, dtype=torch.bfloat16) * 0.1 for slen in all_kv_lens]
    v_seqs = [torch.randn(slen, H, D, dtype=torch.bfloat16) * 0.1 for slen in all_kv_lens]

    k_cache, v_cache, block_table = _make_paged_kv_cache(k_seqs, v_seqs, page_size)

    q_all = torch.cat(q_seqs, dim=0)
    cu_seqlens_q = torch.tensor(
        [0] + [sum(all_q_lens[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
    )
    seqused_k = torch.tensor(all_kv_lens, dtype=torch.int32)

    # Batched forward
    out_all, _ = flash_attn_varlen_func(
        q=q_all,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        max_seqlen_q=max(all_q_lens),
        max_seqlen_k=None,
        seqused_k=seqused_k,
        page_table=block_table,
        softmax_scale=None,
        causal=True,
        num_splits=1,
    )

    # Per-sequence forward
    offset = 0
    for i in range(batch_size):
        out_one, _ = flash_attn_varlen_func(
            q=q_seqs[i],
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=torch.tensor([0, all_q_lens[i]], dtype=torch.int32),
            cu_seqlens_k=None,
            max_seqlen_q=all_q_lens[i],
            max_seqlen_k=None,
            seqused_k=seqused_k[i : i + 1],
            page_table=block_table[i : i + 1],
            softmax_scale=None,
            causal=True,
            num_splits=1,
        )
        diff = (out_all[offset : offset + all_q_lens[i]] - out_one).abs().max().item()
        assert diff < 1e-5, f"Batch invariance violated for seq {i}: max diff {diff}"
        offset += all_q_lens[i]
