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
Minimal test for FA4 (192, 128) DeepSeek shape with paged KV cache.
Verifies that flash_attn_varlen_func produces correct output when
qk_head_dim=192 and v_head_dim=128 using paged attention.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest vexact/tests/test_fa4_deepseek_shape.py -v -s
"""

import pytest
import torch


try:
    from flash_attn.cute import flash_attn_varlen_func  # noqa: F401

    FA4_AVAILABLE = True
except ImportError:
    FA4_AVAILABLE = False

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FA4 tests")

_CUDA_MAJOR = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
requires_fa4_paged_kv = pytest.mark.skipif(
    not FA4_AVAILABLE or _CUDA_MAJOR < 9,
    reason="flash_attn.cute paged KV requires SM90+",
)


def reference_attention(q, k, v, scale):
    """Simple reference attention (no paging, no varlen)."""
    # q: (B, nH, S, D), k: (B, nKVH, S, D), v: (B, nKVH, S, D_v)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    # Causal mask
    S = q.shape[2]
    mask = torch.triu(torch.ones(S, S, device=q.device), diagonal=1).bool()
    scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.float())
    return out.to(q.dtype)


@requires_fa4_paged_kv
def test_fa4_qwen_paged():
    """Test FA4 with DeepSeek (192, 128) shape using paged KV cache."""
    from flash_attn.cute import flash_attn_varlen_func

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # DeepSeek V3 / Moonlight config
    num_heads = 16
    num_kv_heads = 16
    qk_head_dim = 128
    v_head_dim = 128
    seq_len = 32
    page_size = 16
    scale = 1.0 / (qk_head_dim**0.5)

    # Create Q, K, V
    q = torch.randn(seq_len, num_heads, qk_head_dim, device=device, dtype=dtype)
    k = torch.randn(seq_len, num_kv_heads, qk_head_dim, device=device, dtype=dtype)
    v = torch.randn(seq_len, num_kv_heads, v_head_dim, device=device, dtype=dtype)

    # --- Test 1: non-paged varlen (baseline) ---
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    out_varlen, _ = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
    )
    print(f"Non-paged output shape: {out_varlen.shape}")
    assert out_varlen.shape == (seq_len, num_heads, v_head_dim)

    # --- Test 2: paged varlen ---
    num_pages = (seq_len + page_size - 1) // page_size  # 2 pages

    # --- Test 2a: paged with separate K/V dims (known FA4 bug) ---
    k_cache = torch.zeros(num_pages, page_size, num_kv_heads, qk_head_dim, device=device, dtype=dtype)
    v_cache_raw = torch.zeros(num_pages, page_size, num_kv_heads, v_head_dim, device=device, dtype=dtype)
    for i in range(seq_len):
        page_idx = i // page_size
        offset = i % page_size
        k_cache[page_idx, offset] = k[i]
        v_cache_raw[page_idx, offset] = v[i]

    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    context_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    out_paged_raw, _ = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache_raw,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=None,
        max_seqlen_q=seq_len,
        max_seqlen_k=None,
        seqused_k=context_lens,
        page_table=page_table,
        softmax_scale=scale,
        causal=True,
    )
    print(f"Paged (raw 128) output shape: {out_paged_raw.shape}")

    # --- Test 2b: paged with V padded to 192, passed as [..., :128] view ---
    # Workaround: allocate V cache with qk_head_dim, store padded V, pass view
    v_cache_padded = torch.zeros(num_pages, page_size, num_kv_heads, qk_head_dim, device=device, dtype=dtype)
    for i in range(seq_len):
        page_idx = i // page_size
        offset = i % page_size
        v_cache_padded[page_idx, offset, :, :v_head_dim] = v[i]
    # Pass a view that exposes only the first 128 dims — non-contiguous but correct strides
    v_cache_view = v_cache_padded[..., :v_head_dim]
    print(
        "V cache view: \n",
        f"shape={v_cache_view.shape}, stride={v_cache_view.stride()}, contiguous={v_cache_view.is_contiguous()}",
    )

    out_paged, _ = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache_view,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=None,
        max_seqlen_q=seq_len,
        max_seqlen_k=None,
        seqused_k=context_lens,
        page_table=page_table,
        softmax_scale=scale,
        causal=True,
    )
    print(f"Paged (padded view) output shape: {out_paged.shape}")
    assert out_paged.shape == (seq_len, num_heads, v_head_dim)

    # --- Test 3: reference comparison ---
    q_4d = q.unsqueeze(0).transpose(1, 2)  # (1, nH, S, D)
    k_4d = k.unsqueeze(0).transpose(1, 2)  # (1, nKVH, S, D)
    v_4d = v.unsqueeze(0).transpose(1, 2)  # (1, nKVH, S, D_v)
    out_ref = reference_attention(q_4d, k_4d, v_4d, scale)
    out_ref = out_ref.squeeze(0).transpose(0, 1)  # (S, nH, D_v)

    # Compare
    max_diff_varlen = (out_varlen.float() - out_ref.float()).abs().max().item()
    max_diff_paged_raw = (out_paged_raw.float() - out_ref.float()).abs().max().item()
    max_diff_paged = (out_paged.float() - out_ref.float()).abs().max().item()
    max_diff_paged_vs_varlen = (out_paged.float() - out_varlen.float()).abs().max().item()

    print(f"Max diff (non-paged vs ref): {max_diff_varlen:.6f}")
    print(f"Max diff (paged raw 128 vs ref): {max_diff_paged_raw:.6f}  [FA4 paged bug]")
    print(f"Max diff (paged padded view vs ref): {max_diff_paged:.6f}")
    print(f"Max diff (paged padded view vs non-paged): {max_diff_paged_vs_varlen:.6f}")

    # bf16 tolerance
    assert max_diff_varlen < 0.05, f"Non-paged output too far from reference: {max_diff_varlen}"
    assert max_diff_paged < 0.05, f"Paged (padded view) output too far from reference: {max_diff_paged}"
    assert max_diff_paged_vs_varlen < 0.01, f"Paged padded view vs non-paged mismatch: {max_diff_paged_vs_varlen}"

    print("PASSED: FA4 (192, 128) paged attention with padded view workaround works correctly")


@requires_fa4_paged_kv
def test_fa4_deepseek_paged():
    """Test FA4 with DeepSeek (192, 128) shape using paged KV cache."""
    from flash_attn.cute import flash_attn_varlen_func

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # DeepSeek V3 / Moonlight config
    num_heads = 16
    num_kv_heads = 16
    qk_head_dim = 192
    v_head_dim = 128
    seq_len = 32
    page_size = 16
    scale = 1.0 / (qk_head_dim**0.5)

    # Create Q, K, V
    q = torch.randn(seq_len, num_heads, qk_head_dim, device=device, dtype=dtype)
    k = torch.randn(seq_len, num_kv_heads, qk_head_dim, device=device, dtype=dtype)
    v = torch.randn(seq_len, num_kv_heads, v_head_dim, device=device, dtype=dtype)

    # --- Test 1: non-paged varlen (baseline) ---
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    out_varlen, _ = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
    )
    print(f"Non-paged output shape: {out_varlen.shape}")
    assert out_varlen.shape == (seq_len, num_heads, v_head_dim)

    # --- Test 2: paged varlen ---
    num_pages = (seq_len + page_size - 1) // page_size  # 2 pages

    # --- Test 2a: paged with separate K/V dims (known FA4 bug) ---
    k_cache = torch.zeros(num_pages, page_size, num_kv_heads, qk_head_dim, device=device, dtype=dtype)
    v_cache_raw = torch.zeros(num_pages, page_size, num_kv_heads, v_head_dim, device=device, dtype=dtype)
    for i in range(seq_len):
        page_idx = i // page_size
        offset = i % page_size
        k_cache[page_idx, offset] = k[i]
        v_cache_raw[page_idx, offset] = v[i]

    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    context_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    out_paged_raw, _ = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache_raw,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=None,
        max_seqlen_q=seq_len,
        max_seqlen_k=None,
        seqused_k=context_lens,
        page_table=page_table,
        softmax_scale=scale,
        causal=True,
    )
    print(f"Paged (raw 128) output shape: {out_paged_raw.shape}")

    # --- Test 2b: paged with V padded to 192, passed as [..., :128] view ---
    # Workaround: allocate V cache with qk_head_dim, store padded V, pass view
    v_cache_padded = torch.zeros(num_pages, page_size, num_kv_heads, qk_head_dim, device=device, dtype=dtype)
    for i in range(seq_len):
        page_idx = i // page_size
        offset = i % page_size
        v_cache_padded[page_idx, offset, :, :v_head_dim] = v[i]
    # Pass a view that exposes only the first 128 dims — non-contiguous but correct strides
    v_cache_view = v_cache_padded[..., :v_head_dim]
    print(
        "V cache view: \n",
        f"shape={v_cache_view.shape}, stride={v_cache_view.stride()}, contiguous={v_cache_view.is_contiguous()}",
    )

    out_paged, _ = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache_view,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=None,
        max_seqlen_q=seq_len,
        max_seqlen_k=None,
        seqused_k=context_lens,
        page_table=page_table,
        softmax_scale=scale,
        causal=True,
    )
    print(f"Paged (padded view) output shape: {out_paged.shape}")
    assert out_paged.shape == (seq_len, num_heads, v_head_dim)

    # --- Test 3: reference comparison ---
    q_4d = q.unsqueeze(0).transpose(1, 2)  # (1, nH, S, D)
    k_4d = k.unsqueeze(0).transpose(1, 2)  # (1, nKVH, S, D)
    v_4d = v.unsqueeze(0).transpose(1, 2)  # (1, nKVH, S, D_v)
    out_ref = reference_attention(q_4d, k_4d, v_4d, scale)
    out_ref = out_ref.squeeze(0).transpose(0, 1)  # (S, nH, D_v)

    # Compare
    max_diff_varlen = (out_varlen.float() - out_ref.float()).abs().max().item()
    max_diff_paged_raw = (out_paged_raw.float() - out_ref.float()).abs().max().item()
    max_diff_paged = (out_paged.float() - out_ref.float()).abs().max().item()
    max_diff_paged_vs_varlen = (out_paged.float() - out_varlen.float()).abs().max().item()

    print(f"Max diff (non-paged vs ref): {max_diff_varlen:.6f}")
    print(f"Max diff (paged raw 128 vs ref): {max_diff_paged_raw:.6f}  [FA4 paged bug]")
    print(f"Max diff (paged padded view vs ref): {max_diff_paged:.6f}")
    print(f"Max diff (paged padded view vs non-paged): {max_diff_paged_vs_varlen:.6f}")

    # bf16 tolerance
    assert max_diff_varlen < 0.05, f"Non-paged output too far from reference: {max_diff_varlen}"
    assert max_diff_paged < 0.05, f"Paged (padded view) output too far from reference: {max_diff_paged}"
    assert max_diff_paged_vs_varlen < 0.01, f"Paged padded view vs non-paged mismatch: {max_diff_paged_vs_varlen}"

    print("PASSED: FA4 (192, 128) paged attention with padded view workaround works correctly")


def test_store_kvcache_separate_dims():
    """Test that store_kvcache correctly handles different K/V head dims."""
    from vexact.batch_invariant_ops.kv_cache_context import store_kvcache

    device = "cuda"
    dtype = torch.bfloat16
    N, num_heads, k_head_dim, v_head_dim = 8, 16, 192, 128
    page_size = 256
    num_blocks = 2

    key = torch.randn(N, num_heads, k_head_dim, device=device, dtype=dtype)
    value = torch.randn(N, num_heads, v_head_dim, device=device, dtype=dtype)
    k_cache = torch.zeros(num_blocks, page_size, num_heads, k_head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros(num_blocks, page_size, num_heads, v_head_dim, device=device, dtype=dtype)
    slot_mapping = torch.arange(N, dtype=torch.int32, device=device)

    store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    # Verify each token was stored correctly
    for i in range(N):
        slot = slot_mapping[i].item()
        block_idx = slot // page_size
        block_offset = slot % page_size
        assert torch.allclose(k_cache[block_idx, block_offset], key[i]), f"Key mismatch at token {i}"
        assert torch.allclose(v_cache[block_idx, block_offset], value[i]), f"Value mismatch at token {i}"

    print("PASSED: store_kvcache with separate K/V dims works correctly")


if __name__ == "__main__":
    test_store_kvcache_separate_dims()
    test_fa4_qwen_paged()
    test_fa4_deepseek_paged()
