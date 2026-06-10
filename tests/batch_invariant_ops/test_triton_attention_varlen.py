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

import pytest
import torch
import torch.nn as nn

from vexact.batch_invariant_ops.kv_cache_context import set_kv_cache_context
from vexact.batch_invariant_ops.triton_invariant_attention import flash_attention_forward, flash_attn_varlen_func


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton attention tests")


def _make_paged_kv_cache(k_seqs, v_seqs, page_size):
    batch_size = len(k_seqs)
    kv_lens = [k.shape[0] for k in k_seqs]
    pages_per_seq = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    max_pages = max(pages_per_seq)
    total_pages = sum(pages_per_seq)
    kv_heads, k_dim = k_seqs[0].shape[1:]
    v_dim = v_seqs[0].shape[-1]

    k_cache = torch.zeros(total_pages, page_size, kv_heads, k_dim, device=k_seqs[0].device, dtype=k_seqs[0].dtype)
    v_cache = torch.zeros(total_pages, page_size, kv_heads, v_dim, device=v_seqs[0].device, dtype=v_seqs[0].dtype)
    page_table = torch.full((batch_size, max_pages), -1, device=k_seqs[0].device, dtype=torch.int32)

    page_idx = 0
    for seq_idx, kv_len in enumerate(kv_lens):
        for page in range(pages_per_seq[seq_idx]):
            start = page * page_size
            end = min(start + page_size, kv_len)
            page_table[seq_idx, page] = page_idx
            k_cache[page_idx, : end - start] = k_seqs[seq_idx][start:end]
            v_cache[page_idx, : end - start] = v_seqs[seq_idx][start:end]
            page_idx += 1

    return k_cache, v_cache, page_table


def _eager_attention(q_seq, k_seq, v_seq, softmax_scale, causal=True):
    q_t = q_seq.transpose(0, 1)
    k_t = k_seq.transpose(0, 1)
    v_t = v_seq.transpose(0, 1)
    groups = q_t.shape[0] // k_t.shape[0]
    if groups != 1:
        k_t = k_t.repeat_interleave(groups, dim=0)
        v_t = v_t.repeat_interleave(groups, dim=0)

    scores = torch.matmul(q_t.float(), k_t.float().transpose(-1, -2)) * softmax_scale
    if causal:
        q_len = q_seq.shape[0]
        kv_len = k_seq.shape[0]
        q_pos = torch.arange(q_len, device=q_seq.device) + (kv_len - q_len)
        kv_pos = torch.arange(kv_len, device=q_seq.device)
        causal_mask = kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v_seq.dtype)
    if causal:
        probs = torch.where(causal_mask.any(dim=-1)[None, :, None], probs, 0.0)
    return torch.matmul(probs, v_t).transpose(0, 1)


def _assert_nonpaged_backward_matches_eager(
    q_lens,
    kv_lens,
    *,
    q_heads,
    kv_heads,
    head_dim,
    dtype=torch.float32,
    atol=2e-3,
    rtol=2e-3,
):
    q = torch.randn(sum(q_lens), q_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(sum(kv_lens), kv_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(sum(kv_lens), kv_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    cu_q = torch.tensor([0] + [sum(q_lens[: i + 1]) for i in range(len(q_lens))], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0] + [sum(kv_lens[: i + 1]) for i in range(len(kv_lens))], device="cuda", dtype=torch.int32)

    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        softmax_scale=head_dim**-0.5,
        causal=True,
    )

    q_offset = 0
    kv_offset = 0
    ref_chunks = []
    for q_len, kv_len in zip(q_lens, kv_lens):
        ref_chunks.append(
            _eager_attention(
                q_ref[q_offset : q_offset + q_len],
                k_ref[kv_offset : kv_offset + kv_len],
                v_ref[kv_offset : kv_offset + kv_len],
                head_dim**-0.5,
            )
        )
        q_offset += q_len
        kv_offset += kv_len
    ref = torch.cat(ref_chunks, dim=0)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    ref.backward(grad_out)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


def _run_paged(q_seqs, k_seqs, v_seqs, page_size, *, page_kw="page_table"):
    k_cache, v_cache, page_table = _make_paged_kv_cache(k_seqs, v_seqs, page_size)
    q_lens = [q.shape[0] for q in q_seqs]
    kv_lens = [k.shape[0] for k in k_seqs]
    cu_q = torch.tensor([0] + [sum(q_lens[: i + 1]) for i in range(len(q_lens))], device="cuda", dtype=torch.int32)
    kwargs = {
        page_kw: page_table,
        "seqused_k": torch.tensor(kv_lens, device="cuda", dtype=torch.int32),
    }
    return flash_attn_varlen_func(
        torch.cat(q_seqs, dim=0),
        k_cache,
        v_cache,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=None,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=None,
        softmax_scale=q_seqs[0].shape[-1] ** -0.5,
        causal=True,
        **kwargs,
    )[0]


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, 3e-2, 3e-2),
    ],
)
def test_triton_flash_attn_varlen_nonpaged_packed_backward(dtype, atol, rtol):
    torch.manual_seed(0)
    q_lens = [65, 129]
    kv_lens = [65, 129]
    q_heads = 4
    kv_heads = 2
    head_dim = 64
    q = torch.randn(sum(q_lens), q_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(sum(kv_lens), kv_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(sum(kv_lens), kv_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    cu_q = torch.tensor([0, q_lens[0], sum(q_lens)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, kv_lens[0], sum(kv_lens)], device="cuda", dtype=torch.int32)

    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        softmax_scale=head_dim**-0.5,
        causal=True,
    )
    ref = torch.cat(
        [
            _eager_attention(q[: q_lens[0]], k[: kv_lens[0]], v[: kv_lens[0]], head_dim**-0.5),
            _eager_attention(q[q_lens[0] :], k[kv_lens[0] :], v[kv_lens[0] :], head_dim**-0.5),
        ],
        dim=0,
    )
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    out.square().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_triton_flash_attn_varlen_nonpaged_requires_max_seqlen():
    q = torch.randn(1, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 64, device="cuda", dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="max_seqlen_k is required"):
        flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen_q=1)

    with pytest.raises(ValueError, match="max_seqlen_q is required"):
        flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen_k=1)


def test_triton_flash_attn_varlen_nonpaged_decode_packed_backward():
    torch.manual_seed(10)
    q_lens = [1, 17]
    kv_lens = [65, 129]
    q_heads = 4
    kv_heads = 2
    head_dim = 64
    dtype = torch.bfloat16
    q = torch.randn(sum(q_lens), q_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(sum(kv_lens), kv_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(sum(kv_lens), kv_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    cu_q = torch.tensor([0, q_lens[0], sum(q_lens)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, kv_lens[0], sum(kv_lens)], device="cuda", dtype=torch.int32)

    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        softmax_scale=head_dim**-0.5,
        causal=True,
    )
    ref = torch.cat(
        [
            _eager_attention(q[: q_lens[0]], k[: kv_lens[0]], v[: kv_lens[0]], head_dim**-0.5),
            _eager_attention(q[q_lens[0] :], k[kv_lens[0] :], v[kv_lens[0] :], head_dim**-0.5),
        ],
        dim=0,
    )
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)

    out.square().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


@pytest.mark.parametrize(
    ("q_lens", "kv_lens"),
    [
        ([17, 33, 65], [17, 33, 65]),
        ([1, 7, 19], [65, 71, 83]),
    ],
)
def test_triton_flash_attn_varlen_nonpaged_gqa_backward_matches_eager(q_lens, kv_lens):
    torch.manual_seed(101)
    _assert_nonpaged_backward_matches_eager(
        q_lens,
        kv_lens,
        q_heads=4,
        kv_heads=2,
        head_dim=64,
        dtype=torch.float32,
        atol=5e-3,
        rtol=5e-3,
    )


def test_triton_flash_attn_varlen_nonpaged_q_longer_than_k_matches_eager():
    torch.manual_seed(303)
    _assert_nonpaged_backward_matches_eager(
        [5, 9],
        [2, 4],
        q_heads=4,
        kv_heads=2,
        head_dim=32,
        dtype=torch.float32,
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.parametrize("head_dim", [40, 96])
def test_triton_flash_attn_varlen_nonpaged_arbitrary_head_dim_backward_matches_eager(head_dim):
    torch.manual_seed(202)
    _assert_nonpaged_backward_matches_eager(
        [9, 23],
        [41, 57],
        q_heads=6,
        kv_heads=3,
        head_dim=head_dim,
        dtype=torch.float32,
        atol=7e-3,
        rtol=7e-3,
    )


def test_triton_flash_attn_varlen_qwen3_like_nonpaged_backward():
    torch.manual_seed(11)
    q_lens = [257, 385]
    kv_lens = [257, 385]
    q_heads = 16
    kv_heads = 8
    head_dim = 128
    dtype = torch.bfloat16
    q = torch.randn(sum(q_lens), q_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(sum(kv_lens), kv_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(sum(kv_lens), kv_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    cu_q = torch.tensor([0, q_lens[0], sum(q_lens)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, kv_lens[0], sum(kv_lens)], device="cuda", dtype=torch.int32)

    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        softmax_scale=head_dim**-0.5,
        causal=True,
    )
    out.square().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


@pytest.mark.parametrize(
    ("q_lens", "kv_lens"),
    [
        ([1, 1, 1, 1], [17, 31, 40, 49]),
        ([24, 19], [24, 19]),
        ([24, 1, 17, 1], [24, 33, 17, 41]),
    ],
)
def test_triton_flash_attn_varlen_paged_correctness_and_batch_invariance(q_lens, kv_lens):
    torch.manual_seed(1)
    q_heads = 4
    kv_heads = 2
    head_dim = 32
    page_size = 16
    dtype = torch.bfloat16

    q_seqs = [torch.randn(q_len, q_heads, head_dim, device="cuda", dtype=dtype) * 0.1 for q_len in q_lens]
    k_seqs = [torch.randn(kv_len, kv_heads, head_dim, device="cuda", dtype=dtype) * 0.1 for kv_len in kv_lens]
    v_seqs = [torch.randn(kv_len, kv_heads, head_dim, device="cuda", dtype=dtype) * 0.1 for kv_len in kv_lens]

    out = _run_paged(q_seqs, k_seqs, v_seqs, page_size)
    ref = torch.cat(
        [_eager_attention(q, k, v, head_dim**-0.5).to(dtype) for q, k, v in zip(q_seqs, k_seqs, v_seqs)],
        dim=0,
    )
    torch.testing.assert_close(out, ref, atol=3e-3, rtol=3e-3)

    offset = 0
    for q_seq, k_seq, v_seq in zip(q_seqs, k_seqs, v_seqs):
        one = _run_paged([q_seq], [k_seq], [v_seq], page_size)
        torch.testing.assert_close(out[offset : offset + q_seq.shape[0]], one, atol=0, rtol=0)
        offset += q_seq.shape[0]


def test_triton_flash_attn_varlen_page_table_aliases_match():
    torch.manual_seed(2)
    q_seqs = [torch.randn(1, 4, 32, device="cuda", dtype=torch.bfloat16)]
    k_seqs = [torch.randn(33, 2, 32, device="cuda", dtype=torch.bfloat16)]
    v_seqs = [torch.randn(33, 2, 32, device="cuda", dtype=torch.bfloat16)]

    page_out = _run_paged(q_seqs, k_seqs, v_seqs, 16, page_kw="page_table")
    block_out = _run_paged(q_seqs, k_seqs, v_seqs, 16, page_kw="block_table")
    torch.testing.assert_close(page_out, block_out, atol=0, rtol=0)


def test_triton_flash_attn_varlen_paged_prefill_decode_bitwise_alignment():
    torch.manual_seed(3)
    q_heads = 4
    kv_heads = 2
    head_dim = 32
    page_size = 16
    dtype = torch.bfloat16
    kv_lens = [17, 33, 64]

    q_prefill_seqs = [torch.randn(kv_len, q_heads, head_dim, device="cuda", dtype=dtype) for kv_len in kv_lens]
    k_seqs = [torch.randn(kv_len, kv_heads, head_dim, device="cuda", dtype=dtype) for kv_len in kv_lens]
    v_seqs = [torch.randn(kv_len, kv_heads, head_dim, device="cuda", dtype=dtype) for kv_len in kv_lens]

    prefill_out = _run_paged(q_prefill_seqs, k_seqs, v_seqs, page_size)
    decode_out = _run_paged([q_seq[-1:] for q_seq in q_prefill_seqs], k_seqs, v_seqs, page_size)

    offset = 0
    for seq_idx, q_seq in enumerate(q_prefill_seqs):
        torch.testing.assert_close(prefill_out[offset + q_seq.shape[0] - 1], decode_out[seq_idx], atol=0, rtol=0)
        offset += q_seq.shape[0]


def test_triton_flash_attn_varlen_qwen3_like_paged_prefill_decode_bitwise_alignment():
    torch.manual_seed(33)
    q_heads = 16
    kv_heads = 8
    head_dim = 128
    page_size = 256
    dtype = torch.bfloat16
    kv_lens = [74, 75, 76, 77, 78, 79, 80, 81, 82]

    q_prefill_seqs = [torch.randn(kv_len, q_heads, head_dim, device="cuda", dtype=dtype) for kv_len in kv_lens]
    k_seqs = [torch.randn(kv_len, kv_heads, head_dim, device="cuda", dtype=dtype) for kv_len in kv_lens]
    v_seqs = [torch.randn(kv_len, kv_heads, head_dim, device="cuda", dtype=dtype) for kv_len in kv_lens]

    prefill_out = _run_paged(q_prefill_seqs, k_seqs, v_seqs, page_size)
    decode_out = _run_paged([q_seq[-1:] for q_seq in q_prefill_seqs], k_seqs, v_seqs, page_size)
    nonpaged_out, _ = flash_attn_varlen_func(
        torch.cat(q_prefill_seqs, dim=0),
        torch.cat(k_seqs, dim=0),
        torch.cat(v_seqs, dim=0),
        cu_seqlens_q=torch.tensor(
            [0] + [sum(kv_lens[: i + 1]) for i in range(len(kv_lens))], device="cuda", dtype=torch.int32
        ),
        cu_seqlens_k=torch.tensor(
            [0] + [sum(kv_lens[: i + 1]) for i in range(len(kv_lens))], device="cuda", dtype=torch.int32
        ),
        max_seqlen_q=max(kv_lens),
        max_seqlen_k=max(kv_lens),
        softmax_scale=head_dim**-0.5,
        causal=True,
    )

    offset = 0
    for seq_idx, q_seq in enumerate(q_prefill_seqs):
        last_idx = offset + q_seq.shape[0] - 1
        torch.testing.assert_close(prefill_out[last_idx], decode_out[seq_idx], atol=0, rtol=0)
        torch.testing.assert_close(prefill_out[last_idx], nonpaged_out[last_idx], atol=0, rtol=0)
        offset += q_seq.shape[0]


def test_triton_flash_attn_varlen_mla_paged_matches_eager():
    torch.manual_seed(44)
    q_heads = 16
    kv_heads = 16
    qk_head_dim = 192
    v_head_dim = 128
    page_size = 64
    dtype = torch.bfloat16
    q_lens = [1, 17, 65]
    kv_lens = [33, 65, 129]

    q_seqs = [torch.randn(q_len, q_heads, qk_head_dim, device="cuda", dtype=dtype) * 0.1 for q_len in q_lens]
    k_seqs = [torch.randn(kv_len, kv_heads, qk_head_dim, device="cuda", dtype=dtype) * 0.1 for kv_len in kv_lens]
    v_seqs = [torch.randn(kv_len, kv_heads, v_head_dim, device="cuda", dtype=dtype) * 0.1 for kv_len in kv_lens]

    out = _run_paged(q_seqs, k_seqs, v_seqs, page_size)
    ref = torch.cat(
        [_eager_attention(q, k, v, qk_head_dim**-0.5).to(dtype) for q, k, v in zip(q_seqs, k_seqs, v_seqs)],
        dim=0,
    )

    assert out.shape == (sum(q_lens), q_heads, v_head_dim)
    torch.testing.assert_close(out, ref, atol=3e-3, rtol=3e-3)


def test_triton_flash_attn_varlen_paged_backward_raises():
    torch.manual_seed(4)
    q_seqs = [torch.randn(1, 4, 32, device="cuda", dtype=torch.float32, requires_grad=True)]
    k_seqs = [torch.randn(17, 2, 32, device="cuda", dtype=torch.float32)]
    v_seqs = [torch.randn(17, 2, 32, device="cuda", dtype=torch.float32)]
    out = _run_paged(q_seqs, k_seqs, v_seqs, 16)

    with pytest.raises(NotImplementedError, match="paged triton flash_attn_varlen_func backward"):
        out.sum().backward()


def test_triton_flash_attention_forward_wrapper_shape():
    torch.manual_seed(5)
    q_heads = 4
    kv_heads = 2
    head_dim = 32
    page_size = 16
    q = torch.randn(2, q_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    k_seqs = [torch.randn(17, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    v_seqs = [torch.randn(17, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    k_cache, v_cache, page_table = _make_paged_kv_cache(k_seqs, v_seqs, page_size)

    class MockModule(nn.Module):
        layer_idx = 0

    set_kv_cache_context(
        is_paged_attn=True,
        key_cache={0: k_cache},
        value_cache={0: v_cache},
        block_tables=page_table,
        context_lens=torch.tensor([17, 17], device="cuda", dtype=torch.int32),
        slot_mapping=torch.empty(0, device="cuda", dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
        max_seqlen_q=1,
    )

    out, _ = flash_attention_forward(MockModule(), q, None, None, attention_mask=None, scaling=head_dim**-0.5)
    assert out.shape == (2, q_heads, head_dim)
