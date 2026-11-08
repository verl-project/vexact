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

from vexact.batch_invariant_ops.flash_attention import (
    _get_fa_learnable_sink,
    _get_fa_window_size,
    flash_attention_forward,
)
from vexact.batch_invariant_ops.kv_cache_context import set_kv_cache_context


@pytest.mark.parametrize(
    ("sliding_window", "expected"),
    [
        (None, (-1, -1)),
        (1, (0, 0)),
        (4096, (4095, 4095)),
    ],
)
def test_get_fa_window_size(sliding_window, expected):
    assert _get_fa_window_size(sliding_window) == expected


@pytest.mark.parametrize("sliding_window", [0, -1])
def test_get_fa_window_size_rejects_non_positive_values(sliding_window):
    with pytest.raises(ValueError, match="sliding_window must be positive"):
        _get_fa_window_size(sliding_window)


def test_get_fa_learnable_sink_casts_to_query_dtype():
    query = torch.empty(1, dtype=torch.bfloat16)
    sink = torch.randn(2, dtype=torch.float32)

    actual = _get_fa_learnable_sink(sink, query, use_cute=True)

    assert actual.dtype == query.dtype
    torch.testing.assert_close(actual.float(), sink.to(torch.bfloat16).float())


def test_get_fa_learnable_sink_rejects_fa3():
    with pytest.raises(NotImplementedError, match="require the FA4"):
        _get_fa_learnable_sink(torch.zeros(2), torch.empty(1), use_cute=False)


@pytest.mark.parametrize("use_cute", [False, True], ids=["fa3", "fa4"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_attention_sliding_window_matches_reference(use_cute):
    device_major = torch.cuda.get_device_capability()[0]
    if use_cute:
        if device_major < 9:
            pytest.skip("FA4 requires SM90+")
        pytest.importorskip("flash_attn.cute")
    else:
        if device_major != 9:
            pytest.skip("FA3 requires SM90")
        pytest.importorskip("flash_attn_interface")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, seq_len, num_heads, head_dim = 1, 16, 2, 64
    page_size = 256
    sliding_window = 4
    scaling = head_dim**-0.5

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    key_cache = torch.zeros(1, page_size, num_heads, head_dim, device=device, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    set_kv_cache_context(
        is_paged_attn=True,
        key_cache={0: key_cache},
        value_cache={0: value_cache},
        block_tables=torch.tensor([[0]], device=device, dtype=torch.int32),
        context_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
        slot_mapping=torch.arange(seq_len, device=device, dtype=torch.int64),
        query_start_loc=torch.tensor([0, seq_len], device=device, dtype=torch.int32),
        max_seqlen_q=seq_len,
    )

    module = torch.nn.Module()
    module.layer_idx = 0
    actual, _ = flash_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=None,
        scaling=scaling,
        sliding_window=sliding_window,
        use_cute=use_cute,
    )

    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    positions = torch.arange(seq_len, device=device)
    query_positions = positions[:, None]
    key_positions = positions[None, :]
    mask = (key_positions <= query_positions) & (key_positions > query_positions - sliding_window)
    scores.masked_fill_(~mask, float("-inf"))
    expected = torch.matmul(torch.softmax(scores, dim=-1), value.float())
    expected = expected.transpose(1, 2).reshape_as(actual)

    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fa4_learnable_sink_matches_reference():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FA4 requires SM90+")
    pytest.importorskip("flash_attn.cute")

    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, seq_len, num_heads, head_dim = 1, 16, 2, 64
    page_size = 256
    sliding_window = 4
    scaling = head_dim**-0.5

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    sinks = torch.randn(num_heads, device=device, dtype=dtype)
    key_cache = torch.zeros(1, page_size, num_heads, head_dim, device=device, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    set_kv_cache_context(
        is_paged_attn=True,
        key_cache={0: key_cache},
        value_cache={0: value_cache},
        block_tables=torch.tensor([[0]], device=device, dtype=torch.int32),
        context_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
        slot_mapping=torch.arange(seq_len, device=device, dtype=torch.int64),
        query_start_loc=torch.tensor([0, seq_len], device=device, dtype=torch.int32),
        max_seqlen_q=seq_len,
    )

    module = torch.nn.Module()
    module.layer_idx = 0
    actual, _ = flash_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=None,
        scaling=scaling,
        sliding_window=sliding_window,
        s_aux=sinks,
        use_cute=True,
    )

    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    positions = torch.arange(seq_len, device=device)
    query_positions = positions[:, None]
    key_positions = positions[None, :]
    mask = (key_positions <= query_positions) & (key_positions > query_positions - sliding_window)
    scores.masked_fill_(~mask, float("-inf"))
    sink_scores = sinks.float().view(1, num_heads, 1, 1).expand(batch_size, -1, seq_len, -1)
    attention_probs = torch.softmax(torch.cat((scores, sink_scores), dim=-1), dim=-1)[..., :-1]
    expected = torch.matmul(attention_probs, value.float())
    expected = expected.transpose(1, 2).reshape_as(actual)

    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA3 requires CUDA")
@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 9, reason="FA3 needs SM90")
@pytest.mark.parametrize("max_seqlen_q", [1, 64])
def test_vllm_fa3_matches_upstream_fa3(max_seqlen_q):
    """vexact prefers vLLM's FA3 build when the `vllm` extra is installed: it must return the same bits as the
    upstream wheel, including with the max_seqlen_q a pure-decode graph passes."""
    vllm_fa = pytest.importorskip("vllm.vllm_flash_attn").flash_attn_varlen_func
    from flash_attn_interface import flash_attn_with_kvcache

    heads, kv_heads, head_dim, page, batch = 32, 8, 128, 256, 64
    gen = torch.Generator(device="cuda").manual_seed(max_seqlen_q)
    blocks_per_seq = 4
    key_cache = torch.randn(batch * blocks_per_seq, page, kv_heads, head_dim, device="cuda", generator=gen).bfloat16()
    value_cache = torch.randn_like(key_cache)
    block_table = torch.randperm(key_cache.shape[0], device="cuda", generator=gen).view(batch, blocks_per_seq).int()
    seqlens = torch.randint(300, 600, (batch,), device="cuda", generator=gen, dtype=torch.int32)
    query = torch.randn(batch, heads, head_dim, device="cuda", generator=gen).bfloat16()
    cu_seqlens_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    scale = head_dim**-0.5

    upstream = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=cu_seqlens_q,
        cache_seqlens=seqlens,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=scale,
        causal=True,
        page_table=block_table,
        num_splits=1,
    )
    # max_seqlen_k may be any upper bound; vexact passes the block table's capacity.
    fused = vllm_fa(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqlens,
        max_seqlen_k=blocks_per_seq * page,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        num_splits=1,
        fa_version=3,
    )
    assert torch.equal(fused.view(torch.int16), upstream.view(torch.int16))
