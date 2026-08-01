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

from vexact.batch_invariant_ops.flash_attention import _get_fa_window_size, flash_attention_forward
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fa3_sliding_window_matches_reference():
    if torch.cuda.get_device_capability()[0] != 9:
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
