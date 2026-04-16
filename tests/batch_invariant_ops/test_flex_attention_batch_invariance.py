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

import os
import sys

import torch
import torch.nn as nn


# Add parent directory to path to import kv_cache_context
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from vexact.batch_invariant_ops.flex_attention import flex_attention_forward
from vexact.batch_invariant_ops.kv_cache_context import set_kv_cache_context


# from vexact.batch_invariant_ops.flex_attention import flex_attention_forward, repeat_kv

torch.set_default_device("cuda")

# No longer needed - using context system


class MockModule(nn.Module):
    def __init__(self, num_key_value_groups, num_attention_heads, num_key_value_heads, head_dim):
        super().__init__()
        self.num_key_value_groups = num_key_value_groups
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim


def eager_attention(query, key, value, mask, scale):
    # query: (B, H, S_q, D)
    # key: (B, H, S_kv, D)
    # value: (B, H, S_kv, D)
    # mask: (1, 1, S_q, S_kv) or (B, 1, S_q, S_kv)

    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scale
    if mask is not None:
        attn_weights = attn_weights + mask
    attn_weights = torch.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output


def test_flex_attention_paged_batch_invariance():
    # Test parameters
    H = 8  # Number of attention heads
    D = 64  # Head dimension
    page_size = 16
    num_key_value_heads = 4
    num_key_value_groups = H // num_key_value_heads

    # Sequence lengths for a batch of requests
    decode_seq_lens = [1, 1, 1, 1]
    decode_kv_lens = [64, 112, 256, 280]
    decode_batch = len(decode_seq_lens)

    # --- Prepare data for the test ---

    # Create query tensors
    q_tokens = [torch.randn(1, H, D, dtype=torch.bfloat16) * 0.1 for _ in decode_kv_lens]
    q_batched = torch.cat(q_tokens, dim=0)

    # Create linear (contiguous) KV cache for ground truth
    k_linear_all = []
    v_linear_all = []
    for kv_len in decode_kv_lens:
        k_linear_all.append(torch.randn(kv_len, num_key_value_heads, D, dtype=torch.bfloat16) * 0.1)
        v_linear_all.append(torch.randn(kv_len, num_key_value_heads, D, dtype=torch.bfloat16) * 0.1)

    # Create paged KV cache and block tables
    max_pages_per_seq = max((kv_len + page_size - 1) // page_size for kv_len in decode_kv_lens)
    needed_pages = sum((kv_len + page_size - 1) // page_size for kv_len in decode_kv_lens)
    total_pages = needed_pages + 1

    k_cache_paged = torch.randn(total_pages, page_size, num_key_value_heads, D, dtype=torch.bfloat16) * 0.1
    v_cache_paged = torch.randn(total_pages, page_size, num_key_value_heads, D, dtype=torch.bfloat16) * 0.1
    block_table = torch.full((decode_batch, max_pages_per_seq), -1, dtype=torch.int32)

    page_idx = 1
    for i, kv_len in enumerate(decode_kv_lens):
        pages_needed = (kv_len + page_size - 1) // page_size
        block_table[i, :pages_needed] = torch.arange(page_idx, page_idx + pages_needed, dtype=torch.int32)

        full_k = k_linear_all[i]
        full_v = v_linear_all[i]

        for page in range(pages_needed):
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, kv_len)
            actual_len = end_idx - start_idx

            if actual_len > 0:
                k_cache_paged[page_idx + page, :actual_len] = full_k[start_idx:end_idx]
                v_cache_paged[page_idx + page, :actual_len] = full_v[start_idx:end_idx]

        page_idx += pages_needed

    mock_module = MockModule(num_key_value_groups, H, num_key_value_heads, D)
    scaling = float(D) ** -0.5
    # --- Run 1: Paged attention on the whole batch ---

    # Set up KV cache context for batched paged attention
    set_kv_cache_context(
        is_paged_attn=True,
        key_cache={0: k_cache_paged},
        value_cache={0: v_cache_paged},
        block_tables=block_table,
        context_lens=torch.tensor(decode_kv_lens, dtype=torch.int32),
        query_start_loc=torch.tensor(
            [0] + [sum(decode_seq_lens[: i + 1]) for i in range(len(decode_seq_lens))],
            dtype=torch.int32,
        ),
        past_lens=torch.tensor(decode_kv_lens, dtype=torch.int32) - 1,
    )
    output_batched_paged, _ = flex_attention_forward(
        mock_module,
        q_batched.unsqueeze(2),
        None,
        None,
        attention_mask=None,
        scaling=scaling,
    )

    # --- Run 2: Paged attention sequence by sequence ---

    output_single_paged_all = []
    for i in range(decode_batch):
        q_single = q_tokens[i]

        # Set up KV cache context for single sequence paged attention
        set_kv_cache_context(
            is_paged_attn=True,
            key_cache={0: k_cache_paged},
            value_cache={0: v_cache_paged},
            block_tables=block_table[i : i + 1],
            context_lens=torch.tensor([decode_kv_lens[i]], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            past_lens=torch.tensor([decode_kv_lens[i]], dtype=torch.int32) - 1,
        )

        output_single_paged, _ = flex_attention_forward(
            mock_module,
            q_single.permute(1, 0, 2).unsqueeze(0),
            None,
            None,
            attention_mask=None,
            scaling=scaling,
        )
        output_single_paged_all.append(output_single_paged)

    output_single_paged_all = torch.cat(output_single_paged_all, dim=0)

    # --- Run 3: Linear attention (ground truth) sequence by sequence ---

    output_linear_all = []
    for i in range(decode_batch):
        q_single = q_tokens[i]
        k_single = k_linear_all[i].unsqueeze(0)
        v_single = v_linear_all[i].unsqueeze(0)
        q_len = q_single.shape[0]

        # Set up KV cache context for linear attention
        set_kv_cache_context(
            is_paged_attn=False,
            query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
            past_lens=torch.tensor([k_single.shape[1] - q_len], dtype=torch.int32),
        )

        output_linear, _ = flex_attention_forward(
            mock_module,
            q_single.permute(1, 0, 2).unsqueeze(0),
            k_single.transpose(1, 2),
            v_single.transpose(1, 2),
            attention_mask=None,
            scaling=scaling,
        )
        output_linear_all.append(output_linear)

    output_linear_all = torch.cat(output_linear_all, dim=0)
    # --- Compare results ---
    # Compare batched paged vs single paged
    batch_invariance_diff = (output_batched_paged - output_single_paged_all).abs().max()
    print(f"Flex Attention Paged Batch Invariant Difference: {batch_invariance_diff.item()}")

    # Compare batched paged vs linear ground truth
    correctness_diff = (output_batched_paged - output_linear_all).abs().max()
    print(f"Flex Attention Paged Correctness Difference (vs Linear): {correctness_diff.item()}")

    assert batch_invariance_diff < 1e-10, "Batch invariance test failed for paged attention."
    assert correctness_diff < 1e-10, "Correctness test failed for paged attention."

    print("\nFlex Attention paged attention test passed!")


def test_flex_attention_vs_eager():
    # Test parameters
    B = 1  # Batch size
    H = 8  # Number of attention heads
    D = 64  # Head dimension
    S_q = 128  # Query sequence length
    S_kv = 128  # Key/Value sequence length
    num_key_value_heads = 4
    num_key_value_groups = H // num_key_value_heads

    mock_module = MockModule(num_key_value_groups, H, num_key_value_heads, D)
    scaling = float(D) ** -0.5

    # Generate data
    q = torch.randn(B, S_q, H, D, dtype=torch.bfloat16) * 0.1
    k_kv = torch.randn(B, S_kv, num_key_value_heads, D, dtype=torch.bfloat16) * 0.1
    v_kv = torch.randn(B, S_kv, num_key_value_heads, D, dtype=torch.bfloat16) * 0.1

    # Create causal mask
    causal_mask = torch.full((S_q, S_kv), -float("inf"), device=q.device, dtype=q.dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S_q, S_kv)

    # --- Eager attention ---
    q_eager = q.transpose(1, 2)  # (B, H, S_q, D)
    k_eager = k_kv.transpose(1, 2).repeat_interleave(num_key_value_groups, dim=1)  # (B, H, S_kv, D)
    v_eager = v_kv.transpose(1, 2).repeat_interleave(num_key_value_groups, dim=1)  # (B, H, S_kv, D)

    eager_output = eager_attention(q_eager, k_eager, v_eager, causal_mask, scaling)

    # --- Flex attention ---
    set_kv_cache_context(
        is_paged_attn=False,
        query_start_loc=torch.tensor([0, S_q], dtype=torch.int32),
        past_lens=torch.tensor([0], dtype=torch.int32),
    )
    flex_output, _ = flex_attention_forward(
        mock_module,
        q.transpose(1, 2),
        k_kv.transpose(1, 2),
        v_kv.transpose(1, 2),
        attention_mask=causal_mask,
        scaling=scaling,
    )

    # --- Compare results ---
    diff = (flex_output - eager_output.squeeze(0).transpose(0, 1)).abs().max()
    print(f"\nFlex Attention vs Eager Attention Difference: {diff.item()}")
    # Relative difference
    denom = eager_output.squeeze(0).transpose(0, 1).abs().max()
    rel_diff = diff / (denom + 1e-8)
    print(f"Flex Attention vs Eager Attention Relative Difference: {rel_diff.item()}")
    # torch.allclose with rtol
    allclose = torch.allclose(flex_output, eager_output.squeeze(0).transpose(0, 1), rtol=1e-3, atol=1e-3)
    print(f"torch.allclose (rtol=1e-3, atol=1e-3): {allclose}")
    assert allclose, "Flex Attention vs Eager Attention test failed (allclose rtol=1e-3, atol=1e-3)."
    print("\nFlex Attention vs Eager Attention test passed!")


if __name__ == "__main__":
    test_flex_attention_paged_batch_invariance()  # passed with/w.o. compile
    # test_flex_attention_chunk_prefill()  # passed with/w.o. compile
    test_flex_attention_vs_eager()  # passed with/w.o. compile
    # test_flex_attention_chunk_prefill_with_cache()  # passed with/w.o. compile
