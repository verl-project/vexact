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

import torch


try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Flash Attention not available, skipping tests")

torch.set_default_device("cuda")


def test_flash_attention_decode_batch_invariance():
    if not FLASH_ATTN_AVAILABLE:
        print("Flash Attention not available, skipping test")
        return True, True

    # Set up test parameters
    H = 1  # Number of attention heads
    D = 8  # Head dimension
    page_size = 256  # KV cache page size (must be divisible by 256 for Flash Attention)

    # Simulate mixed batch: prefill sequences and decode sequences
    # Prefill sequences: processing multiple new tokens
    prefill_seq_lens = [300]  # New tokens being processed
    prefill_kv_lens = [300]  # Existing KV cache lengths

    # Create cumulative sequence lengths
    prefill_cu_seqlens_q = torch.tensor(
        [0] + [sum(prefill_seq_lens[: i + 1]) for i in range(len(prefill_seq_lens))],
        dtype=torch.int32,
    )
    prefill_cu_seqlens_k = torch.tensor(
        [0] + [sum(prefill_kv_lens[: i + 1]) for i in range(len(prefill_seq_lens))],
        dtype=torch.int32,
    )

    max_seqlen_q = max(prefill_seq_lens)
    max_seqlen_k = max(prefill_kv_lens)

    # Create query tensors (new tokens)
    prefill_q_tokens = []
    prefill_k_tokens = []
    prefill_v_tokens = []
    for seq_len in prefill_seq_lens:
        prefill_q_tokens.append(torch.randn(seq_len, H, D, dtype=torch.bfloat16) * 0.1)
    for seq_len in prefill_kv_lens:
        prefill_k_tokens.append(torch.randn(seq_len, H, D, dtype=torch.bfloat16) * 0.1)
        prefill_v_tokens.append(torch.randn(seq_len, H, D, dtype=torch.bfloat16) * 0.1)

    prefill_q = torch.cat(prefill_q_tokens, dim=0)
    prefill_k = torch.cat(prefill_k_tokens, dim=0)
    prefill_v = torch.cat(prefill_v_tokens, dim=0)

    all_prefill_out = flash_attn_varlen_func(
        prefill_q,
        prefill_k,
        prefill_v,
        cu_seqlens_q=prefill_cu_seqlens_q,
        cu_seqlens_k=prefill_cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )

    # One important finding: For chunk prefill, if not start chunking at 0, it will have different result
    chunck_prefill_out = flash_attn_varlen_func(
        prefill_q[128:280],
        prefill_k[:280],
        prefill_v[:280],
        cu_seqlens_q=torch.tensor([0, 280], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 280], dtype=torch.int32),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )
    prefill_diff = (chunck_prefill_out - all_prefill_out[128:280]).abs().max()
    print(f"flash_attn_varlen_func paged Flash Attention Batch Invariant Difference for batch: {prefill_diff.item()}")

    # Decode sequences: generating one new token
    decode_seq_lens = [1, 1, 1, 1]  # Single new token
    decode_kv_lens = [64, 112, 256, 280]  # Existing KV cache lengths

    decode_batch = len(decode_seq_lens)

    decode_cu_seqlens_q = torch.tensor(
        [0] + [sum(decode_seq_lens[: i + 1]) for i in range(decode_batch)],
        dtype=torch.int32,
    )
    decode_cu_seqlens_k = torch.tensor(
        [0] + [sum(decode_kv_lens[: i + 1]) for i in range(decode_batch)],
        dtype=torch.int32,
    )
    decode_context_lens = torch.tensor(decode_kv_lens, dtype=torch.int32)

    q_linear = [prefill_q[decode_kv_lens[i] : decode_kv_lens[i] + 1] for i in range(decode_batch)]
    decode_q = torch.cat(q_linear)

    # Create paged KV cache
    # Calculate total pages needed
    max_pages_per_seq = max((kv_len + page_size - 1) // page_size for kv_len in decode_kv_lens)
    total_pages = sum((kv_len + page_size - 1) // page_size for kv_len in decode_kv_lens)

    # Create paged KV cache tensors: [num_pages, page_size, num_heads, head_dim]
    k_cache_paged = torch.randn(total_pages, page_size, H, D, dtype=torch.bfloat16) * 0.1
    v_cache_paged = torch.randn(total_pages, page_size, H, D, dtype=torch.bfloat16) * 0.1

    # Create block table: [batch_size, max_pages_per_seq]
    block_table = torch.full((decode_batch, max_pages_per_seq), -1, dtype=torch.int32)

    # Fill the paged cache and block table
    page_idx = 0
    k_linear = []  # For concatenated format
    v_linear = []

    for i, (seq_len, kv_len) in enumerate(zip(decode_seq_lens, decode_kv_lens)):
        pages_needed = (kv_len + page_size - 1) // page_size
        block_table[i, :pages_needed] = torch.arange(page_idx, page_idx + pages_needed, dtype=torch.int32)

        full_k = prefill_k[:kv_len]
        full_v = prefill_v[:kv_len]

        # Store in linear format for comparison
        k_linear.append(full_k)
        v_linear.append(full_v)

        # Fill paged cache
        for page in range(pages_needed):
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, kv_len)
            actual_len = end_idx - start_idx

            if actual_len > 0:
                k_cache_paged[page_idx + page, :actual_len] = full_k[start_idx:end_idx]
                v_cache_paged[page_idx + page, :actual_len] = full_v[start_idx:end_idx]

        page_idx += pages_needed

    # For non-paged comparison, use linear concatenated format
    all_q = torch.cat(prefill_q_tokens + q_linear)
    all_k = torch.cat(prefill_k_tokens + k_linear, dim=0)
    all_v = torch.cat(prefill_v_tokens + v_linear, dim=0)

    # Combine all sequences
    all_seq_lens = decode_seq_lens + prefill_seq_lens
    all_kv_lens = decode_kv_lens + prefill_kv_lens
    all_seq_lens = prefill_seq_lens + decode_seq_lens
    all_kv_lens = prefill_kv_lens + decode_kv_lens
    total_batches = len(all_seq_lens)
    all_cu_seqlens_q = torch.tensor(
        [0] + [sum(all_seq_lens[: i + 1]) for i in range(total_batches)],
        dtype=torch.int32,
    )
    all_cu_seqlens_k = torch.tensor(
        [0] + [sum(all_kv_lens[: i + 1]) for i in range(total_batches)],
        dtype=torch.int32,
    )

    p_and_d_varlen_out = flash_attn_varlen_func(
        all_q,
        all_k,
        all_v,
        cu_seqlens_q=all_cu_seqlens_q,
        cu_seqlens_k=all_cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        block_table=None,
    )

    for i in range(decode_batch):
        decode_pos = all_cu_seqlens_q[i + 1]
        fused_paged_diff = (p_and_d_varlen_out[decode_pos] - all_prefill_out[decode_kv_lens[i] - 1]).abs().max()
        print(
            f"Fused Prefill and Decode flash_attn_varlen_func paged Flash Attention Batch Invariant Difference "
            f"for batch {i} on position {decode_kv_lens[i] - 1}: {fused_paged_diff.item()}"
        )

    all_varlen_out = flash_attn_varlen_func(
        decode_q,
        k_cache_paged,
        v_cache_paged,
        cu_seqlens_q=decode_cu_seqlens_q,
        cu_seqlens_k=decode_cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        block_table=block_table,
    )
    one_varlen_out = flash_attn_varlen_func(
        decode_q[0 : decode_seq_lens[0]],
        k_cache_paged,
        v_cache_paged,
        cu_seqlens_q=decode_cu_seqlens_q[0:2],
        cu_seqlens_k=decode_cu_seqlens_k[0:2],
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        block_table=block_table[0:1],
    )

    # Compare paged attention outputs
    paged_diff = (all_varlen_out[0 : decode_seq_lens[0]] - one_varlen_out).abs().max()
    print(f"Decode flash_attn_varlen_func paged Flash Attention Batch Invariant Difference: {paged_diff.item()}")

    for i in range(decode_batch):
        decode_paged_diff = (all_varlen_out[i] - all_prefill_out[decode_kv_lens[i] - 1]).abs().max()
        print(
            f"Decode flash_attn_varlen_func paged Flash Attention Batch Invariant Difference "
            f"for batch {i} on position {decode_kv_lens[i] - 1}: {decode_paged_diff.item()}"
        )

    all_kvcache_out = flash_attn_with_kvcache(
        decode_q.unsqueeze(1),
        k_cache_paged,
        v_cache_paged,
        cache_seqlens=decode_context_lens,
        block_table=block_table,
        softmax_scale=None,
        causal=True,
    )
    one_kvcache_out = flash_attn_with_kvcache(
        decode_q.unsqueeze(1)[0:1],
        k_cache_paged,
        v_cache_paged,
        cache_seqlens=decode_context_lens[0:1],
        block_table=block_table[0:1],
        softmax_scale=None,
        causal=True,
    )

    # Compare paged attention outputs
    kv_diff = (all_kvcache_out[0 : decode_seq_lens[0]] - one_kvcache_out).abs().max()
    print(f"Decode flash_attn_with_kvcache paged Flash Attention Batch Invariant Difference: {paged_diff.item()}")

    for i in range(decode_batch):
        decode_paged_diff = (all_kvcache_out[i] - all_prefill_out[decode_kv_lens[i] - 1]).abs().max()
        print(
            f"Decode flash_attn_with_kvcache paged Flash Attention Batch Invariant Difference "
            f"for batch {i} on position {decode_kv_lens[i] - 1}: {decode_paged_diff.item()}"
        )

    return paged_diff and kv_diff


def test_flash_attention_prefill_decode_batch_invariance():
    """Test Flash Attention varlen with mixed prefill and decode tokens using paged attention.

    This test simulates a realistic inference scenario where some sequences are in prefill
    phase (processing multiple new tokens) and others are in decode phase (generating
    one token at a time), using paged attention to manage the KV cache.
    """
    if not FLASH_ATTN_AVAILABLE:
        print("Flash Attention not available, skipping test")
        return True, True

    # Set up test parameters
    H = 8  # Number of attention heads
    D = 64  # Head dimension
    page_size = 256  # KV cache page size (must be divisible by 256 for Flash Attention)

    # Simulate mixed batch: prefill sequences and decode sequences
    # Prefill sequences: processing multiple new tokens
    prefill_seq_lens = [300, 220]  # New tokens being processed
    prefill_kv_lens = [300, 220]  # Existing KV cache lengths

    # Decode sequences: generating one new token
    decode_seq_lens = [1, 1]  # Single new token
    decode_kv_lens = [64, 112]  # Existing KV cache lengths

    # Combine all sequences
    # all_seq_lens = decode_seq_lens + prefill_seq_lens
    # all_kv_lens =  decode_kv_lens + prefill_kv_lens
    all_seq_lens = prefill_seq_lens + decode_seq_lens
    all_kv_lens = prefill_kv_lens + decode_kv_lens
    total_batches = len(all_seq_lens)

    # Create cumulative sequence lengths
    cu_seqlens_q = torch.tensor(
        [0] + [sum(all_seq_lens[: i + 1]) for i in range(total_batches)],
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0] + [sum(all_kv_lens[: i + 1]) for i in range(total_batches)],
        dtype=torch.int32,
    )

    max_seqlen_q = max(all_seq_lens)
    max_seqlen_k = max(all_kv_lens)

    # Create query tensors (new tokens)
    q_tokens = []
    for seq_len in all_seq_lens:
        q_tokens.append(torch.randn(seq_len, H, D, dtype=torch.bfloat16) * 0.1)
    q = torch.cat(q_tokens, dim=0)

    # Create paged KV cache
    # Calculate total pages needed
    max_pages_per_seq = max((kv_len + page_size - 1) // page_size for kv_len in all_kv_lens)
    total_pages = sum((kv_len + page_size - 1) // page_size for kv_len in all_kv_lens)

    # Create paged KV cache tensors: [num_pages, page_size, num_heads, head_dim]
    k_cache_paged = torch.randn(total_pages, page_size, H, D, dtype=torch.bfloat16) * 0.1
    v_cache_paged = torch.randn(total_pages, page_size, H, D, dtype=torch.bfloat16) * 0.1

    # Create block table: [batch_size, max_pages_per_seq]
    block_table = torch.full((total_batches, max_pages_per_seq), -1, dtype=torch.int32)

    # Fill the paged cache and block table
    page_idx = 0
    k_linear = []  # For concatenated format
    v_linear = []

    for i, (seq_len, kv_len) in enumerate(zip(all_seq_lens, all_kv_lens)):
        pages_needed = (kv_len + page_size - 1) // page_size
        block_table[i, :pages_needed] = torch.arange(page_idx, page_idx + pages_needed, dtype=torch.int32)

        # Create sequence data (existing cache + new tokens)
        existing_len = kv_len - seq_len
        existing_k = torch.randn(existing_len, H, D, dtype=torch.bfloat16) * 0.1
        existing_v = torch.randn(existing_len, H, D, dtype=torch.bfloat16) * 0.1
        new_k = torch.randn(seq_len, H, D, dtype=torch.bfloat16) * 0.1
        new_v = torch.randn(seq_len, H, D, dtype=torch.bfloat16) * 0.1

        full_k = torch.cat([existing_k, new_k], dim=0)
        full_v = torch.cat([existing_v, new_v], dim=0)

        # Store in linear format for comparison
        k_linear.append(full_k)
        v_linear.append(full_v)

        # Fill paged cache
        for page in range(pages_needed):
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, kv_len)
            actual_len = end_idx - start_idx

            if actual_len > 0:
                k_cache_paged[page_idx + page, :actual_len] = full_k[start_idx:end_idx]
                v_cache_paged[page_idx + page, :actual_len] = full_v[start_idx:end_idx]

        page_idx += pages_needed

    # For non-paged comparison, use linear concatenated format
    k = torch.cat(k_linear, dim=0)
    v = torch.cat(v_linear, dim=0)

    all_page_out = flash_attn_varlen_func(
        q,
        k_cache_paged,
        v_cache_paged,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        block_table=block_table,
    )
    one_all_page_out = flash_attn_varlen_func(
        q[0 : all_seq_lens[0]],
        k_cache_paged,
        v_cache_paged,
        cu_seqlens_q=cu_seqlens_q[0:2],
        cu_seqlens_k=cu_seqlens_k[0:2],
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        block_table=block_table[0:1],
    )

    # Compare paged attention outputs
    paged_diff = (all_page_out[0 : all_seq_lens[0]] - one_all_page_out).abs().max()
    print(f"Paged Flash Attention Batch Invariant Difference: {paged_diff.item()}")

    all_linear_out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )
    one_linear_out = flash_attn_varlen_func(
        q[0 : all_seq_lens[0]],
        k[0 : all_kv_lens[0]],
        v[0 : all_kv_lens[0]],
        cu_seqlens_q=cu_seqlens_q[0:2],
        cu_seqlens_k=cu_seqlens_k[0:2],
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )

    linear_diff = (all_linear_out[0 : all_seq_lens[0]] - one_linear_out).abs().max()
    print(f"Linear Flash Attention Batch Invariant Difference: {linear_diff.item()}")

    # Compare paged vs linear (should be identical when no batch invariant mode)
    # paged_vs_linear_diff = (out_paged_standard - out_linear_standard).abs().max()
    # print(f"Paged vs Linear Attention Difference: {paged_vs_linear_diff.item()}")

    print("\nTest Configuration:")
    print(f"Prefill sequence lengths: {prefill_seq_lens}")
    print(f"Decode sequence lengths: {decode_seq_lens}")
    print(f"KV cache lengths: {all_kv_lens}")
    print(f"Page size: {page_size}")
    print(f"Total tokens - Q: {sum(all_seq_lens)}, K/V: {sum(all_kv_lens)}")
    print(f"Total pages: {total_pages}")

    # Check if outputs are consistent
    paged_passed = paged_diff.item() < 1e-5
    linear_passed = linear_diff.item() < 1e-5

    print("\nResults:")
    print(f"Paged attention batch invariant consistency: {'PASS' if paged_passed else 'FAIL'}")
    print(f"Linear attention batch invariant consistency: {'PASS' if linear_passed else 'FAIL'}")
    # print(f"Paged vs Linear equivalence: {'PASS' if paged_vs_linear_passed else 'FAIL'}")

    return paged_passed and linear_passed


if __name__ == "__main__":
    if not FLASH_ATTN_AVAILABLE:
        print("Flash Attention 2 not available. Please install with: pip install flash-attn")
        exit(1)

    print("Testing Flash Attention 2 with Mixed Prefill/Decode and Paged Attention")
    print("=" * 80)

    batch_invariant_result = test_flash_attention_prefill_decode_batch_invariance()

    test_flash_attention_decode_batch_invariance()

    print("\n" + "=" * 80)
    print("Test Summary:")
    print(f"Batch invariant consistency: {'PASS' if batch_invariant_result else 'FAIL'}")

    if batch_invariant_result:
        print(
            "✓ Flash Attention (both paged and linear) produces identical results with and without batch invariant mode"
        )
    else:
        print("✗ Flash Attention results differ between standard and batch invariant modes")
