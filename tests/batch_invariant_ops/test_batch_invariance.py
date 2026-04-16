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
import torch.nn.functional as F
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

from vexact.batch_invariant_ops import set_batch_invariant_mode


torch.set_default_device("cuda")

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass


def test_batch_invariance():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B * D).reshape(B, D)
    b = torch.linspace(-100, 100, D * D).reshape(D, D)

    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[1:2], b)

    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = torch.mm(a, b)[1:2]

    # Check if results are identical
    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0


def test_bmm_batch_invariance():
    B = 4
    K = 64  # Keep K dimension constant for valid matrix multiplication
    # Different M and N dimensions for each batch element
    M_sizes = [32, 48, 24, 56]
    N_sizes = [40, 64, 32, 48]
    max_M = max(M_sizes)
    max_N = max(N_sizes)

    # Create zero-padded tensors
    a = torch.zeros(B, max_M, K)
    b = torch.zeros(B, K, max_N)

    # Fill in actual data for each batch with different dimensions
    for i, (M, N) in enumerate(zip(M_sizes, N_sizes)):
        a[i, :M, :] = torch.linspace(-10, 10, M * K).reshape(M, K)
        b[i, :, :N] = torch.linspace(-10, 10, K * N).reshape(K, N)

    # Method 1: Batch matrix multiplication with single batch (first element)
    first_M, first_N = M_sizes[0], N_sizes[0]
    single_a = a[1:2, 1:first_M, :]
    single_b = b[1:2, :, 1:first_N]
    out1 = torch.bmm(single_a, single_b)

    # Method 2: Batch matrix multiplication with full batch, then slice
    out_full = torch.bmm(a, b)
    out2 = out_full[1:2, 1:first_M, 1:first_N]

    # Check if results are identical for the first batch element
    diff = (out1 - out2).abs().max()
    print(f"BMM Difference (variable dimensions): {diff.item()}")
    print(f"Matrix dimensions: {list(zip(M_sizes, N_sizes))}")

    return diff.item() == 0


def test_softmax_simple_batch_invariance():
    """Test simple softmax batch invariance with pre-computed logits.

    This is the original test that shows no variance because it doesn't
    involve the Q@K^T computation that creates batch-dependent numerical behavior.
    """
    B = 4
    H = 8
    # Different sequence lengths for each batch element
    seq_lengths = [32, 48, 24, 56]
    max_seq_len = max(seq_lengths)

    # # Create padded logits with different effective sequence lengths
    # logits = torch.zeros(B, H, max_seq_len, max_seq_len, dtype=torch.bfloat16)
    masks = torch.full((B, 1, max_seq_len, max_seq_len), float("-inf"), dtype=torch.bfloat16)
    logits = torch.zeros(B, H, max_seq_len, max_seq_len, dtype=torch.bfloat16)

    # Fill in logits and masks for each sequence length
    for i, seq_len in enumerate(seq_lengths):
        # Create logits for this sequence
        logits[i, :, :seq_len, :seq_len] = torch.linspace(-10, 10, H * seq_len * seq_len, dtype=torch.bfloat16).reshape(
            H, seq_len, seq_len
        )

        # Create causal mask for this sequence length
        causal_mask = torch.tril(torch.ones(1, seq_len, seq_len, dtype=torch.bool))
        masks[i, :, :seq_len, :seq_len] = torch.where(causal_mask, 0.0, float("-inf"))

    # Apply masks to logits
    masked_logits = logits + masks

    # Method 1: Softmax with single batch (first element only)
    first_seq_len = seq_lengths[0]
    single_logits = masked_logits[:1, :, :first_seq_len, :first_seq_len]
    out1 = F.softmax(single_logits, dim=-1)

    # Method 2: Softmax with full batch, then slice
    out_full = F.softmax(masked_logits, dim=-1)
    out2 = out_full[:1, :, :first_seq_len, :first_seq_len]

    # Check if results are identical for the first sequence
    diff = (out1 - out2).abs().max()
    print(f"Simple Softmax Difference (variable seq lengths): {diff.item()}")
    print(f"Sequence lengths: {seq_lengths}")

    return diff.item() < 1e-6  # Use small tolerance for floating point


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flash-attn cross entropy requires CUDA")
@pytest.mark.parametrize("num_tokens, vocab_size", [(256, 1024), (513, 1536), (256, 151936)])
def test_flash_attn_cross_entropy_batch_invariance(num_tokens: int, vocab_size: int) -> None:
    torch.manual_seed(123)
    logits = torch.randn(num_tokens, vocab_size, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, vocab_size, (num_tokens,), device="cuda", dtype=torch.long)

    with set_batch_invariant_mode(True):
        full_losses, _ = cross_entropy_loss(logits, labels, inplace_backward=False)

    permutation = torch.randperm(num_tokens, device=logits.device)
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(num_tokens, device=logits.device)

    with set_batch_invariant_mode(True):
        perm_losses, _ = cross_entropy_loss(
            logits[permutation].contiguous(),
            labels[permutation].contiguous(),
            inplace_backward=False,
        )

    torch.testing.assert_close(
        full_losses,
        perm_losses[inverse_permutation],
        rtol=0.0,
        atol=0.0,
    )


# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    is_deterministic = test_batch_invariance()
    print(f"MM Deterministic: {is_deterministic}")

    is_deterministic_bmm = test_bmm_batch_invariance()
    print(f"BMM Deterministic: {is_deterministic_bmm}")

    is_deterministic_softmax_simple = test_softmax_simple_batch_invariance()
    print(f"Simple Softmax Deterministic: {is_deterministic_softmax_simple}")

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    is_deterministic = test_batch_invariance()
    print(f"MM Deterministic: {is_deterministic}")

    is_deterministic_bmm = test_bmm_batch_invariance()
    print(f"BMM Deterministic: {is_deterministic_bmm}")

    is_deterministic_softmax_simple = test_softmax_simple_batch_invariance()
    print(f"Simple Softmax Deterministic: {is_deterministic_softmax_simple}")
