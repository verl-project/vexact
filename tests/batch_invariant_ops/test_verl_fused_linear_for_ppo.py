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

"""Batch-invariance checks for VeRL's FusedLinearForPPO helper."""

import pytest
import torch

from verl.utils.experimental.torch_functional import FusedLinearForPPO
from verl.utils.torch_functional import logprobs_from_logits
from vexact.batch_invariant_ops import set_batch_invariant_mode


def _build_inputs(
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    vocab_size: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    hidden = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda", dtype=torch.long)
    return hidden, weight, labels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FusedLinearForPPO requires CUDA")
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, vocab_size, chunk_size",
    [
        (16, 2048, 1024, 4096, 64),
        (1, 33, 96, 384, 7),
    ],
)
def test_fused_linear_for_ppo_batch_invariance_forward(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    vocab_size: int,
    chunk_size: int,
) -> None:
    fused_linear_for_ppo = FusedLinearForPPO(chunk_size=chunk_size)
    hidden, weight, labels = _build_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        seed=11,
    )

    with set_batch_invariant_mode(True):
        full_log_probs, _ = fused_linear_for_ppo(hidden, weight, labels)

    total_tokens = batch_size * seq_len
    permutation = torch.randperm(total_tokens, device=hidden.device)
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(total_tokens, device=hidden.device)

    hidden_perm = hidden.view(total_tokens, hidden_size)[permutation].contiguous().view_as(hidden)
    labels_perm = labels.view(total_tokens)[permutation].contiguous().view_as(labels)

    with set_batch_invariant_mode(True):
        perm_log_probs, _ = fused_linear_for_ppo(hidden_perm, weight, labels_perm)

    torch.testing.assert_close(
        full_log_probs.view(-1),
        perm_log_probs.view(-1)[inverse_permutation],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FusedLinearForPPO requires CUDA")
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, vocab_size, chunk_size",
    [
        (16, 2048, 1024, 4096, 64),
        (1, 33, 96, 384, 7),
    ],
)
def test_fused_linear_for_ppo_log_probs_float32_in_bfloat16_model(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    vocab_size: int,
    chunk_size: int,
) -> None:
    """
    checks whether VeRL FusedLinearForPPO can produce the exact same logprobs
    with vexact when given the same hidden states
    """
    fused_linear_for_ppo = FusedLinearForPPO(chunk_size=chunk_size)
    hidden, weight, labels = _build_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        seed=42,
        dtype=torch.bfloat16,
    )
    rolled_labels = torch.roll(labels, shifts=-1, dims=-1)

    with set_batch_invariant_mode(True):
        # verl FusedLinearForPPO
        fused_log_probs, _ = fused_linear_for_ppo(hidden, weight, rolled_labels)
        # vexact when use_fp32_logits
        logits = torch.matmul(hidden, weight.t())
        ref_log_probs = logprobs_from_logits(
            logits=logits.to(torch.float32), labels=rolled_labels, inplace_backward=False
        )

    assert fused_log_probs.dtype == torch.float32, (
        f"Expected float32 log_probs but got {fused_log_probs.dtype}; log_probs must not be downcast to the model dtype"
    )
    torch.testing.assert_close(
        fused_log_probs,
        ref_log_probs,
        rtol=0,
        atol=0,
    )
