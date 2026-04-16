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

"""Pytest checks for fused MoE Triton kernels."""

import pytest
import torch

from vexact.batch_invariant_ops.fused_moe import fused_moe_forward


def _build_test_inputs(
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    ffn_dim: int,
    num_experts: int,
    top_k: int,
    seed: int | None = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    total_tokens = batch_size * seq_len

    hidden_states = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)
    selected_experts = torch.randint(
        low=0,
        high=num_experts,
        size=(total_tokens, top_k),
        device=device,
        dtype=torch.int32,
    )
    gate_logits = torch.randn(total_tokens, top_k, device=device, dtype=torch.float32)
    gate_weights = torch.softmax(gate_logits, dim=-1).to(dtype)

    fc1_1_weight = torch.randn(num_experts, ffn_dim, hidden_size, device=device, dtype=dtype)
    fc1_2_weight = torch.randn(num_experts, ffn_dim, hidden_size, device=device, dtype=dtype)
    fc2_weight = torch.randn(num_experts, hidden_size, ffn_dim, device=device, dtype=dtype)

    return {
        "num_experts": num_experts,
        "hidden_states": hidden_states,
        "selected_experts": selected_experts,
        "gate_weights": gate_weights,
        "fc1_1_weight": fc1_1_weight,
        "fc1_2_weight": fc1_2_weight,
        "fc2_weight": fc2_weight,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused MoE kernels require CUDA")
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, ffn_dim, num_experts, top_k",
    [
        (4, 3, 64, 128, 4, 2),
        (2, 6, 96, 192, 6, 2),
        (1, 8, 128, 256, 4, 1),
    ],
)
def test_fused_moe_batch_invariance(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    ffn_dim: int,
    num_experts: int,
    top_k: int,
) -> None:
    device = torch.device("cuda")
    threshold = 1e-3

    diffs: list[float] = []
    for seed in range(3):
        inputs = _build_test_inputs(
            device=device,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            seed=seed,
        )
        gate_weights = inputs["gate_weights"]
        selected_experts = inputs["selected_experts"]
        hidden_states = inputs["hidden_states"]
        fc1_1_weight = inputs["fc1_1_weight"]
        fc1_2_weight = inputs["fc1_2_weight"]
        fc2_weight = inputs["fc2_weight"]

        full_out = fused_moe_forward(
            num_experts,
            gate_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
        )

        token_outputs = []
        for token_idx in range(hidden_states.shape[0]):
            token_outputs.append(
                fused_moe_forward(
                    num_experts,
                    gate_weights[token_idx : token_idx + 1].clone(),
                    selected_experts[token_idx : token_idx + 1].clone(),
                    hidden_states[token_idx : token_idx + 1].clone(),
                    fc1_1_weight,
                    fc1_2_weight,
                    fc2_weight,
                )
            )

        stacked_out = torch.cat(token_outputs, dim=0)
        diff = (full_out - stacked_out).abs().max().float().item()
        diffs.append(diff)

    assert max(diffs) <= threshold
