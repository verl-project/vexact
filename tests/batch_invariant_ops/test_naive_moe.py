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

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vexact.batch_invariant_ops import set_batch_invariant_mode


transformers_activations = pytest.importorskip("transformers.activations")
ACT2FN = transformers_activations.ACT2FN


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = int(expert_idx.item())
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


def _build_qwen3_moe_config(hidden_size: int = 64, num_experts: int = 4, top_k: int = 2) -> SimpleNamespace:
    intermediate_size = hidden_size * 4
    return SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
        hidden_act="silu",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Batch invariance overrides require CUDA kernels",
)
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        (4, 1024, 512),
        (4, 2048, 512),
        (2, 4096, 1024),
    ],
)
def test_sparse_moe_batch_invariance(batch_size: int, seq_len: int, hidden_size: int) -> None:
    device = torch.device("cuda")
    config = _build_qwen3_moe_config(hidden_size=hidden_size)
    moe_block = Qwen3MoeSparseMoeBlock(config).to(device)
    moe_block.eval()

    with torch.no_grad():
        for param in moe_block.parameters():
            param.uniform_(-0.5, 0.5)

    for seed in range(3):
        torch.manual_seed(seed)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.float32, device=device)

        with set_batch_invariant_mode(True):
            full_out, _ = moe_block(hidden_states)
            per_example_outputs = []
            for i in range(batch_size):
                per_out, _ = moe_block(hidden_states[i : i + 1].clone())
                per_example_outputs.append(per_out)
        stacked_out = torch.cat(per_example_outputs, dim=0)

        assert torch.allclose(full_out, stacked_out, atol=0.0, rtol=0.0)
