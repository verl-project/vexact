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
from transformers import AutoModelForCausalLM, GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts, GptOssForCausalLM

from vexact.inferencer.model_loader import init_on_device_without_buffers
from vexact.models.gpt_oss import modeling_gpt_oss


def test_gpt_oss_experts_use_fused_quack(monkeypatch):
    captured = {}

    def fake_fused_quack(**kwargs):
        captured.update(kwargs)
        return torch.tensor([1.0])

    monkeypatch.setattr(modeling_gpt_oss, "_run_fused_quack", fake_fused_quack)
    experts = SimpleNamespace(
        num_experts=4,
        gate_up_proj=torch.tensor([1.0]),
        gate_up_proj_bias=torch.tensor([2.0]),
        down_proj=torch.tensor([3.0]),
        down_proj_bias=torch.tensor([4.0]),
        alpha=1.702,
        limit=7.0,
    )
    hidden_states = torch.tensor([[5.0]])
    router_indices = torch.tensor([[1]])
    routing_weights = torch.tensor([[0.5]])

    output = modeling_gpt_oss.gpt_oss_experts_forward(experts, hidden_states, router_indices, routing_weights)

    assert torch.equal(output, torch.tensor([1.0]))
    assert captured == {
        "num_experts": 4,
        "routing_weights": routing_weights,
        "selected_experts": router_indices,
        "hidden_states": hidden_states,
        "gate_up_proj": experts.gate_up_proj,
        "gate_up_proj_bias": experts.gate_up_proj_bias,
        "down_proj": experts.down_proj,
        "down_proj_bias": experts.down_proj_bias,
        "alpha": 1.702,
        "limit": 7.0,
    }


def test_gpt_oss_experts_require_routing_inputs():
    with pytest.raises(ValueError, match="requires router_indices and routing_weights"):
        modeling_gpt_oss.gpt_oss_experts_forward(SimpleNamespace(), torch.tensor([[1.0]]))


def test_gpt_oss_patch_replaces_experts_forward():
    modeling_gpt_oss.apply_gpt_oss_patches()

    assert GptOssExperts.forward is modeling_gpt_oss.gpt_oss_experts_forward


def test_gpt_oss_uses_common_transformers_model_path():
    config = GptOssConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        head_dim=16,
    )
    with init_on_device_without_buffers("meta"):
        model = AutoModelForCausalLM.from_config(config)

    assert isinstance(model, GptOssForCausalLM)
    assert type(model).__module__.startswith("transformers.models.gpt_oss")
    assert model.model.layers[0].mlp.experts.forward.__func__ is modeling_gpt_oss.gpt_oss_experts_forward
