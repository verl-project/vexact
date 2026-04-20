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

from unittest.mock import patch

import torch
import torch.nn as nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

from vexact.config import PPInfo
from vexact.inferencer.model_loader import ModelCreator
from vexact.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts,
    Qwen3MoeSparseMoeBlock,
    apply_qwen3_moe_patches,
    load_qwen3_moe_weights,
)


apply_qwen3_moe_patches()


class _DummyModel(nn.Module):
    """Minimal container to expose MoE and tied embedding weights."""

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.block = Qwen3MoeSparseMoeBlock(config)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


def _build_test_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        num_hidden_layers=1,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=3,
        num_experts_per_tok=2,
        vocab_size=128,
    )


def _make_expert_weight(config: Qwen3MoeConfig, expert_idx: int, proj: str) -> torch.Tensor:
    shape = (
        (config.hidden_size, config.moe_intermediate_size)
        if proj == "down_proj"
        else (config.moe_intermediate_size, config.hidden_size)
    )
    return torch.full(shape, float(expert_idx + 1))


def test_moe_block_init_uses_fused_experts():
    """Ensure patched MoE blocks use fused expert containers."""
    block = Qwen3MoeSparseMoeBlock(_build_test_config())
    assert isinstance(block.experts, Qwen3MoeExperts)


def test_load_qwen3_moe_weights_single_expert_updates():
    """Verify one-by-one expert weights are copied into fused parameters."""
    config = _build_test_config()
    model = _DummyModel(config)
    with torch.no_grad():
        model.block.experts.gate_proj.zero_()
        model.block.experts.up_proj.zero_()
        model.block.experts.down_proj.zero_()

    weight_items = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        key = f"block.experts.0.{proj}.weight"
        weight_items.append((key, _make_expert_weight(config, 0, proj)))

    loaded_blocks = load_qwen3_moe_weights(model, weight_items, tied_weight_keys=["lm_head.weight"])
    assert loaded_blocks == 1
    torch.testing.assert_close(model.block.experts.gate_proj[0], _make_expert_weight(config, 0, "gate_proj"))
    torch.testing.assert_close(model.block.experts.gate_proj[1], torch.zeros_like(model.block.experts.gate_proj[1]))


def test_qwen3_moe_load_weights_ties_embed_tokens_and_loads_experts():
    """Validate custom load_weights ties embeddings and loads fused expert weights."""
    config = _build_test_config()
    model = _DummyModel(config)
    weight_items = []

    embed_weight = torch.full(
        model.model.embed_tokens.weight.shape,
        5.0,
    )
    weight_items.append(("model.embed_tokens.weight", embed_weight))
    weight_items.append(("lm_head.weight", torch.full_like(model.lm_head.weight, 7.0)))

    for expert_idx in range(config.num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"block.experts.{expert_idx}.{proj}.weight"
            weight_items.append((key, _make_expert_weight(config, expert_idx, proj)))

    loaded_blocks = load_qwen3_moe_weights(model, weight_items, tied_weight_keys=["lm_head.weight"])
    assert loaded_blocks == 1
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    torch.testing.assert_close(model.block.experts.down_proj[2], _make_expert_weight(config, 2, "down_proj"))


def test_model_loader_uses_custom_load_weights():
    """Ensure ModelCreator dispatches to custom load_weights when available."""
    config = _build_test_config()
    config.model_type = "qwen3_moe"
    pp_info = PPInfo(pp_rank=0, pp_size=1)

    class _DummyCausalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Identity()
            self.load_weights_called = False

        def named_parameters(self, *args, **kwargs):
            return iter([])

        def named_buffers(self, *args, **kwargs):
            return iter([])

        def load_weights(self, weight_iterators, tied_weight_keys=None):
            self.load_weights_called = True
            return 0

    with (
        patch("vexact.inferencer.model_loader._load_state_dict", return_value=[]),
        patch("vexact.inferencer.model_loader.AutoModelForCausalLM.from_config", return_value=_DummyCausalModel()),
        patch("vexact.inferencer.model_loader.TorchMemorySaverAdapter.get_instance") as memory_saver,
    ):
        memory_saver.return_value.region.return_value.__enter__.return_value = None
        memory_saver.return_value.region.return_value.__exit__.return_value = None
        loader = ModelCreator(config, model_path="unused", device=torch.device("cpu"), pp_info=pp_info)
        loader.create_model()
        assert loader._causal_model.load_weights_called
