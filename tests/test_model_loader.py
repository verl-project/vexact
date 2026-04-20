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

import pytest
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from vexact.config import PPInfo
from vexact.inferencer.model_loader import (
    ModelCreator,
    PPMissingLayer,
    TransformersForCausalLM,
    load_weights_from_weight_iterator,
)


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda")


@pytest.fixture(scope="module")
def model_path() -> str:
    return os.environ["VEXACT_TESTS_MODEL_PATH"]


@pytest.fixture(scope="module")
def model_config(model_path, device) -> PretrainedConfig:
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": device},
        "trust_remote_code": True,
    }
    config = AutoConfig.from_pretrained(model_path, **model_kwargs)
    config.num_hidden_layers = 3
    return config


@pytest.fixture(scope="module")
def original_model_named_params(model_config):
    causal_model = AutoModelForCausalLM.from_config(model_config).to("cuda")
    return dict(causal_model.model.named_parameters())


@pytest.fixture
def input_ids_list() -> list[int]:
    return [151644, 872, 198, 785, 2310, 326, 57909, 53416, 11]


@pytest.fixture
def batch_input_ids(input_ids_list, device) -> torch.Tensor:
    return torch.tensor(input_ids_list, device=device, dtype=torch.long).unsqueeze(0)


@pytest.fixture
def batch_position_ids(input_ids_list, device) -> torch.Tensor:
    return torch.arange(0, len(input_ids_list), device=device, dtype=torch.long).unsqueeze(0)


@pytest.fixture
def batch_attention_mask(batch_input_ids) -> torch.Tensor:
    return torch.ones_like(batch_input_ids)


def test_create_model_first_rank(
    model_config,
    model_path,
    original_model_named_params,
    device,
):
    model_creator = ModelCreator(model_config, model_path, device, PPInfo(3, 0))
    causal_model = model_creator.create_model()
    assert isinstance(causal_model.lm_head, PPMissingLayer)
    model = causal_model.model
    named_params = dict(model.named_parameters())
    expected_layers = set(
        [
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.k_norm.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
        ]
    )
    assert set(named_params.keys()) == expected_layers
    for name in expected_layers:
        torch.equal(original_model_named_params[name], named_params[name])


def test_create_model_mid_rank(model_config, model_path, original_model_named_params, device):
    model_creator = ModelCreator(model_config, model_path, device, PPInfo(3, 1))
    causal_model = model_creator.create_model()
    assert isinstance(causal_model.lm_head, PPMissingLayer)
    model = causal_model.model
    named_params = dict(model.named_parameters())
    expected_layers = set(
        [
            "layers.1.self_attn.q_proj.weight",
            "layers.1.self_attn.k_proj.weight",
            "layers.1.self_attn.v_proj.weight",
            "layers.1.self_attn.o_proj.weight",
            "layers.1.self_attn.q_norm.weight",
            "layers.1.self_attn.k_norm.weight",
            "layers.1.mlp.gate_proj.weight",
            "layers.1.mlp.up_proj.weight",
            "layers.1.mlp.down_proj.weight",
            "layers.1.input_layernorm.weight",
            "layers.1.post_attention_layernorm.weight",
        ]
    )
    assert set(named_params.keys()) == expected_layers
    for name in expected_layers:
        torch.equal(original_model_named_params[name], named_params[name])


def test_create_model_last_rank(model_config, model_path, original_model_named_params, device):
    model_creator = ModelCreator(model_config, model_path, device, PPInfo(3, 2))
    causal_model = model_creator.create_model()
    assert isinstance(causal_model.lm_head, nn.Linear)
    model = causal_model.model
    named_params = dict(model.named_parameters())
    expected_layers = set(
        [
            "embed_tokens.weight",
            "norm.weight",
            "layers.2.self_attn.q_proj.weight",
            "layers.2.self_attn.k_proj.weight",
            "layers.2.self_attn.v_proj.weight",
            "layers.2.self_attn.o_proj.weight",
            "layers.2.self_attn.q_norm.weight",
            "layers.2.self_attn.k_norm.weight",
            "layers.2.mlp.gate_proj.weight",
            "layers.2.mlp.up_proj.weight",
            "layers.2.mlp.down_proj.weight",
            "layers.2.input_layernorm.weight",
            "layers.2.post_attention_layernorm.weight",
        ]
    )
    assert set(named_params.keys()) == expected_layers
    for name in expected_layers:
        torch.equal(original_model_named_params[name], named_params[name])


def test_pp_wrapper_load_weights_accepts_model_prefix_keys():
    class DummyBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(4, 3)
            self.proj = nn.Linear(3, 3, bias=False)

    base_model = DummyBaseModel()
    lm_head = nn.Linear(3, 4, bias=False)
    wrapper = TransformersForCausalLM(base_model, lm_head, PPInfo(2, 0))
    config = PretrainedConfig(tie_word_embeddings=False)

    new_embed = torch.randn_like(base_model.embed_tokens.weight)
    new_proj = torch.randn_like(base_model.proj.weight)
    new_lm_head = torch.randn_like(lm_head.weight)

    load_weights_from_weight_iterator(
        wrapper,
        config,
        [
            ("model.embed_tokens.weight", new_embed),
            ("model.proj.weight", new_proj),
            ("lm_head.weight", new_lm_head),
        ],
    )

    assert torch.equal(base_model.embed_tokens.weight, new_embed)
    assert torch.equal(base_model.proj.weight, new_proj)
    assert torch.equal(lm_head.weight, new_lm_head)
