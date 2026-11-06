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

"""Fused QKV / gate-up projections must give the bits of the separate ones and follow weight syncs."""

import copy

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3MoeConfig

from vexact.batch_invariant_ops import set_batch_invariant_mode
from vexact.models.fused_projections import apply_fused_projection_patches, fuse_projections


if not torch.cuda.is_available():
    pytest.skip("Batch-invariant matmul requires CUDA", allow_module_level=True)


def _config(kind: str, bias: bool):
    common = dict(
        hidden_size=512,
        intermediate_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        num_hidden_layers=2,
        vocab_size=1024,
        attention_bias=bias,
        attn_implementation="eager",
    )
    if kind == "qwen3":
        return Qwen3Config(**common)
    # Dense MLP in every layer: exercises Qwen3MoeMLP and Qwen3MoeAttention without the expert kernels.
    return Qwen3MoeConfig(
        **common, mlp_only_layers=[0, 1], num_experts=4, num_experts_per_tok=2, moe_intermediate_size=256
    )


def _models(kind: str, bias: bool):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(_config(kind, bias)).to("cuda", torch.bfloat16).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)
    fused = copy.deepcopy(model)
    assert fuse_projections(fused) == 4  # 2 layers x (attention + MLP)
    apply_fused_projection_patches()
    return model, fused


def _logits(model, ids):
    with torch.no_grad(), set_batch_invariant_mode(True):
        return model(input_ids=ids).logits


@pytest.mark.parametrize("kind", ["qwen3", "qwen3_moe"])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_projections_match_separate(kind, bias):
    model, fused = _models(kind, bias)
    for tokens in (1, 37, 256):
        ids = torch.randint(0, 1024, (1, tokens), device="cuda")
        assert torch.equal(_logits(fused, ids).view(torch.int16), _logits(model, ids).view(torch.int16))


@pytest.mark.parametrize("kind", ["qwen3", "qwen3_moe"])
def test_weight_sync_writes_into_fused_weight(kind):
    model, fused = _models(kind, bias=False)
    new = {name: torch.randn_like(p) * 0.05 for name, p in model.named_parameters() if "_proj" in name}
    fused_params = dict(fused.named_parameters())
    with torch.no_grad():
        for name, value in new.items():  # what load_state_dict / verl weight syncs do
            dict(model.named_parameters())[name].copy_(value)
            fused_params[name].copy_(value)
    attn = fused.model.layers[0].self_attn
    weight = attn._vexact_fused[0]
    assert attn.q_proj.weight.data_ptr() == weight.data_ptr()
    assert torch.equal(weight[: attn.q_proj.out_features], new["model.layers.0.self_attn.q_proj.weight"])
    ids = torch.randint(0, 1024, (1, 64), device="cuda")
    assert torch.equal(_logits(fused, ids).view(torch.int16), _logits(model, ids).view(torch.int16))


def test_grad_enabled_calls_use_separate_projections():
    model, fused = _models("qwen3", bias=False)
    ids = torch.randint(0, 1024, (1, 16), device="cuda")
    with set_batch_invariant_mode(True):
        out = fused(input_ids=ids).logits
        ref = model(input_ids=ids).logits
    assert out.requires_grad
    assert torch.equal(out.detach().view(torch.int16), ref.detach().view(torch.int16))
