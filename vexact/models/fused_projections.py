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
"""Fused QKV and gate/up projections for no-grad HF Qwen3 / Qwen3-MoE forwards.

Rollout computes q, k and v (and gate and up) with one GEMM over the concatenated weights instead of three
(two). The batch-invariant matmul reduces every output column over K in the same order whatever N is, so the
fused GEMM returns exactly the bits of the separate ones, and training, which keeps separate projections, stays
aligned (tests/test_fused_projections.py).

``fuse_projections`` allocates one weight per attention / MLP module and turns the separate weights into views of
it. Loading and weight syncs copy into each parameter, so they fill the fused weight in place and no second copy
exists. Grad-enabled calls and modules that were not fused use the original forwards.
"""

import sys

import torch
import torch.nn.functional as F


_QKV = ("q_proj", "k_proj", "v_proj")
_GATE_UP = ("gate_proj", "up_proj")


def _attention_classes():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

    return Qwen3Attention, Qwen3MoeAttention


def _mlp_classes():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

    return Qwen3MLP, Qwen3MoeMLP


def _fuse(module: torch.nn.Module, names: tuple[str, ...]) -> None:
    linears = [getattr(module, name) for name in names]
    w0 = linears[0].weight
    weight = torch.empty((sum(lin.out_features for lin in linears), w0.shape[1]), dtype=w0.dtype, device=w0.device)
    has_bias = linears[0].bias is not None
    bias = torch.empty(weight.shape[0], dtype=w0.dtype, device=w0.device) if has_bias else None
    start = 0
    for lin in linears:
        end = start + lin.out_features
        weight[start:end].copy_(lin.weight.data)
        lin.weight.data = weight[start:end]
        if has_bias:
            bias[start:end].copy_(lin.bias.data)
            lin.bias.data = bias[start:end]
        start = end
    module._vexact_fused = (weight, bias, [lin.out_features for lin in linears])


def fuse_projections(model: torch.nn.Module) -> int:
    """Fuse the q/k/v and gate/up weights of every HF Qwen3 / Qwen3-MoE attention and dense MLP in ``model``.
    Call after the parameters are materialized and before weights are loaded. Returns the number of fused modules."""
    attention, mlp = _attention_classes(), _mlp_classes()
    fused = 0
    for module in model.modules():
        if isinstance(module, attention):
            _fuse(module, _QKV)
            fused += 1
        elif isinstance(module, mlp):
            _fuse(module, _GATE_UP)
            fused += 1
    return fused


def _make_attention_forward(original):
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        fused = getattr(self, "_vexact_fused", None)
        if fused is None or torch.is_grad_enabled():
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
        # Same as the HF Qwen3 / Qwen3-MoE forward, except for the single q/k/v GEMM.
        hf = sys.modules[type(self).__module__]
        weight, bias, sizes = fused
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q, k, v = F.linear(hidden_states, weight, bias).split(sizes, dim=-1)
        query_states = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(k.view(hidden_shape)).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = hf.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = hf.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, hf.eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    forward._vexact_fused_projections = True
    return forward


def _make_mlp_forward(original):
    def forward(self, x):
        fused = getattr(self, "_vexact_fused", None)
        if fused is None or torch.is_grad_enabled():
            return original(self, x)
        weight, bias, sizes = fused
        gate, up = F.linear(x, weight, bias).split(sizes, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)

    forward._vexact_fused_projections = True
    return forward


def apply_fused_projection_patches() -> None:
    """Route no-grad forwards of fused Qwen3 / Qwen3-MoE attention and dense MLP modules through one GEMM.
    Idempotent."""
    for cls in (*_attention_classes(), *_mlp_classes()):
        if getattr(cls.forward, "_vexact_fused_projections", False):
            continue
        make = _make_attention_forward if cls in _attention_classes() else _make_mlp_forward
        cls.forward = make(cls.forward)
