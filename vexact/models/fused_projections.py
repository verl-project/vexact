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
"""Projections of one module computed by a single GEMM in no-grad forwards.

Rollout runs ``q_proj``, ``k_proj`` and ``v_proj`` (and ``gate_proj`` and ``up_proj``) over the same input, so
one GEMM over the concatenated weights replaces three (two) and reads the activations once. The
batch-invariant matmul reduces every output column over K in the same order whatever N is, so the fused GEMM
returns exactly the bits of the separate ones, and training, which keeps the separate projections, stays
aligned (tests/test_fused_projections.py).

A model declares what may be fused once, next to its other patches:

    register_fused_attention(Qwen3Attention)   # ("q_proj", "k_proj", "v_proj")
    register_fused_mlp(Qwen3MLP)               # ("gate_proj", "up_proj")

That records the group on the class (``_vexact_fuse_groups``) and routes its forward through the fused path.
``fuse_projections(model)`` then walks the modules of one model, allocates a weight per declared group and
turns the separate weights into views of it, so parameter names and shapes are unchanged and loading and
weight syncs, which copy into each parameter, fill the fused weight in place.

The fused path is used only for a module that was actually fused and only with autograd off; anything else
runs the model's own forward, so a model whose forward changes upstream, or one that is never fused, keeps
working unchanged.
"""

import sys

import torch
import torch.nn.functional as F
from torch import nn


QKV = ("q_proj", "k_proj", "v_proj")
GATE_UP = ("gate_proj", "up_proj")


def _fuse_group(module: nn.Module, names: tuple[str, ...]) -> None:
    linears = [getattr(module, name) for name in names]
    reference = linears[0].weight
    rows = sum(lin.out_features for lin in linears)
    weight = torch.empty((rows, reference.shape[1]), dtype=reference.dtype, device=reference.device)
    bias = None if linears[0].bias is None else torch.empty(rows, dtype=reference.dtype, device=reference.device)
    start = 0
    for lin in linears:
        end = start + lin.out_features
        weight[start:end].copy_(lin.weight.data)
        lin.weight.data = weight[start:end]
        if bias is not None:
            bias[start:end].copy_(lin.bias.data)
            lin.bias.data = bias[start:end]
        start = end
    fused = getattr(module, "_vexact_fused", None)
    if fused is None:
        fused = {}
        module._vexact_fused = fused
    fused[names] = (weight, bias, [lin.out_features for lin in linears])


def fuse_projections(model: nn.Module) -> int:
    """Fuse every group declared by the classes of ``model``'s modules. Call once the parameters are
    materialized and before weights are loaded. Returns the number of groups fused."""
    fused = 0
    for module in model.modules():
        for names in getattr(type(module), "_vexact_fuse_groups", ()):
            _fuse_group(module, names)
            fused += 1
    return fused


def fused_projection(module: nn.Module, names: tuple[str, ...], x: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
    """One GEMM for ``names``, split back per projection, or None when the fused path does not apply."""
    if torch.is_grad_enabled():
        return None
    entry = getattr(module, "_vexact_fused", {}).get(names)
    if entry is None:
        return None
    weight, bias, sizes = entry
    return F.linear(x, weight, bias).split(sizes, dim=-1)


def _make_attention_forward(original):
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        projected = fused_projection(self, QKV, hidden_states)
        if projected is None:
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
        # The remainder follows the HF Qwen3 / Qwen3-MoE attention forward, which both models share.
        hf = sys.modules[type(self).__module__]
        query, key, value = projected
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key.view(hidden_shape)).transpose(1, 2)
        value_states = value.view(hidden_shape).transpose(1, 2)

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
        projected = fused_projection(self, GATE_UP, x)
        if projected is None:
            return original(self, x)
        gate, up = projected
        return self.down_proj(self.act_fn(gate) * up)

    forward._vexact_fused_projections = True
    return forward


def _register(cls: type, names: tuple[str, ...], make_forward) -> None:
    declared = tuple(getattr(cls, "_vexact_fuse_groups", ()))
    if names not in declared:
        cls._vexact_fuse_groups = declared + (names,)
    if not getattr(cls.forward, "_vexact_fused_projections", False):
        cls.forward = make_forward(cls.forward)


def register_fused_attention(cls: type, names: tuple[str, ...] = QKV) -> None:
    """Declare ``names`` fusable on an attention class whose forward matches HF Qwen3's. Idempotent."""
    _register(cls, names, _make_attention_forward)


def register_fused_mlp(cls: type, names: tuple[str, ...] = GATE_UP) -> None:
    """Declare ``names`` fusable on a gated-MLP class whose forward is ``down_proj(act(gate) * up)``.
    Idempotent."""
    _register(cls, names, _make_mlp_forward)
