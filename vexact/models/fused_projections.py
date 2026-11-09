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

    register_fused_qwen3_attention(Qwen3Attention)   # ("q_proj", "k_proj", "v_proj")
    register_fused_gated_mlp(Qwen3MLP)               # ("gate_proj", "up_proj")

That records the group on the class (``_vexact_fuse_groups``) and routes its forward through the fused path.
Each register function reproduces one upstream forward, so it only fits classes whose forward is that one:
``register_fused_gated_mlp`` fits the ``down_proj(act_fn(gate_proj(x)) * up_proj(x))`` body shared by Qwen3,
Qwen3-MoE, Llama, Qwen2, Qwen2-MoE, Mistral, Olmo3, DeepSeek-V3, Qwen3-Next and Gemma3, while
``register_fused_qwen3_attention`` fits only Qwen3 and Qwen3-MoE. Olmo3, Gemma3 and Qwen3-Next also have
q/k norms but apply them differently, so registering them there would change results, not raise. Each
registration therefore checks the class's forward against the body it reproduces and raises if upstream
changed it, rather than silently running a stale copy.
``fuse_projections(model)`` then walks the modules of one model, allocates a weight per declared group and
turns the separate weights into views of it, so parameter names and shapes are unchanged and loading and
weight syncs, which copy into each parameter, fill the fused weight in place.

A fused gate/up GEMM also hands its output to the dense-SwiGLU kernel whole (``swiglu``, the fused-MoE
kernel without the routing weight), which replaces ``act_fn(gate) * up`` -- two strided reads of that
output, a silu and a multiply -- with one pass, for a bf16 MLP whose activation is SiLU. The kernel replays
aten's roundings, so the bits are eager's; any other dtype or activation keeps the eager expression.

The fused path is used only for a module that was actually fused and only with autograd off; anything else
runs the model's own forward, so a model whose forward changes upstream, or one that is never fused, keeps
working unchanged.
"""

import hashlib
import inspect
import re
import sys

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import SiLUActivation

from vexact.batch_invariant_ops.fused_moe_rollout import swiglu


QKV = ("q_proj", "k_proj", "v_proj")
GATE_UP = ("gate_proj", "up_proj")

# What the dense-SwiGLU kernel replays. transformers maps hidden_act "silu" to its own SiLUActivation
# (``nn.functional.silu``, so aten's bits) and "swish" to nn.SiLU; neither is a subclass of the other.
_SILU = (nn.SiLU, SiLUActivation)

# Fingerprints of the transformers forwards the fused paths below reproduce (transformers 5.16.1:
# Qwen3Attention / Qwen3MoeAttention share one body, Qwen3MLP / Qwen3MoeMLP the other). A mismatch means
# upstream changed the forward and the copy here has to be re-checked against it.
_QWEN3_ATTENTION_FORWARD = "55ed6e460ce288d0"
_GATED_MLP_FORWARD = "4cd949b84db8e002"


def _forward_fingerprint(func) -> str:
    source = re.sub(r"#.*", "", inspect.getsource(func))
    return hashlib.sha256(re.sub(r"\s+", " ", source).strip().encode()).hexdigest()[:16]


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


def fused_projection(module: nn.Module, names: tuple[str, ...], x: torch.Tensor, split: bool = True):
    """One GEMM for ``names``, split back per projection (``split=False``: the GEMM output with the
    projections' columns still concatenated), or None when the fused path does not apply."""
    if torch.is_grad_enabled():
        return None
    entry = getattr(module, "_vexact_fused", {}).get(names)
    if entry is None:
        return None
    weight, bias, sizes = entry
    out = F.linear(x, weight, bias)
    return out.split(sizes, dim=-1) if split else out


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
        fc1 = fused_projection(self, GATE_UP, x, split=False)
        if fc1 is None:
            return original(self, x)
        if fc1.dtype == torch.bfloat16 and isinstance(self.act_fn, _SILU):
            # One pass over the fused GEMM output instead of silu, a multiply and two strided reads of it.
            # The kernel replays aten's bf16 roundings, so the bits are eager's.
            return self.down_proj(swiglu(fc1))
        # gate_proj and up_proj both project to intermediate_size in every class registered here.
        gate, up = fc1.split(fc1.shape[-1] // 2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)

    forward._vexact_fused_projections = True
    return forward


def _register(cls: type, names: tuple[str, ...], make_forward, expected: str) -> None:
    if not getattr(cls.forward, "_vexact_fused_projections", False):
        found = _forward_fingerprint(cls.forward)
        if found != expected:
            raise RuntimeError(
                f"{cls.__module__}.{cls.__qualname__}.forward does not match the body "
                f"vexact/models/fused_projections.py reproduces (expected {expected}, found {found}). "
                "Re-check that copy against the current forward before registering this class."
            )
        cls.forward = make_forward(cls.forward)
    declared = tuple(getattr(cls, "_vexact_fuse_groups", ()))
    if names not in declared:
        cls._vexact_fuse_groups = declared + (names,)


def register_fused_qwen3_attention(cls: type, names: tuple[str, ...] = QKV) -> None:
    """Declare ``names`` fusable on a class whose forward is HF Qwen3Attention's (per-head q/k norms over the
    reshaped projections, ``sliding_window`` passed on): Qwen3 and Qwen3-MoE. Idempotent."""
    _register(cls, names, _make_attention_forward, _QWEN3_ATTENTION_FORWARD)


def register_fused_gated_mlp(cls: type, names: tuple[str, ...] = GATE_UP) -> None:
    """Declare ``names`` fusable on a class whose forward is ``down_proj(act_fn(gate_proj(x)) * up_proj(x))``,
    which many HF models share verbatim. Idempotent."""
    _register(cls, names, _make_mlp_forward, _GATED_MLP_FORWARD)
