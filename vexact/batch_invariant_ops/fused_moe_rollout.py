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
"""Inference-only fused-MoE forward that is bitwise identical to VeOmni's Triton fused MoE.

VeOmni's forward (``MergedFc1TritonFusedMoeExpertFunction``) runs, per MoE layer:
``moe_scatter`` -> fc1 grouped GEMM -> ``silu(gate)`` -> ``* up`` -> scatter of the gate weights ->
``* gate_weight`` -> fc2 grouped GEMM -> ``moe_gather``. This module produces the same bits with
less memory traffic:

- the token gather is fused into fc1's loads (triton_kernels ``matmul_ogs`` with ``GatherIndx``)
  for prefill-size calls, so ``moe_scatter`` disappears there (a pure load cannot change bits);
- ``silu``, ``* up``, the gate-weight scatter and ``* gate_weight`` become one Triton pass that
  replays VeOmni's three bf16 roundings with IEEE ``expf`` / division (exact for every finite bf16
  gate value, see tests/batch_invariant_ops/test_fused_moe_rollout.py);
- fc1 / fc2 use the grouped-GEMM dispatch from ``moe_group_gemm_tuning`` (tuned VeOmni launch or
  ``matmul_ogs``), which already reproduces VeOmni's bits.

It is used only when autograd is off (rollout, and no-grad log-prob passes); anything that needs
gradients, expert parallelism, split fc1 weights or a SwiGLU clamp goes to VeOmni's function.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _swiglu_gate_kernel(
    fc1_ptr,  # [rows, 2 * INTER] bf16, expert-sorted rows: [gate | up]
    gate_w_ptr,  # [T * topk] bf16 routing weights, flat (token, slot) order
    order_ptr,  # [rows] flat (token, slot) index of each expert-sorted row
    out_ptr,  # [rows, INTER] bf16
    INTER: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < INTER
    gate = tl.load(fc1_ptr + row * 2 * INTER + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(fc1_ptr + row * 2 * INTER + INTER + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(gate_w_ptr + tl.load(order_ptr + row)).to(tl.float32)
    # aten::silu on bf16: fp32 x / (1 + expf(-x)), rounded to bf16.
    act = tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
    # bf16 * bf16 in aten: fp32 product rounded to bf16, twice.
    act = (act * up).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * INTER + cols, (act * w).to(tl.bfloat16), mask=mask)


def swiglu_gate(fc1_output: torch.Tensor, gate_weights_flat: torch.Tensor, sorted_order: torch.Tensor):
    """``(silu(fc1[:, :I]) * fc1[:, I:]) * gate_weights_flat[sorted_order][:, None]`` with VeOmni's
    exact bf16 rounding sequence, in one pass."""
    rows, two_i = fc1_output.shape
    inter = two_i // 2
    out = torch.empty((rows, inter), dtype=fc1_output.dtype, device=fc1_output.device)
    if rows == 0:
        return out
    block_i = min(1024, triton.next_power_of_2(inter))
    _swiglu_gate_kernel[(rows, triton.cdiv(inter, block_i))](
        fc1_output, gate_weights_flat, sorted_order, out, INTER=inter, BLOCK_I=block_i, num_warps=4
    )
    return out


def _wrap_fused_moe_forward(veomni_forward):
    """Wrap VeOmni's bound ``_fused_moe_forward`` with the inference-only path above."""
    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.ops.kernels.moe import group_gemm as fused_moe_module
    from veomni.ops.kernels.moe._kernels.kernel.moe import expert_histogram, moe_gather, moe_scatter
    from veomni.ops.kernels.moe._scatter import compute_expert_scatter_index

    from .moe_group_gemm_tuning import _OGS_MIN_ROWS, _load_matmul_ogs

    matmul_ogs = _load_matmul_ogs()

    def forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        fc1_1_2_weight=None,
        swiglu_limit=None,
    ):
        if (
            torch.is_grad_enabled()
            or fc1_1_2_weight is None
            or fc1_1_weight is not None
            or fc1_2_weight is not None
            or swiglu_limit is not None
            or hidden_states.dtype != torch.bfloat16
            or routing_weights.dtype != torch.bfloat16
            or get_parallel_state().ep_enabled
        ):
            return veomni_forward(
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_weight,
                fc1_2_weight,
                fc2_weight,
                fc1_1_2_weight,
                swiglu_limit=swiglu_limit,
            )

        group_gemm = fused_moe_module.group_gemm_same_nk  # tuned / matmul_ogs dispatch, VeOmni-identical
        topk = selected_experts.shape[-1]
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        splits = expert_histogram(selected_experts, num_experts)
        sorted_order, scatter_index = compute_expert_scatter_index(selected_experts)
        cumsum = torch.cumsum(splits, dim=0)
        rows = sorted_order.shape[0]

        if rows > _OGS_MIN_ROWS:
            gather = (sorted_order.to(torch.int32), scatter_index.reshape(-1).to(torch.int32))
            fc1 = matmul_ogs(x, fc1_1_2_weight.transpose(1, 2), splits.to(torch.int32), topk, gather)
        else:
            fc1 = group_gemm(
                a=moe_scatter(x, scatter_index),
                b=fc1_1_2_weight,
                cumsum_M=cumsum,
                max_M=rows,
                transpose_a=False,
                transpose_b=True,
            )
        act = swiglu_gate(fc1, routing_weights.reshape(-1), sorted_order)
        fc2 = group_gemm(a=act, b=fc2_weight, cumsum_M=cumsum, max_M=rows, transpose_a=False, transpose_b=True)
        return moe_gather(fc2, scatter_index).reshape(hidden_states.shape)

    forward._vexact_rollout_fused = True
    return forward


def apply_rollout_fused_moe() -> None:
    """Wrap VeOmni's currently bound fused-MoE forward, which must be the Triton one. Idempotent."""
    import veomni.ops.kernels.moe as moe_ops
    from veomni.ops.kernels.moe.group_gemm import group_gemm_fused_moe_forward

    current = moe_ops._fused_moe_forward
    if getattr(current, "_vexact_rollout_fused", False):
        return
    if current is not group_gemm_fused_moe_forward:
        # Only the Triton grouped-GEMM forward is reproduced bit for bit.
        raise RuntimeError(f"expected VeOmni's Triton fused-MoE forward to be bound, got {current!r}")
    moe_ops._fused_moe_forward = _wrap_fused_moe_forward(current)
