# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import torch
from quack.gemm_interface import gemm

from .group_gemm.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk
from .group_gemm.kernel.moe import expert_histogram, moe_gather, moe_scatter


class FusedMoeExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    ):
        # MOE Step 3: dispatch input tokens to the experts
        # result shape is (batch_size * sequence_len * topk, hidden_size)
        # MOE Step 3-1: compute the token num for each expert
        # splits shape (num_experts)
        splits = expert_histogram(expert_index, num_experts)

        # MOE Step 3-2: compute the each token's index in result
        # scatter_index shape (batch_size * sequence_len, topk)
        # TODO(wenyawei): opt it
        scatter_index = expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)

        # MOE Step 3-3: compute the result, select tokens by scatter_index, and put them together
        # scatter_output shape (batch_size * sequence_len * topk, hidden_size)
        scatter_output = moe_scatter(hidden_states, scatter_index)

        # MOE Step 4: compute linear layer 1-1
        # Not consistent.
        cumsum_t = torch.cumsum(splits, dim=0)
        fc1_1_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_1_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 6: compute linear layer 1-2
        # fc1_2_output shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_2_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 5: compute the actication of linear layer 1-1
        # TODO(wenyawei): act function
        # fc1_1_activation shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # MOE Step 7: compute final result of linear layer 1
        fc1_activation = fc1_1_activation * fc1_2_output

        # MOE Step 8: compute the the weighted linear layer 1 result
        # MOE Step 8-1: compute scattered_gate_weight, shape is (batch_size * sequence_len * topk)
        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        # MOE Step 8-2: multiply activate with scattered_gate_weight
        # fc1_weighted_output shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_weighted_output = fc1_activation * scattered_gate_weight

        # MOE Step 9: compute linear layer 2
        # result shape is (batch_size * sequence_len * topk, hidden_size)
        fc2_output = group_gemm_same_nk(
            a=fc1_weighted_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 10: gather the final token result by averate the the topk token results
        expert_output = moe_gather(fc2_output, scatter_index)

        # reshape the output with input shape
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.save_for_backward(
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        ) = ctx.saved_tensors
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        # MOE Step 10
        grad_fc2_output = moe_scatter(grad_output, scatter_index)

        # MOE Step 9
        # grad_fc1_weighted_output = torch.empty_like(fc1_weighted_output)

        # dgrad
        grad_fc1_weighted_output = group_gemm_same_nk(
            a=grad_fc2_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_fc2_output,
                b=fc1_weighted_output,
                c=grad_fc2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 8
        # MOE Step 8-2
        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight

        # MOE Step 8-1
        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        # recompute during backward
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # MOE Step 7
        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation

        # MOE Step 6
        # grad_scatter_output_2 = torch.empty_like(scatter_output)

        # dgrad
        grad_scatter_output_2 = group_gemm_same_nk(
            a=grad_fc1_2_output,
            b=fc1_2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_2_output,
                b=scatter_output,
                c=grad_fc1_2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 5
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        # MOE Step 4
        # grad_scatter_output_1 = torch.empty_like(scatter_output)

        # dgrad
        grad_scatter_output_1 = group_gemm_same_nk(
            a=grad_fc1_1_output,
            b=fc1_1_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
            group_gemm_same_mn(
                a=grad_fc1_1_output,
                b=scatter_output,
                c=grad_fc1_1_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 3
        # MOE Step 3-3
        grad_scatter_output = grad_scatter_output_1 + grad_scatter_output_2
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)

        # MOE Step 3-2: no grad
        # MOE Step 3-1: no grad

        # reshape the result with input shape
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_fc1_1_weight,  # fc1_1_weight
            grad_fc1_2_weight,  # fc1_2_weight
            grad_fc2_weight,  # fc2_weight
        )


def _build_moe_indices(expert_index: torch.Tensor, num_experts: int):
    """Build cu_seqlens_m, A_idx, and scatter_index from expert routing.

    Args:
        expert_index: [T, topk] expert assignments.
        num_experts: total number of experts.

    Returns:
        cu_seqlens_m: [E+1] cumulative token counts per expert (int32).
        A_idx: [T*topk] token indices sorted by expert assignment (int32).
        scatter_index: [T, topk] indices for moe_gather/moe_scatter (int32).
    """
    topk = expert_index.shape[1]
    flat = expert_index.flatten()
    sorted_order = flat.argsort(stable=True)
    scatter_index = sorted_order.argsort().int().view(expert_index.shape)
    # A_idx maps expert-sorted positions to original token indices (0..T-1).
    # sorted_order values are flat indices (t*topk + k), so integer-divide by topk.
    A_idx = (sorted_order // topk).int()

    splits = expert_histogram(expert_index, num_experts)
    cu_seqlens_m = torch.zeros(num_experts + 1, dtype=torch.int32, device=expert_index.device)
    cu_seqlens_m[1:] = torch.cumsum(splits, dim=0).int()

    return cu_seqlens_m, A_idx, scatter_index


class QuackFusedMoeExpertFunction(torch.autograd.Function):
    """Fused MoE with split fc1 weights using quack GEMM."""

    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    ):
        cu_seqlens_m, A_idx, scatter_index = _build_moe_indices(expert_index, num_experts)

        # Transpose weights for forward: [E, N, K] -> [E, K, N] (view, no copy).
        # quack internally transposes back before calling CUTLASS kernels,
        # so no .contiguous() call is needed here.
        fc1_1_w_t = fc1_1_weight.transpose(1, 2)
        fc1_2_w_t = fc1_2_weight.transpose(1, 2)
        fc2_w_t = fc2_weight.transpose(1, 2)

        # fc1_1: [T*topk, I] (expert-sorted via A_idx)
        fc1_1_output = gemm(hidden_states, fc1_1_w_t, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx)

        # fc1_2: [T*topk, I]
        fc1_2_output = gemm(hidden_states, fc1_2_w_t, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx)

        # SiLU activation + gate multiply
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_activation = fc1_1_activation * fc1_2_output

        # Apply routing weights.
        # Note: A_idx alone cannot replace scatter_index here because A_idx only
        # carries token indices (sorted_order // topk) but not the topk-slot index,
        # so it cannot address into gate_weights[T, topk] without extra bookkeeping.
        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        fc1_weighted_output = fc1_activation * scattered_gate_weight

        # fc2: input is already expert-sorted, no A_idx needed
        fc2_output = gemm(fc1_weighted_output, fc2_w_t, cu_seqlens_m=cu_seqlens_m)

        # Gather output tokens back to original order
        expert_output = moe_gather(fc2_output, scatter_index)
        del fc2_output
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.save_for_backward(
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            cu_seqlens_m,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            cu_seqlens_m,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        ) = ctx.saved_tensors
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        # Step 10: scatter grad to expert-sorted order
        grad_fc2_output = moe_scatter(grad_output, scatter_index)

        # Step 9 dgrad: grad @ fc2_weight (original layout [E, H, I] is already [K, N] for quack)
        grad_fc1_weighted_output = gemm(grad_fc2_output, fc2_weight, cu_seqlens_m=cu_seqlens_m)

        # Step 9 wgrad: grad_fc2_output.T @ fc1_weighted_output → [E, H, I]
        # cu_seqlens_k mode: A=[M, total_K] @ B=[total_K, N] → [L, M, N] per expert group.
        # Pass .T view (not .T.contiguous()) — quack varlen_k requires A to be m-major.
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = gemm(grad_fc2_output.T, fc1_weighted_output, cu_seqlens_k=cu_seqlens_m)
        del fc1_weighted_output

        # Step 8-2: routing weight backward
        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight
        del scattered_gate_weight

        # Step 8-1: gate weight backward
        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)
        del fc1_activation
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        del grad_scattered_gate_weight
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        # Recompute SiLU activation
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # Step 7
        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        del fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation
        del grad_fc1_activation, fc1_1_activation

        # Step 6 dgrad: fc1_2_weight [E, I, H] is already [K, N] for quack
        grad_scatter_output_2 = gemm(grad_fc1_2_output, fc1_2_weight, cu_seqlens_m=cu_seqlens_m)

        # Step 5: SiLU backward
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        del fc1_1_output

        # Step 4 dgrad: fc1_1_weight [E, I, H] is already [K, N] for quack
        grad_scatter_output_1 = gemm(grad_fc1_1_output, fc1_1_weight, cu_seqlens_m=cu_seqlens_m)

        # Recompute scatter_output for wgrad
        scatter_output = moe_scatter(hidden_states, scatter_index)

        # Step 6 wgrad: grad_fc1_2_output.T @ scatter_output → [E, I, H]
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = gemm(grad_fc1_2_output.T, scatter_output, cu_seqlens_k=cu_seqlens_m)
        del grad_fc1_2_output

        # Step 4 wgrad: grad_fc1_1_output.T @ scatter_output → [E, I, H]
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = gemm(grad_fc1_1_output.T, scatter_output, cu_seqlens_k=cu_seqlens_m)
        del grad_fc1_1_output, scatter_output

        # Step 3: gather gradients back to original token order
        grad_scatter_output = grad_scatter_output_1 + grad_scatter_output_2
        del grad_scatter_output_1, grad_scatter_output_2
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_fc1_1_weight,  # fc1_1_weight
            grad_fc1_2_weight,  # fc1_2_weight
            grad_fc2_weight,  # fc2_weight
        )


def fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
):
    routing_weights = routing_weights.bfloat16()
    hidden_states = hidden_states.bfloat16()
    from vexact.utils.device import DEVICE_MAJOR

    if DEVICE_MAJOR == 9:
        expert_fn = FusedMoeExpertFunction
    elif DEVICE_MAJOR == 10:
        expert_fn = QuackFusedMoeExpertFunction
    else:
        raise NotImplementedError(f"Unsupported device major version: {DEVICE_MAJOR}")

    final_hidden_states = expert_fn.apply(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    )

    return final_hidden_states
