import torch

from vexact.batch_invariant_ops.moe_permute import moe_gather, moe_permute
from vexact.utils.device import DEVICE_MAJOR
from vexact.utils.imports import HAS_QUACK, has_quack


if HAS_QUACK:
    from quack import fused_moe


def _is_cuda_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda


def _validate_tensors_on_cuda(*tensors: torch.Tensor) -> None:
    if not all(_is_cuda_tensor(t) for t in tensors):
        raise ValueError("All tensors must be CUDA tensors")


def _validate_bfloat16_tensors(*tensors: torch.Tensor) -> None:
    if not all(t.dtype == torch.bfloat16 for t in tensors):
        raise ValueError("All tensors must be bfloat16 tensors")


def _validate_int_tensors(*tensors: torch.Tensor) -> None:
    if not all(t.dtype in (torch.int32, torch.int64) for t in tensors):
        raise ValueError("All tensors must be int32 or int64 tensors")


def _validate_contiguous_tensors(*tensors: torch.Tensor) -> None:
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("All tensors must be contiguous")


class QuackFusedMoeExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        num_experts: int,
        routing_weights: torch.Tensor,
        expert_index: torch.Tensor,
        hidden_states: torch.Tensor,
        fc1_1_weight: torch.Tensor,
        fc1_2_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
    ):
        if not has_quack():
            raise ImportError("quack is not installed")
        if DEVICE_MAJOR < 9:
            raise RuntimeError("Quack fused MoE requires SM90+ GPUs")

        _validate_tensors_on_cuda(
            routing_weights,
            expert_index,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
        )
        _validate_bfloat16_tensors(routing_weights, hidden_states, fc1_1_weight, fc1_2_weight, fc2_weight)
        _validate_int_tensors(expert_index)

        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        assert routing_weights.shape[0] == expert_index.shape[0] == hidden_states.shape[0]
        assert routing_weights.shape[1] == expert_index.shape[1]
        top_k = routing_weights.shape[1]
        assert fc1_1_weight.shape[0] == fc1_2_weight.shape[0] == fc2_weight.shape[0] == num_experts
        assert fc1_1_weight.shape[1] == fc1_2_weight.shape[1] == fc2_weight.shape[2]
        assert fc1_1_weight.shape[2] == fc1_2_weight.shape[2] == hidden_states.shape[-1]
        assert fc2_weight.shape[1] == hidden_states.shape[-1]

        # TODO: support top_k > 1 for quack fused moe.
        if top_k != 1:
            raise NotImplementedError("top_k > 1 is not supported for quack fused moe")

        (scatter_output, scatter_index, expert_offset) = moe_permute(hidden_states, expert_index, num_experts)
        hidden_states_by_expert = torch.empty_like(scatter_output)
        fc1_1_output = torch.empty(
            (hidden_states_by_expert.shape[0], fc1_1_weight.shape[1]),
            device=hidden_states_by_expert.device,
            dtype=hidden_states_by_expert.dtype,
        )
        fc1_2_output = torch.empty(
            (hidden_states_by_expert.shape[0], fc1_2_weight.shape[1]),
            device=hidden_states_by_expert.device,
            dtype=hidden_states_by_expert.dtype,
        )
        output_by_expert = torch.empty_like(scatter_output)
        expert_offset = expert_offset.to(torch.int32)

        fused_moe.fused_moe_fwd(
            scatter_output,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            expert_offset,
            fc1_1_output,
            fc1_2_output,
            hidden_states_by_expert,
            output_by_expert,
        )

        output = moe_gather(output_by_expert, scatter_index)
        output = output.reshape(hidden_states_shape)
        ctx.save_for_backward(
            scatter_output,
            scatter_index,
            expert_offset,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            hidden_states_by_expert,
        )
        ctx.num_experts = num_experts
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            scatter_output,
            scatter_index,
            expert_offset,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            hidden_states_by_expert,
        ) = ctx.saved_tensors
        num_experts = ctx.num_experts

        grad_output = grad_output.contiguous()
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        grad_output_by_expert = moe_permute(grad_output, scatter_index, num_experts)[0]

        grad_scatter_output = torch.empty_like(scatter_output)
        grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
        grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
        grad_fc2_weight = torch.empty_like(fc2_weight)

        fused_moe.fused_moe_bwd(
            grad_output_by_expert,
            scatter_output,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            expert_offset,
            fc1_1_output,
            fc1_2_output,
            hidden_states_by_expert,
            grad_scatter_output,
            grad_fc1_1_weight,
            grad_fc1_2_weight,
            grad_fc2_weight,
        )

        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)
        grad_hidden_states = grad_hidden_states.reshape(grad_output.shape)

        return (
            None,
            None,
            None,
            grad_hidden_states,
            grad_fc1_1_weight,
            grad_fc1_2_weight,
            grad_fc2_weight,
        )


class FusedMoeExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        num_experts: int,
        routing_weights: torch.Tensor,
        expert_index: torch.Tensor,
        hidden_states: torch.Tensor,
        fc1_1_weight: torch.Tensor,
        fc1_2_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
    ):
        _validate_tensors_on_cuda(
            routing_weights,
            expert_index,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
        )
        _validate_bfloat16_tensors(routing_weights, hidden_states, fc1_1_weight, fc1_2_weight, fc2_weight)
        _validate_int_tensors(expert_index)

        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        assert routing_weights.shape[0] == expert_index.shape[0] == hidden_states.shape[0]
        assert routing_weights.shape[1] == expert_index.shape[1]
        top_k = routing_weights.shape[1]
        assert fc1_1_weight.shape[0] == fc1_2_weight.shape[0] == fc2_weight.shape[0] == num_experts
        assert fc1_1_weight.shape[1] == fc1_2_weight.shape[1] == fc2_weight.shape[2]
        assert fc1_1_weight.shape[2] == fc1_2_weight.shape[2] == hidden_states.shape[-1]
        assert fc2_weight.shape[1] == hidden_states.shape[-1]

        (scatter_output, scatter_index, expert_offset) = moe_permute(hidden_states, expert_index, num_experts)
        hidden_states_by_expert = torch.empty_like(scatter_output)
        fc1_1_output = torch.empty(
            (hidden_states_by_expert.shape[0], fc1_1_weight.shape[1]),
            device=hidden_states_by_expert.device,
            dtype=hidden_states_by_expert.dtype,
        )
        fc1_2_output = torch.empty(
            (hidden_states_by_expert.shape[0], fc1_2_weight.shape[1]),
            device=hidden_states_by_expert.device,
            dtype=hidden_states_by_expert.dtype,
        )
        output_by_expert = torch.empty_like(scatter_output)

        for expert_id in range(num_experts):
            start = expert_offset[expert_id]
            end = expert_offset[expert_id + 1]
            if start == end:
                continue
            tokens = scatter_output[start:end]
            fc1_1 = torch.matmul(tokens, fc1_1_weight[expert_id].T)
            fc1_2 = torch.matmul(tokens, fc1_2_weight[expert_id].T)
            hidden = torch.nn.functional.silu(fc1_1) * fc1_2
            output_by_expert[start:end] = torch.matmul(hidden, fc2_weight[expert_id].T)
            hidden_states_by_expert[start:end] = hidden
            fc1_1_output[start:end] = fc1_1
            fc1_2_output[start:end] = fc1_2

        output = moe_gather(output_by_expert, scatter_index)
        output = output.reshape(hidden_states_shape)

        ctx.save_for_backward(
            routing_weights,
            expert_index,
            hidden_states,
            scatter_output,
            scatter_index,
            expert_offset,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            hidden_states_by_expert,
        )
        ctx.num_experts = num_experts
        ctx.top_k = top_k
        ctx.hidden_states_shape = hidden_states_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            routing_weights,
            expert_index,
            hidden_states,
            scatter_output,
            scatter_index,
            expert_offset,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            hidden_states_by_expert,
        ) = ctx.saved_tensors
        num_experts = ctx.num_experts
        top_k = ctx.top_k

        grad_output = grad_output.contiguous()
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        grad_output_by_expert = moe_permute(grad_output, scatter_index, num_experts)[0]

        grad_scatter_output = torch.zeros_like(scatter_output)
        grad_fc1_1_weight = torch.zeros_like(fc1_1_weight)
        grad_fc1_2_weight = torch.zeros_like(fc1_2_weight)
        grad_fc2_weight = torch.zeros_like(fc2_weight)

        for expert_id in range(num_experts):
            start = expert_offset[expert_id]
            end = expert_offset[expert_id + 1]
            if start == end:
                continue
            tokens = scatter_output[start:end]
            grad_output_tokens = grad_output_by_expert[start:end]
            fc1_1 = fc1_1_output[start:end]
            fc1_2 = fc1_2_output[start:end]
            hidden = hidden_states_by_expert[start:end]

            grad_fc2_weight[expert_id] += torch.matmul(grad_output_tokens.T, hidden)
            grad_hidden = torch.matmul(grad_output_tokens, fc2_weight[expert_id])
            grad_fc1_1 = grad_hidden * fc1_2 * torch.sigmoid(fc1_1) * (1 + fc1_1 * (1 - torch.sigmoid(fc1_1)))
            grad_fc1_2 = grad_hidden * torch.nn.functional.silu(fc1_1)
            grad_fc1_1_weight[expert_id] += torch.matmul(grad_fc1_1.T, tokens)
            grad_fc1_2_weight[expert_id] += torch.matmul(grad_fc1_2.T, tokens)
            grad_scatter_output[start:end] += torch.matmul(grad_fc1_1, fc1_1_weight[expert_id])
            grad_scatter_output[start:end] += torch.matmul(grad_fc1_2, fc1_2_weight[expert_id])

        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)
        grad_hidden_states = grad_hidden_states.reshape(ctx.hidden_states_shape)

        return (
            None,
            None,
            None,
            grad_hidden_states,
            grad_fc1_1_weight,
            grad_fc1_2_weight,
            grad_fc2_weight,
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

    if DEVICE_MAJOR >= 9:
        expert_fn = QuackFusedMoeExpertFunction
    else:
        expert_fn = FusedMoeExpertFunction

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
