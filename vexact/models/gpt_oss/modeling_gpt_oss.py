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

"""Patch HuggingFace GPT-OSS experts to use VeOmni's fused quack kernel."""

import logging

import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts


logger = logging.getLogger(__name__)
_PATCHED = False


def _run_fused_quack(
    *,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    # Import lazily: register_models() also runs in CPU-only verl workers, while
    # the quack kernel itself is only available on SM90+ rollout workers.
    from veomni.ops.kernels.moe.quack_gemm_interleave_gate_up import (
        quack_gemm_gpt_oss_fused_moe_forward,
    )

    return quack_gemm_gpt_oss_fused_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        gate_up_proj=gate_up_proj,
        gate_up_proj_bias=gate_up_proj_bias,
        down_proj=down_proj,
        down_proj_bias=down_proj_bias,
        alpha=alpha,
        limit=limit,
    )


def gpt_oss_experts_forward(
    self: GptOssExperts,
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor | None = None,
    routing_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if router_indices is None or routing_weights is None:
        raise ValueError("GPT-OSS fused_quack requires router_indices and routing_weights")

    return _run_fused_quack(
        num_experts=self.num_experts,
        routing_weights=routing_weights,
        selected_experts=router_indices,
        hidden_states=hidden_states,
        gate_up_proj=self.gate_up_proj,
        gate_up_proj_bias=self.gate_up_proj_bias,
        down_proj=self.down_proj,
        down_proj_bias=self.down_proj_bias,
        alpha=self.alpha,
        limit=self.limit,
    )


def apply_gpt_oss_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    GptOssExperts.forward = gpt_oss_experts_forward
    logger.info("Applied GPT-OSS fused_quack monkey patch.")
    _PATCHED = True


__all__ = ["apply_gpt_oss_patches", "gpt_oss_experts_forward"]
