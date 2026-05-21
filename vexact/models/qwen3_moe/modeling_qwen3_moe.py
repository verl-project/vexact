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

"""
Monkey patch HuggingFace Transformers Qwen3-MoE model to support high-performance fused MoE kernels.
"""

import logging
from types import SimpleNamespace
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3MoeForSequenceClassification,
    Qwen3MoeForTokenClassification,
    Qwen3MoePreTrainedModel,
    Qwen3MoeSparseMoeBlock,
)


logger = logging.getLogger(__name__)

### Patching HuggingFace Qwen3Moe modeling


class Qwen3MoeExperts(nn.Module):
    """
    Fused experts container that stores all expert weights in stacked tensors.

    Storage layout matches VeOmni's v5-patched ``Qwen3MoeExperts`` (fused
    ``gate_up_proj`` of shape ``(num_experts, 2*intermediate, hidden)`` plus
    separate ``down_proj``). This keeps the rollout-side parameter names
    identical to the actor-side names so that verl's bucketed FSDP→rollout
    weight sync can copy tensors directly without an unfuse step.
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_dim),
            requires_grad=True,
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_size),
            requires_grad=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_idx: Optional[int] = None,
        routing_weights: Optional[torch.Tensor] = None,
        selected_experts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if expert_idx is not None:
            gate_up = torch.matmul(hidden_states, self.gate_up_proj[expert_idx].transpose(0, 1))
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = self.act_fn(gate) * up
            return torch.matmul(hidden, self.down_proj[expert_idx].transpose(0, 1))

        assert routing_weights is not None and selected_experts is not None, (
            "routing_weights and selected_experts must be provided when expert_idx is None"
        )

        # Use VeOmni's fused MoE kernel directly (fc1_1_2_weight path), so the
        # rollout side and VeOmni's actor side compute experts with identical
        # arithmetic — required for `rollout_probs_diff_max == 0`.
        from veomni.ops import fused_moe_forward

        return fused_moe_forward(
            num_experts=self.num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            fc1_1_weight=None,
            fc1_2_weight=None,
            fc2_weight=self.down_proj,
            fc1_1_2_weight=self.gate_up_proj,
        )


def moe_block_init(self: Qwen3MoeSparseMoeBlock, config: Qwen3MoeConfig):
    super(Qwen3MoeSparseMoeBlock, self).__init__()
    self.num_experts = config.num_experts
    self.top_k = config.num_experts_per_tok
    self.norm_topk_prob = config.norm_topk_prob
    self._fused_config = SimpleNamespace(
        num_experts=config.num_experts,
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        hidden_act=config.hidden_act,
    )

    self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    self.experts = Qwen3MoeExperts(config)


def moe_block_forward(self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor):
    if not isinstance(self.experts, Qwen3MoeExperts):
        raise AssertionError(
            "Qwen3-MoE experts are not fused. Ensure apply_qwen3_moe_patches() is called before loading the model."
        )

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = self.experts(
        hidden_states,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
    )
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    # transformers v5: Qwen3MoeSparseMoeBlock.forward returns only hidden_states
    # (router_logits collected via OutputRecorder hook on Qwen3MoeTopKRouter,
    # which our plain-Linear gate bypasses; rollout doesn't need aux-loss stats).
    return final_hidden_states


### Model patching ends

### Utilities for loading fused experts weights from separate tensors in the checkpoint
_PATCHED = False


# TODO: this function currently does not work
# the fused experts are not initialized at all now
# when we use from_pretrained to load the model
### These should be removed after we upgrade to HF Transformers v5.
@classmethod
def _patched_load_pretrained_model(
    cls,
    model,
    state_dict,
    checkpoint_files,
    *model_args,
    **model_kwargs,
):
    """
    The function we will use to override the original model loading method
    We leverage the original model loading method first then modify the experts module

    :param cls: the Qwen3Moe model class
    :param model: original model module
    :param state_dict: original model module
    :param checkpoint_files: safetensors to read
    :param model_args: huggingface model args
    :param model_kwargs: huggingface model kwargs
    """

    raise NotImplementedError(
        "When using VeXact Qwen3Moe, please use from_config to initialize the model",
        "then use VeXact model loader to load weights",
    )


### custom weight loading method for vexact weight loaders
### we expect models are created first by from_config first
### then weights are loaded by load_weights,
### we will override the load_weights method to fuse experts weights
### These should be removed after we upgrade to HF Transformers v5.
_EXPERT_PROJS = {"gate_proj", "up_proj", "down_proj", "gate_up_proj"}


def load_qwen3_moe_weights(
    self: nn.Module,
    weight_iterator: Iterable[tuple[str, torch.Tensor]],
    tied_weight_keys: Optional[list[str]] = None,
) -> int:
    """
    custom weight loading function to handle fused experts weights stored in separate tensors in the checkpoint.

    :param self: A patched Qwen3-MoE model instance with Qwen3MoeExperts modules
    :type self: nn.Module
    :param weight_iterator: Description
    :type weight_iterator: Iterable[tuple[str, torch.Tensor]]
    :param tied_weight_keys: Description
    :type tied_weight_keys: list[str]
    :return: Description
    :rtype: int
    """

    full_param_dict = dict(self.named_parameters())
    direct_loaded_blocks: set[str] = set()
    embed_tokens_weight = None
    # extra name and weight tensor from the weight iterator
    for full_name, loaded_weight in weight_iterator:
        if ".experts." in full_name and full_name.endswith(".weight"):
            # Disk checkpoints (HF) store experts as per-expert separate keys:
            #   <prefix>.experts.<expert_idx>.<proj>.weight
            # Example:
            #   model.layers.0.mlp.experts.3.gate_proj.weight
            # Fuse them into the rollout-side ``gate_up_proj``/``down_proj``
            # stacked tensors at the right per-expert slot.
            prefix, rest = full_name.split(".experts.", 1)
            try:
                expert_idx_str, proj, suffix = rest.split(".")
            except ValueError:
                # Not a 3-part suffix, skip.
                expert_idx_str, proj, suffix = None, None, None
            if suffix == "weight" and proj in _EXPERT_PROJS and expert_idx_str is not None and expert_idx_str.isdigit():
                expert_idx = int(expert_idx_str)
                try:
                    block = self.get_submodule(prefix)
                except AttributeError:
                    block = None
                experts = getattr(block, "experts", None) if block is not None else None
                if isinstance(experts, Qwen3MoeExperts) and expert_idx < experts.num_experts:
                    with torch.no_grad():
                        if proj == "gate_proj":
                            target_slice = experts.gate_up_proj[expert_idx, : experts.intermediate_size, :]
                            target_slice.copy_(loaded_weight.to(device=target_slice.device, dtype=target_slice.dtype))
                        elif proj == "up_proj":
                            target_slice = experts.gate_up_proj[expert_idx, experts.intermediate_size :, :]
                            target_slice.copy_(loaded_weight.to(device=target_slice.device, dtype=target_slice.dtype))
                        elif proj == "gate_up_proj":
                            # FSDP-sync path: actor (VeOmni v5) ships the
                            # already-fused per-expert tensor under
                            # ``experts.{idx}.gate_up_proj.weight``; copy
                            # straight into the matching per-expert slot.
                            target_slice = experts.gate_up_proj[expert_idx]
                            target_slice.copy_(loaded_weight.to(device=target_slice.device, dtype=target_slice.dtype))
                        elif proj == "down_proj":
                            target_slice = experts.down_proj[expert_idx]
                            target_slice.copy_(loaded_weight.to(device=target_slice.device, dtype=target_slice.dtype))
                    direct_loaded_blocks.add(prefix)
                continue

        # FSDP-sync from the actor (VeOmni v5) delivers experts as fused stacked
        # tensors named ``<prefix>.experts.gate_up_proj`` and
        # ``<prefix>.experts.down_proj``. Those names land directly in
        # ``full_param_dict`` (our experts are stored in the same fused layout),
        # so the elif below copies them in one shot.

        # in the checkpoints there is a weight with name model.embed_tokens.weight
        if full_name == "model.embed_tokens.weight":
            embed_tokens_weight = loaded_weight

        if full_name in full_param_dict:
            with torch.no_grad():
                full_param_dict[full_name].data.copy_(loaded_weight)

    # tied_weight_keys should be empty when tie_word_embedding config is false
    # controlled in load_weights_from_weight_iterator
    for param_name in tied_weight_keys:
        if param_name in full_param_dict:
            if "model.embed_tokens.weight" in full_param_dict:
                # the registered param in the model instance also has name model.embed_tokens.weight
                logger.info(f"[VEXACT] Tying weight {param_name} model.embed_tokens.weight")
                full_param_dict[param_name].data = full_param_dict["model.embed_tokens.weight"].data
            elif embed_tokens_weight is not None:
                logger.info(
                    f"[VEXACT] Detected weight tying keys {param_name},"
                    "model.embed_tokens.weight exists in checkpoints but do not exist in model params dict"
                    f"we are copying model.embed_tokens.weight into {param_name} directly."
                )
                with torch.no_grad():
                    full_param_dict[param_name].data.copy_(embed_tokens_weight)

    return len(direct_loaded_blocks)


def apply_qwen3_moe_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    Qwen3MoePreTrainedModel._load_pretrained_model = _patched_load_pretrained_model
    Qwen3MoePreTrainedModel.load_weights = load_qwen3_moe_weights
    Qwen3MoeSparseMoeBlock.__init__ = moe_block_init
    Qwen3MoeSparseMoeBlock.forward = moe_block_forward

    logger.info("Applied Qwen3-MoE monkey patches.")
    _PATCHED = True


__all__ = [
    "Qwen3MoeForCausalLM",
    "Qwen3MoeForTokenClassification",
    "Qwen3MoeForSequenceClassification",
    "apply_qwen3_moe_patches",
]
