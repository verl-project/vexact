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
Monkey patch HuggingFace Transformers DeepSeek-V3 model to support high-performance fused MoE kernels
and MLA (Multi-head Latent Attention) compatibility with vexact attention backends.
"""

import logging
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3ForCausalLM,
    DeepseekV3ForSequenceClassification,
    DeepseekV3ForTokenClassification,
    DeepseekV3MLP,
    DeepseekV3PreTrainedModel,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
)
from transformers.processing_utils import Unpack


logger = logging.getLogger(__name__)

### Config compatibility for older DeepSeek V3 checkpoints


def _ensure_config_compat(config):
    """
    Ensure derived config attributes exist for configs saved by older transformers versions
    or custom config classes (via auto_map). The built-in DeepseekV3Config computes these
    in __init__, but custom configs from older checkpoints may not have them.
    """
    if not hasattr(config, "qk_head_dim"):
        config.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    if not hasattr(config, "head_dim"):
        config.head_dim = config.qk_rope_head_dim
    if not hasattr(config, "rope_interleave"):
        # DeepSeek V3 / Moonlight weights store RoPE dimensions in interleaved format [r0,i0,r1,i1,...].
        # apply_rotary_pos_emb_interleave rearranges to [r0,r1,...,i0,i1,...] before rotate_half.
        # Match HF's built-in DeepseekV3Config default (True) for custom configs that omit this field.
        config.rope_interleave = True
    if not hasattr(config, "_moe_implementation"):
        config._moe_implementation = "fused"


_original_pretrained_init = DeepseekV3PreTrainedModel.__init__


def _compat_pretrained_init(self, config, *args, **kwargs):
    _ensure_config_compat(config)
    _original_pretrained_init(self, config, *args, **kwargs)


### Patching HuggingFace DeepseekV3 modeling


# ============================================================================
# MoE classes copied from VeOmni to ensure identical behavior with actor side.
# Only the fused_moe_forward import differs (uses vexact's batch-invariant version).
# ============================================================================


class PatchDeepseekV3TopkRouter(nn.Module):
    """Identical to VeOmni's PatchDeepseekV3TopkRouter."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.n_routed_experts), requires_grad=False)

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits


class PatchDeepseekV3NaiveMoe(nn.Module):
    """Identical to VeOmni's PatchDeepseekV3NaiveMoe, but uses vexact's fused_moe_forward."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self._moe_implementation = getattr(config, "_moe_implementation", "fused")

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from vexact.batch_invariant_ops.fused_moe import fused_moe_forward

        final_hidden_states = torch.zeros_like(hidden_states)

        if self._moe_implementation == "eager":
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate = nn.functional.linear(current_state, self.gate_proj[expert_idx])
                up = nn.functional.linear(current_state, self.up_proj[expert_idx])
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        elif self._moe_implementation == "fused":
            top_k_weights = top_k_weights.to(final_hidden_states.dtype)
            final_hidden_states = fused_moe_forward(
                num_experts=self.num_experts,
                routing_weights=top_k_weights,
                selected_experts=top_k_index,
                hidden_states=hidden_states,
                fc1_1_weight=self.gate_proj,
                fc1_2_weight=self.up_proj,
                fc2_weight=self.down_proj,
            )
        else:
            raise ValueError(f"Invalid moe implementation: {self._moe_implementation}")

        return final_hidden_states


class PatchDeepseekV3MoE(nn.Module):
    """Identical to VeOmni's PatchDeepseekV3MoE."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = PatchDeepseekV3NaiveMoe(config)
        self.gate = PatchDeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


# Keep backward-compat aliases
DeepseekV3Experts = PatchDeepseekV3NaiveMoe


def deepseek_v3_attention_forward(
    self: DeepseekV3Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
    key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

    if self.q_lora_rank is None:
        q_states = self.q_proj(hidden_states)
    else:
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    cos, sin = position_embeddings
    if self.config.rope_interleave:  # support using interleaved weights for efficiency
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    else:
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


### Model patching ends

### Utilities for loading fused experts weights from separate tensors in the checkpoint
_PATCHED = False


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
    raise NotImplementedError(
        "When using VeXact DeepseekV3, please use from_config to initialize the model, "
        "then use VeXact model loader to load weights",
    )


### Custom weight loading method for vexact weight loaders.
### We expect models are created first by from_config,
### then weights are loaded by load_weights.
### We override load_weights to fuse expert weights.
### These should be removed after we upgrade to HF Transformers v5.
_EXPERT_PROJS = {"gate_proj", "up_proj", "down_proj"}


def load_deepseek_v3_weights(
    self: nn.Module,
    weight_iterator: Iterable[tuple[str, torch.Tensor]],
    tied_weight_keys: Optional[list[str]] = None,
) -> int:
    """
    Custom weight loading function to handle fused experts weights stored in separate tensors in the checkpoint.

    :param self: A patched DeepSeek-V3 model instance with DeepseekV3Experts modules
    :param weight_iterator: Iterator of (name, tensor) pairs from checkpoint
    :param tied_weight_keys: Weight keys to tie (e.g., lm_head.weight -> embed_tokens.weight)
    :return: Number of expert blocks that were directly loaded
    """
    full_param_dict = dict(self.named_parameters())
    direct_loaded_blocks: set[str] = set()
    embed_tokens_weight = None

    for full_name, loaded_weight in weight_iterator:
        if ".experts." in full_name and full_name.endswith(".weight"):
            # Expected expert key format:
            #   <prefix>.experts.<expert_idx>.<proj>.weight
            # Example:
            #   model.layers.3.mlp.experts.42.gate_proj.weight
            # Copy received single expert weight into corresponding position in fused expert params
            prefix, rest = full_name.split(".experts.", 1)
            try:
                expert_idx_str, proj, suffix = rest.split(".")
            except ValueError:
                expert_idx_str, proj, suffix = None, None, None
            if suffix == "weight" and proj in _EXPERT_PROJS and expert_idx_str is not None and expert_idx_str.isdigit():
                expert_idx = int(expert_idx_str)
                try:
                    block = self.get_submodule(prefix)
                except AttributeError:
                    block = None
                experts = getattr(block, "experts", None) if block is not None else None
                target = getattr(experts, proj, None) if isinstance(experts, PatchDeepseekV3NaiveMoe) else None
                if target is not None and expert_idx < target.shape[0]:
                    with torch.no_grad():
                        target[expert_idx].copy_(loaded_weight.to(device=target.device, dtype=target.dtype))
                    direct_loaded_blocks.add(prefix)
                continue

        if full_name == "model.embed_tokens.weight":
            embed_tokens_weight = loaded_weight

        # Skip rotary_emb.inv_freq — it is a non-persistent buffer initialized
        # in fp32 during model construction and never needs reloading.
        if full_name.endswith(".rotary_emb.inv_freq"):
            continue

        if full_name in full_param_dict:
            with torch.no_grad():
                full_param_dict[full_name].data.copy_(loaded_weight)

    # tied_weight_keys should be empty when tie_word_embedding config is false
    # controlled in load_weights_from_weight_iterator
    for param_name in tied_weight_keys or []:
        if param_name in full_param_dict:
            if "model.embed_tokens.weight" in full_param_dict:
                logger.info(f"[VEXACT] Tying weight {param_name} to model.embed_tokens.weight")
                full_param_dict[param_name].data = full_param_dict["model.embed_tokens.weight"].data
            elif embed_tokens_weight is not None:
                logger.info(
                    f"[VEXACT] Detected weight tying keys {param_name}, "
                    "model.embed_tokens.weight exists in checkpoints but does not exist in model params dict. "
                    f"Copying model.embed_tokens.weight into {param_name} directly."
                )
                with torch.no_grad():
                    full_param_dict[param_name].data.copy_(embed_tokens_weight)

    return len(direct_loaded_blocks)


def _make_deterministic_rope_forward():
    """Build a RotaryEmbedding.forward that uses a deterministic Triton bmm kernel.

    The default ``inv_freq @ position_ids`` dispatches to cuBLAS bmm which is
    non-deterministic on the first call for certain GPU architectures.  Replacing
    it with an explicit Triton batched-GEMM kernel eliminates this issue.
    """
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import dynamic_rope_update

    from vexact.batch_invariant_ops import triton_bmm

    @torch.no_grad()
    @dynamic_rope_update
    def _deterministic_rope_forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = triton_bmm(
                inv_freq_expanded.float().contiguous(),
                position_ids_expanded.float().contiguous(),
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    return _deterministic_rope_forward


def apply_deepseek_v3_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    DeepseekV3PreTrainedModel.__init__ = _compat_pretrained_init
    DeepseekV3PreTrainedModel._load_pretrained_model = _patched_load_pretrained_model
    DeepseekV3PreTrainedModel.load_weights = load_deepseek_v3_weights

    # Replace MoE class with VeOmni-aligned version (identical router, MoE, shared experts).
    # Only the attention forward differs (VeXact uses fa-invariant with paged KV).
    import transformers.models.deepseek_v3.modeling_deepseek_v3 as hf_deepseek_v3

    hf_deepseek_v3.DeepseekV3MoE = PatchDeepseekV3MoE

    DeepseekV3Attention.forward = deepseek_v3_attention_forward

    # Patch RotaryEmbedding to use deterministic Triton bmm for cos/sin computation
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding

    DeepseekV3RotaryEmbedding.forward = _make_deterministic_rope_forward()

    # Patch RMSNorm to use batch-invariant Triton kernel
    _patch_rms_norm_batch_invariant()

    logger.info("Applied DeepSeek-V3 monkey patches.")
    _PATCHED = True


def _patch_rms_norm_batch_invariant():
    """Replace DeepseekV3RMSNorm.forward with batch-invariant Triton implementation."""
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm

    from vexact.batch_invariant_ops import batch_invariant_rms_norm

    def _bi_rms_norm_forward(self, hidden_states):
        return batch_invariant_rms_norm(hidden_states, self.weight, self.variance_epsilon)

    DeepseekV3RMSNorm.forward = _bi_rms_norm_forward


__all__ = [
    "DeepseekV3ForCausalLM",
    "DeepseekV3ForTokenClassification",
    "DeepseekV3ForSequenceClassification",
    "apply_deepseek_v3_patches",
]
