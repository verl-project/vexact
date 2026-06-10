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

import pytest
import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


def _tiny_deepseek_v3_config() -> DeepseekV3Config:
    return DeepseekV3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=16,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
        architectures=["DeepseekV3ForCausalLM"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deepseek_fused_lce_log_probs_backward() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from veomni.arguments import OpsImplementationConfig
    from veomni.models import build_foundation_model
    from vexact.batch_invariant_ops import triton_flash_attention_forward
    from vexact.batch_invariant_ops.kv_cache_context import _context_storage

    ALL_ATTENTION_FUNCTIONS["triton-invariant"] = triton_flash_attention_forward
    if hasattr(_context_storage, "kv_cache_context"):
        delattr(_context_storage, "kv_cache_context")

    config = _tiny_deepseek_v3_config()
    ops_implementation = OpsImplementationConfig(
        attn_implementation="triton-invariant",
        moe_implementation="eager",
        cross_entropy_loss_implementation="chunk_loss",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
    )
    model = build_foundation_model(
        config_path=config,
        weights_path=None,
        torch_dtype="bfloat16",
        init_device="cuda",
        ops_implementation=ops_implementation,
    )
    model.train()

    seq_lens = [8, 11]
    total_tokens = sum(seq_lens)
    input_ids = torch.randint(3, config.vocab_size, (1, total_tokens), device="cuda")
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.cat([torch.arange(seq_len, device="cuda") for seq_len in seq_lens]).unsqueeze(0)
    cu_seqlens = torch.tensor([0, seq_lens[0], total_tokens], dtype=torch.int32, device="cuda")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max(seq_lens),
        max_length_k=max(seq_lens),
        labels=labels,
        use_cache=False,
        return_dict=True,
        return_log_probs=True,
    )
    assert outputs.log_probs is not None
    assert outputs.log_probs.shape == input_ids.shape

    loss = -(outputs.log_probs * attention_mask).sum() / attention_mask.sum()
    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_norm = torch.stack(
        [param.grad.detach().float().norm() for param in model.parameters() if param.grad is not None]
    ).sum()
    assert torch.count_nonzero(grad_norm) > 0
