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

"""Register Vexact-specific model overrides with Transformers."""

import os


_REGISTERED = False


def register_models() -> None:
    """Import patched modules once to activate monkey patches."""
    global _REGISTERED
    if _REGISTERED:
        return

    disable_vexact_patch = os.getenv("VEXACT_DISABLE_MODEL_PATCH", "0") == "1"
    if not disable_vexact_patch:
        # Bind VeOmni's fused MoE kernel so the rollout side reuses the same
        # implementation as VeOmni's actor side (required for bitwise
        # alignment on MoE archs). Use Triton on SM90/Hopper and quack on
        # SM100+/Blackwell, matching vexact.batch_invariant_ops.fused_moe.
        #
        # ``register_models`` runs in every verl process that imports vexact,
        # including CPU-only AgentLoopWorker / data workers that don't have
        # the GPU kernels available. Skip silently there.
        import torch

        from veomni.ops.kernels.moe import apply_veomni_fused_moe_patch

        preferred_moe_kernel = "triton"
        try:
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
                preferred_moe_kernel = "quack"
        except RuntimeError:
            pass

        try:
            apply_veomni_fused_moe_patch(fused_moe_kernel=preferred_moe_kernel)
            print(f"[VEXACT] register_models(): bound VeOmni fused MoE kernel ({preferred_moe_kernel})")
        except RuntimeError as preferred_error:
            print(
                "[VEXACT] register_models(): skipping VeOmni MoE kernel binding "
                f"(no GPU kernel available: {preferred_moe_kernel}={preferred_error})"
            )
        else:
            if preferred_moe_kernel == "triton":
                # Bit-identical launch tuning of the Triton expert GEMMs (see module docstring).
                from vexact.batch_invariant_ops.moe_group_gemm_tuning import apply_moe_group_gemm_tuning

                apply_moe_group_gemm_tuning()
                print("[VEXACT] register_models(): tuned VeOmni Triton grouped-GEMM launches")
                # Inference-only fused MoE forward, bitwise identical to VeOmni's (no-grad calls only;
                # see vexact/batch_invariant_ops/fused_moe_rollout.py).
                from vexact.batch_invariant_ops.fused_moe_rollout import apply_rollout_fused_moe

                apply_rollout_fused_moe()
                print("[VEXACT] register_models(): inference fused MoE forward active")

        # Bit-exact fused RMSNorm / RoPE for no-grad calls (HF Qwen3, Qwen3-MoE).
        from vexact.batch_invariant_ops.fused_norm_rope import apply_fused_norm_rope_patches

        apply_fused_norm_rope_patches()
        print("[VEXACT] register_models(): fused RMSNorm/RoPE for inference (Qwen3, Qwen3-MoE)")

        # One GEMM for q/k/v and for gate/up in no-grad forwards (weights fused at model creation).
        from vexact.models.fused_projections import apply_fused_projection_patches

        apply_fused_projection_patches()
        print("[VEXACT] register_models(): fused QKV and gate/up projections for inference (Qwen3, Qwen3-MoE)")

        from .qwen3_moe.modeling_qwen3_moe import apply_qwen3_moe_patches

        apply_qwen3_moe_patches()
        print("[VEXACT] register_models(): Qwen3-MoE monkey patch active")

        from .deepseek_v3.modeling_deepseek_v3 import apply_deepseek_v3_patches

        apply_deepseek_v3_patches()
        print("[VEXACT] register_models(): DeepSeek-V3 monkey patch active")

        from .gpt_oss.modeling_gpt_oss import apply_gpt_oss_patches

        apply_gpt_oss_patches()
        print("[VEXACT] register_models(): GPT-OSS fused_quack monkey patch active")

    use_liger_patch = os.getenv("VEXACT_LIGER_PATCH", "0") == "1"
    if use_liger_patch:
        # This path mimics VeOmni Qwen3 GPU patch to use Liger kernels
        import transformers.models.qwen3.modeling_qwen3 as hf_qwen3

        print("[VEXACT] register_models(): Applying liger kernel to Qwen3.")

        from liger_kernel.transformers.rope import liger_rotary_pos_emb

        hf_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb
        print("[VEXACT] register_models(): patched Liger rope")

        # Note: LigerRMSNorm is not batch-invariant
        # from liger_kernel.transformers.rms_norm import LigerRMSNorm
        # hf_qwen3.Qwen3RMSNorm = LigerRMSNorm
        # print("[VEXACT] register_models: patched Liger RMSNorm")

        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        hf_qwen3.Qwen3MLP = LigerSwiGLUMLP
        print("[VEXACT] register_models(): patched Liger SwiGLUMLP")

    _REGISTERED = True
