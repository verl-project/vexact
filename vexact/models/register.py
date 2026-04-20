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
        from .qwen3_moe.modeling_qwen3_moe import apply_qwen3_moe_patches

        apply_qwen3_moe_patches()
        print("[VEXACT] register_models(): Qwen3-MoE monkey patch active")

        from .deepseek_v3.modeling_deepseek_v3 import apply_deepseek_v3_patches

        apply_deepseek_v3_patches()
        print("[VEXACT] register_models(): DeepSeek-V3 monkey patch active")

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
