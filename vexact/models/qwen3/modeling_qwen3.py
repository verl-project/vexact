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
"""Vexact patches for HF Qwen3."""

import logging

from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP

from vexact.models.fused_projections import register_fused_gated_mlp, register_fused_qwen3_attention


logger = logging.getLogger(__name__)

_PATCHED = False


def apply_qwen3_patches() -> None:
    """Declare the projections vexact may fuse for inference (see fused_projections). Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    register_fused_qwen3_attention(Qwen3Attention)  # q_proj, k_proj, v_proj
    register_fused_gated_mlp(Qwen3MLP)  # gate_proj, up_proj

    logger.info("Applied Qwen3 monkey patches.")
    _PATCHED = True


__all__ = ["apply_qwen3_patches"]
