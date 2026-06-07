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

from .batch_invariant_ops import (
    AttentionBlockSize,
    batch_invariant_rms_norm,
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
    get_batch_invariant_attention_block_size,
    is_batch_invariant_mode_enabled,
    log_softmax,
    matmul_persistent,
    mean_dim,
    set_batch_invariant_mode,
    triton_bmm,
)
from .flash_attention import flash_attention_forward, flash_attention_forward_cute
from .flex_attention import flex_attention_forward
from .triton_invariant_attention import flash_attention_forward as triton_flash_attention_forward
from .triton_invariant_attention import flash_attn_varlen_func as triton_flash_attn_varlen_func


__version__ = "0.1.0"

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
    "matmul_persistent",
    "log_softmax",
    "mean_dim",
    "get_batch_invariant_attention_block_size",
    "AttentionBlockSize",
    "flash_attention_forward",
    "flash_attention_forward_cute",
    "flex_attention_forward",
    "triton_flash_attn_varlen_func",
    "triton_flash_attention_forward",
    "triton_bmm",
    "batch_invariant_rms_norm",
]
