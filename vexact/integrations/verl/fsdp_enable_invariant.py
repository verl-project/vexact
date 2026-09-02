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

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# Load verl's fused-linear module before enabling batch-invariant mode so the mode can
# null its Liger dispatch handle (see ``_disable_verl_liger_fused_linear``).
import verl.utils.experimental.torch_functional  # noqa: F401
from vexact.batch_invariant_ops import (
    enable_batch_invariant_mode,
    triton_flash_attention_forward,
)
from vexact.models.register import register_models as _register_models


# Also re-applies the verl Liger hook when the mode was already enabled by an earlier import.
enable_batch_invariant_mode()

ALL_ATTENTION_FUNCTIONS["triton-invariant"] = triton_flash_attention_forward
_register_models()
print("fsdp worker batch invariant enabled.")
