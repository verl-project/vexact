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

"""Shared MoE backend selection helpers."""


def select_fused_moe_kernel(requested_kernel: str | None = None, device_major: int | None = None) -> str:
    """Return the fused MoE kernel that is supported on the target device."""
    if device_major is None:
        try:
            import torch

            if torch.cuda.is_available():
                device_major = torch.cuda.get_device_capability()[0]
        except RuntimeError:
            pass

    if device_major is not None and device_major < 10 and requested_kernel == "quack":
        return "triton"
    if requested_kernel is not None:
        return requested_kernel
    if device_major is not None and device_major >= 10:
        return "quack"
    return "triton"
