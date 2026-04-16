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
import torch.nn.functional as F

from vexact.batch_invariant_ops import set_batch_invariant_mode


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("batch", [8, 64])
@pytest.mark.parametrize(
    "model_config",
    [
        # Qwen3-VL-2B/4B-Instruct
        # https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json
        # https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/config.json
        {
            "name": "Qwen3-VL-2B and 4B",
            "hidden_size": 1024,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "in_channels": 3,
        },
        # Qwen3-VL-8B/32B-Instruct
        # https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json
        # https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/blob/main/config.json
        {
            "name": "Qwen3-VL-8B and 32B",
            "hidden_size": 1152,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "in_channels": 3,
        },
    ],
)
def test_conv3d_determinism_and_batch_invariance(dtype: torch.dtype, batch: int, model_config: dict) -> None:
    device = torch.device("cuda")

    # Extract dimensions from config
    in_channels = model_config["in_channels"]
    out_channels = model_config["hidden_size"]
    # Kernel size is [temporal_patch_size, patch_size, patch_size]
    # Stride is same as kernel size for PatchEmbed
    # See https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L68 # noqa: E501 long url
    kt, kh, kw = (
        model_config["temporal_patch_size"],
        model_config["patch_size"],
        model_config["patch_size"],
    )

    # Input spatial dims (arbitrary valid size multiple of patch)
    # e.g. 2 frames, 32x32 image
    d = kt * 2
    h = kh * 2
    w = kw * 2

    # Create deterministic inputs using linspace
    # Input shape: (N, C_in, D, H, W)
    input_numel = batch * in_channels * d * h * w
    a = torch.linspace(-1.0, 1.0, input_numel, dtype=dtype, device=device).reshape(batch, in_channels, d, h, w)

    # Weight shape: (C_out, C_in, KT, KH, KW)
    weight_numel = out_channels * in_channels * kt * kh * kw
    weight = torch.linspace(-1.0, 1.0, weight_numel, dtype=dtype, device=device).reshape(
        out_channels, in_channels, kt, kh, kw
    )

    # Bias shape: (C_out,)
    bias = torch.linspace(-1.0, 1.0, out_channels, dtype=dtype, device=device)

    # Stride same as kernel
    stride = (kt, kh, kw)

    # NOTE: torch.use_deterministic_algorithms(True) is not used here. But PyTorch' warns
    # that conv3d might be non-deterministic when not using torch.use_deterministic_algorithms(True).
    # torch.use_deterministic_algorithms(True)

    # NOTE: Disabling cudnn backend is necessary. Otherwise the ops is not batch invariant when test with
    # batch_size=64 vs batch_size=1.
    torch.backends.cudnn.enabled = False

    with set_batch_invariant_mode(True):
        # 1. Test run-to-run determinism
        # Using F.conv3d instead of nn.Conv3d functional call to avoid confusion
        out1 = F.conv3d(a, weight, bias=bias, stride=stride)

        # Run 10 times to ensure bitwise close to the baseline run
        for i in range(10):
            current = F.conv3d(a, weight, bias=bias, stride=stride)
            assert torch.allclose(out1, current, atol=0.0, rtol=0.0), f"Run-to-run determinism failed at iteration {i}"

        # 2. Test batch invariance
        # Expected: single item forwarded alone == single item sliced from batch forward
        single_input = a[:1]
        single_token_out = F.conv3d(single_input, weight, bias=bias, stride=stride)

        full_batch_out_slice = out1[:1]

        assert torch.allclose(single_token_out, full_batch_out_slice, atol=0.0, rtol=0.0), (
            f"Batch invariance failed for {model_config['name']}! Max diff: "
            f"{(single_token_out - full_batch_out_slice).abs().max()}"
        )
