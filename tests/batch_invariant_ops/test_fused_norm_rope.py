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

"""Fused RMSNorm / RoPE must equal the eager HF expressions bit for bit under batch-invariant mode."""

import pytest
import torch

from vexact.batch_invariant_ops import set_batch_invariant_mode


if not torch.cuda.is_available():
    pytest.skip("Fused norm/RoPE kernels require CUDA", allow_module_level=True)

from vexact.batch_invariant_ops.fused_norm_rope import _rms_supported, rms_norm, rope  # noqa: E402


def _bits(t):
    return t.view(torch.int16)


# Qwen3 q/k norms (128) and hidden sizes; odd row counts exercise the packed 128-wide kernel's tail.
@pytest.mark.parametrize("n, rows", [(128, 33), (128, 4097), (128, 65536), (1024, 4096), (2048, 8192), (4096, 4096)])
def test_rms_norm_matches_eager(n, rows):
    assert _rms_supported(n)
    g = torch.Generator(device="cuda").manual_seed(n + rows)
    scale = torch.exp2(torch.randint(-8, 9, (rows, 1), device="cuda", generator=g).float())
    x = (torch.randn(rows, n, device="cuda", generator=g) * scale).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(n, device="cuda", generator=g)).to(torch.bfloat16)
    with set_batch_invariant_mode(True):
        xf = x.float()
        ref = w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
        out = rms_norm(x, w, 1e-6)
    assert torch.equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("tokens, heads", [(1, 32), (300, 40), (512, 4), (8192, 32)])
def test_rope_matches_eager(tokens, heads):
    d = 128
    x = (torch.randn(tokens, heads, d, device="cuda") * 4).to(torch.bfloat16)
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, d, 2, device="cuda").float() / d))
    pos = torch.arange(tokens, device="cuda").float() + 1000
    emb = torch.cat([pos[:, None] * inv_freq[None]] * 2, dim=-1)
    cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
    rotated = torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)
    ref = x * cos[:, None, :] + rotated * sin[:, None, :]
    assert torch.equal(_bits(rope(x, cos, sin)), _bits(ref))
