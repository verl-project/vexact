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

from vexact.batch_invariant_ops import set_batch_invariant_mode


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Batch invariant overrides require CUDA kernels",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch, dim", [(32, 256), (64, 512), (128, 1024)])
def test_mm_batch_invariance(dtype: torch.dtype, batch: int, dim: int) -> None:
    device = torch.device("cuda")

    if dtype is torch.bfloat16:
        try:
            torch.mm(
                torch.ones((2, 2), dtype=dtype, device=device),
                torch.ones((2, 2), dtype=dtype, device=device),
            )
        except (RuntimeError, TypeError):
            pytest.skip(f"torch.mm does not support {dtype} on {device.type}")

    a = torch.linspace(-1.0, 1.0, batch * dim, dtype=dtype, device=device).reshape(batch, dim)
    b = torch.linspace(-1.0, 1.0, dim * dim, dtype=dtype, device=device).reshape(dim, dim)

    with set_batch_invariant_mode(True):
        single_token = torch.mm(a[:1], b)
        full_batch = torch.mm(a, b)[:1]

    assert torch.allclose(single_token, full_batch, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton persistent matmul requires CUDA")
@pytest.mark.parametrize("n, k", [(4096, 2048), (2048, 6144), (151936, 2048)])
@pytest.mark.parametrize("m", [1, 7, 64, 65, 129, 1000])
def test_bf16_tile_candidates_are_bitwise_identical(monkeypatch, m: int, n: int, k: int) -> None:
    """Tile choice must not change bits: every candidate that fits the GPU reduces K in the same order,
    and rows beyond M (EVEN_M=False) are masked without affecting valid rows."""
    from vexact.batch_invariant_ops import batch_invariant_ops as bio

    g = torch.Generator(device="cuda").manual_seed(m)
    scale = torch.exp2(torch.randint(-4, 5, (m, k), device="cuda", generator=g).float())
    a = (torch.randn(m, k, device="cuda", generator=g) * scale).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    bias = (torch.randn(n, device="cuda", generator=g) * 0.1).to(torch.bfloat16)

    c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    launches = [bio._launch_args(a, w.t(), c, bb, bio.get_compute_units()) for bb in (None, bias)]
    cfgs = [bio._as_cfg(cand) for cand in bio._BF16_MATMUL_CANDIDATES]
    cfgs = [cfg for cfg in cfgs if all(bio._fits_shared_memory(args, flags, cfg) for args, flags in launches)]
    assert len(cfgs) >= 2

    outs = []
    for cfg in cfgs:
        monkeypatch.setattr(bio, "_bf16_matmul_config", lambda *_, cfg=cfg: cfg)
        outs.append((bio.matmul_persistent(a, w.t()), bio.matmul_persistent(a, w.t(), bias)))
    for plain, biased in outs[1:]:
        assert torch.equal(plain.view(torch.int16), outs[0][0].view(torch.int16))
        assert torch.equal(biased.view(torch.int16), outs[0][1].view(torch.int16))
