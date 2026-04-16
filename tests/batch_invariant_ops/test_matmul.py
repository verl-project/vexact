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
