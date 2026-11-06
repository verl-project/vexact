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

"""The tuned fused-MoE expert GEMM must return VeOmni's bits at every routed-row count."""

import pytest
import torch


pytest.importorskip("veomni")
if not torch.cuda.is_available():
    pytest.skip("Triton grouped GEMM requires CUDA", allow_module_level=True)

from veomni.ops.kernels.moe import group_gemm as fused_moe_module  # noqa: E402
from veomni.ops.kernels.moe._kernels.kernel.moe import expert_histogram, moe_scatter  # noqa: E402
from veomni.ops.kernels.moe._scatter import compute_expert_scatter_index  # noqa: E402
from vexact.batch_invariant_ops.moe_group_gemm_tuning import apply_moe_group_gemm_tuning  # noqa: E402


E, TOPK, HIDDEN, INTER = 128, 8, 2048, 768  # Qwen3-30B-A3B


@pytest.fixture(scope="module")
def gemms():
    original = fused_moe_module.group_gemm_same_nk
    apply_moe_group_gemm_tuning()
    tuned = fused_moe_module.group_gemm_same_nk
    fused_moe_module.group_gemm_same_nk = original
    return original, tuned


@pytest.mark.parametrize("tokens", [1, 16, 64, 300, 512, 1024, 4096])
def test_tuned_group_gemm_matches_veomni(gemms, tokens):
    original, tuned = gemms
    g = torch.Generator(device="cuda").manual_seed(tokens)
    logits = torch.randn(tokens, E, device="cuda", generator=g) + torch.linspace(0, 1.5, E, device="cuda")
    idx = logits.topk(TOPK, dim=-1).indices.to(torch.int32)
    cumsum = torch.cumsum(expert_histogram(idx, E), dim=0)
    _, scatter_index = compute_expert_scatter_index(idx)
    x = torch.randn(tokens, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    w1 = (torch.randn(E, 2 * INTER, HIDDEN, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E, HIDDEN, INTER, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    a2 = torch.randn(tokens * TOPK, INTER, device="cuda", generator=g).to(torch.bfloat16)
    for a, w in ((moe_scatter(x, scatter_index), w1), (a2, w2)):
        kw = dict(a=a, b=w, cumsum_M=cumsum, max_M=a.shape[0], transpose_a=False, transpose_b=True)
        assert torch.equal(tuned(**kw).view(torch.int16), original(**kw).view(torch.int16))
