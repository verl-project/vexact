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

"""The inference-only fused-MoE forward must return VeOmni's bits."""

import pytest
import torch


pytest.importorskip("veomni")
if not torch.cuda.is_available():
    pytest.skip("Triton fused MoE requires CUDA", allow_module_level=True)

import veomni.ops.kernels.moe as moe_ops  # noqa: E402
from veomni.ops.kernels.moe import apply_veomni_fused_moe_patch  # noqa: E402
from veomni.ops.kernels.moe.group_gemm import group_gemm_fused_moe_forward  # noqa: E402
from vexact.batch_invariant_ops.fused_moe_rollout import apply_rollout_fused_moe, swiglu_gate  # noqa: E402
from vexact.batch_invariant_ops.moe_group_gemm_tuning import apply_moe_group_gemm_tuning  # noqa: E402


E, TOPK, HIDDEN, INTER = 128, 8, 2048, 768  # Qwen3-30B-A3B


def _bits(t):
    return t.view(torch.int16)


def test_swiglu_gate_replays_eager_roundings():
    # Every finite bf16 value as a gate input, plus wide-range random gates.
    all_bf16 = torch.arange(1 << 16, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    gate = all_bf16[torch.isfinite(all_bf16)]
    gate = torch.cat([gate, (torch.randn(1 << 20, device="cuda") * 8).to(torch.bfloat16)])
    gate = gate[: gate.numel() // INTER * INTER].view(-1, INTER)
    rows = gate.shape[0]
    up = (torch.randn(rows, INTER, device="cuda") * 3).to(torch.bfloat16)
    weights = torch.rand(rows, device="cuda").to(torch.bfloat16)
    order = torch.randperm(rows, device="cuda")

    ref = torch.ops.aten.silu(gate) * up * weights[order][:, None]
    out = swiglu_gate(torch.cat([gate, up], dim=1), weights, order)
    same = (_bits(out) == _bits(ref)) | (torch.isnan(out) & torch.isnan(ref))
    assert bool(same.all())


def _routing(tokens, flip=False):
    bias = torch.linspace(0, 1.5, E, device="cuda")
    logits = torch.randn(tokens, E, device="cuda") + (bias.flip(0) if flip else bias)
    w, idx = torch.softmax(logits, -1).topk(TOPK, dim=-1)
    return (w / w.sum(-1, keepdim=True)).to(torch.bfloat16), idx


@pytest.fixture(scope="module")
def fused_forward():
    apply_veomni_fused_moe_patch(fused_moe_kernel="triton")
    apply_moe_group_gemm_tuning()
    apply_rollout_fused_moe()
    return moe_ops._fused_moe_forward


# 256 tokens x top-8 = 2048 routed rows: both sides of the matmul_ogs switch.
@pytest.mark.parametrize("tokens", [1, 8, 64, 256, 257, 512, 2048, 4096])
@torch.inference_mode()
def test_fused_forward_matches_veomni(fused_forward, tokens):
    torch.manual_seed(tokens)
    w1 = (torch.randn(E, 2 * INTER, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E, HIDDEN, INTER, device="cuda") * 0.02).to(torch.bfloat16)

    def run(fwd, gw, idx, x):
        return fwd(E, gw, idx, x, None, None, w2, fc1_1_2_weight=w1)

    x = torch.randn(tokens, HIDDEN, device="cuda").to(torch.bfloat16)
    gw, idx = _routing(tokens)
    assert torch.equal(_bits(run(fused_forward, gw, idx, x)), _bits(run(group_gemm_fused_moe_forward, gw, idx, x)))

    # CUDA-graph replay with new routing written into the captured inputs.
    sx, sgw, sidx = x.clone(), gw.clone(), idx.clone()
    run(fused_forward, sgw, sidx, sx)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run(fused_forward, sgw, sidx, sx)
    x2 = torch.randn_like(x)
    gw2, idx2 = _routing(tokens, flip=True)
    sx.copy_(x2)
    sgw.copy_(gw2)
    sidx.copy_(idx2)
    graph.replay()
    assert torch.equal(_bits(out), _bits(run(group_gemm_fused_moe_forward, gw2, idx2, x2)))
