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
"""Inference-only fused-MoE forward that is bitwise identical to VeOmni's Triton fused MoE.

VeOmni's forward (``MergedFc1TritonFusedMoeExpertFunction``) runs, per MoE layer:
``moe_scatter`` -> fc1 grouped GEMM -> ``silu(gate)`` -> ``* up`` -> scatter of the gate weights ->
``* gate_weight`` -> fc2 grouped GEMM -> ``moe_gather``. This module produces the same bits with
less memory traffic:

- the token gather is fused into fc1's loads (triton_kernels ``matmul_ogs`` with ``GatherIndx``)
  for prefill-size calls, so ``moe_scatter`` disappears there (a pure load cannot change bits);
- ``silu``, ``* up``, the gate-weight scatter and ``* gate_weight`` become one Triton pass that
  replays VeOmni's three bf16 roundings with IEEE ``expf`` / division (exact for every finite bf16
  gate value, see tests/batch_invariant_ops/test_fused_moe_rollout.py);
- fc1 / fc2 use the grouped-GEMM dispatch from ``moe_group_gemm_tuning`` (tuned VeOmni launch or
  ``matmul_ogs``), which already reproduces VeOmni's bits.

It is used only when autograd is off (rollout, and no-grad log-prob passes); anything that needs
gradients, expert parallelism, split fc1 weights or a SwiGLU clamp goes to VeOmni's function.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _swiglu_gate_kernel(
    fc1_ptr,  # [rows, 2 * INTER] bf16, expert-sorted rows: [gate | up]
    gate_w_ptr,  # [T * topk] bf16 routing weights, flat (token, slot) order
    order_ptr,  # [rows] flat (token, slot) index of each expert-sorted row
    out_ptr,  # [rows, INTER] bf16
    INTER: tl.constexpr,
    BLOCK_I: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,  # False (dense MLP): plain silu(gate) * up, no routing weight
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < INTER
    gate = tl.load(fc1_ptr + row * 2 * INTER + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(fc1_ptr + row * 2 * INTER + INTER + cols, mask=mask, other=0.0).to(tl.float32)
    # aten::silu on bf16: fp32 x / (1 + expf(-x)), rounded to bf16.
    act = tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
    # bf16 * bf16 in aten: fp32 product rounded to bf16, twice.
    act = (act * up).to(tl.bfloat16).to(tl.float32)
    if HAS_WEIGHT:
        w = tl.load(gate_w_ptr + tl.load(order_ptr + row)).to(tl.float32)
        act = act * w
    tl.store(out_ptr + row * INTER + cols, act.to(tl.bfloat16), mask=mask)


def _launch_swiglu(fc1_output: torch.Tensor, gate_weights_flat, sorted_order, has_weight: bool):
    inter = fc1_output.shape[-1] // 2
    out = torch.empty((*fc1_output.shape[:-1], inter), dtype=fc1_output.dtype, device=fc1_output.device)
    rows = out.numel() // inter
    if rows == 0:
        return out
    block_i = min(1024, triton.next_power_of_2(inter))
    _swiglu_gate_kernel[(rows, triton.cdiv(inter, block_i))](
        fc1_output,
        gate_weights_flat,
        sorted_order,
        out,
        INTER=inter,
        BLOCK_I=block_i,
        HAS_WEIGHT=has_weight,
        num_warps=4,
    )
    return out


def swiglu_gate(fc1_output: torch.Tensor, gate_weights_flat: torch.Tensor, sorted_order: torch.Tensor):
    """``(silu(fc1[:, :I]) * fc1[:, I:]) * gate_weights_flat[sorted_order][:, None]`` with VeOmni's
    exact bf16 rounding sequence, in one pass."""
    return _launch_swiglu(fc1_output, gate_weights_flat, sorted_order, has_weight=True)


def swiglu(fc1_output: torch.Tensor):
    """``silu(fc1[..., :I]) * fc1[..., I:]`` for a dense gated MLP, in one pass over a fused gate/up GEMM
    output. bf16 only: it is the kernel above without the routing weight, so it replays aten's roundings
    (silu to bf16, product to bf16) and gives eager's bits."""
    return _launch_swiglu(fc1_output, None, None, has_weight=False)


@triton.jit
def _expert_block_hist_kernel(expert_ptr, hist_ptr, N, BLOCK: tl.constexpr, EP: tl.constexpr):
    """Histogram of expert ids per BLOCK-sized chunk (no atomics)."""
    b = tl.program_id(0)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    e = tl.load(expert_ptr + offs, mask=m, other=0).to(tl.int32)
    tl.store(hist_ptr + b * EP + tl.arange(0, EP), tl.histogram(e, EP, mask=m))


@triton.jit
def _expert_prefix_kernel(
    hist_ptr,
    base_ptr,
    splits_ptr,
    cumsum_ptr,
    n_blocks,
    E,
    EP: tl.constexpr,
    EB: tl.constexpr,
    CHUNK_T: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Program p owns expert columns [p * EB, (p + 1) * EB). Every program sums all columns to get each
    expert's first output row (exclusive cumsum over experts), then scans its own columns down the
    block axis: base[b, e] = first row of expert e + rows of expert e in blocks before b."""
    ex = tl.arange(0, EP)
    rows_t = tl.arange(0, CHUNK_T)
    totals = tl.zeros([EP], dtype=tl.int32)
    for c in range(0, n_blocks, CHUNK_T):
        m = (c + rows_t < n_blocks)[:, None]
        totals += tl.sum(tl.load(hist_ptr + (c + rows_t)[:, None] * EP + ex[None, :], mask=m, other=0), axis=0)
    inclusive = tl.cumsum(totals, axis=0)
    if tl.program_id(0) == 0:
        tl.store(splits_ptr + ex, totals, mask=ex < E)
        tl.store(cumsum_ptr + ex, inclusive.to(tl.int64), mask=ex < E)
    cols = tl.program_id(0) * EB + tl.arange(0, EB)
    # this program's slice of the exclusive cumsum (EB x EP select; no register gather needed)
    running = tl.sum(tl.where(cols[:, None] == ex[None, :], (inclusive - totals)[None, :], 0), axis=1)
    rows = tl.arange(0, CHUNK)
    for c in range(0, n_blocks, CHUNK):
        m = (c + rows < n_blocks)[:, None]
        tile = tl.load(hist_ptr + (c + rows)[:, None] * EP + cols[None, :], mask=m, other=0)
        excl = tl.cumsum(tile, axis=0) - tile
        tl.store(base_ptr + (c + rows)[:, None] * EP + cols[None, :], running[None, :] + excl, mask=m)
        running += tl.sum(tile, axis=0)


@triton.jit
def _expert_scatter_kernel(expert_ptr, base_ptr, order_ptr, scatter_ptr, N, BLOCK: tl.constexpr, EP: tl.constexpr):
    """Stable position of each element of a block. Sorting the unique keys (expert * BLOCK + i) puts the
    block in (expert, index) order; an element at sorted slot j has rank j - (block rows with a smaller
    expert) among its expert's rows. Writes scatter_index[i] = pos and sorted_order[pos] = i."""
    b = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    offs = b * BLOCK + i
    m = offs < N
    e = tl.load(expert_ptr + offs, mask=m, other=0).to(tl.int32)
    h = tl.histogram(e, EP, mask=m)
    smaller = tl.cumsum(h, axis=0) - h  # block rows with a smaller expert id
    key = tl.sort(tl.where(m, e * BLOCK + i, EP * BLOCK))  # padding sorts last
    valid = key < EP * BLOCK
    e_s = tl.where(valid, key // BLOCK, 0)
    src = b * BLOCK + key % BLOCK
    pos = tl.load(base_ptr + b * EP + e_s, mask=valid, other=0) + i - tl.gather(smaller, e_s, 0)
    tl.store(scatter_ptr + src, pos, mask=valid)
    tl.store(order_ptr + pos, src.to(tl.int64), mask=valid)


def _sort_block(n: int) -> int:
    # Larger blocks mean fewer histogram rows to scan; smaller ones keep enough programs busy for small n.
    # The result does not depend on the block size (the stable order is unique).
    return min(1024, max(128, triton.next_power_of_2(triton.cdiv(n, 128))))


def expert_counting_sort(selected_experts: torch.Tensor, num_experts: int):
    """Stable counting sort of expert ids. Returns ``(splits int32 [E], cumsum int64 [E], sorted_order int64
    [N], scatter_index int32 [T, topk])``, identical to VeOmni's ``expert_histogram`` +
    ``compute_expert_scatter_index`` (stable argsort) + ``torch.cumsum``, in three small kernels instead of
    a general-purpose radix sort."""
    flat = selected_experts.reshape(-1)
    n = flat.numel()
    ep = triton.next_power_of_2(num_experts)
    block = _sort_block(n)
    n_blocks = max(1, triton.cdiv(n, block))
    eb = min(16, ep)
    dev = flat.device
    hist = torch.empty((n_blocks, ep), dtype=torch.int32, device=dev)
    base = torch.empty((n_blocks, ep), dtype=torch.int32, device=dev)
    splits = torch.empty((num_experts,), dtype=torch.int32, device=dev)
    cumsum = torch.empty((num_experts,), dtype=torch.int64, device=dev)
    sorted_order = torch.empty((n,), dtype=torch.int64, device=dev)
    scatter_index = torch.empty((n,), dtype=torch.int32, device=dev)
    nw = 8 if block >= 512 else 4
    nb_p2 = triton.next_power_of_2(n_blocks)  # prefix tiles: up to ~8K entries, no taller than the block count
    _expert_block_hist_kernel[(n_blocks,)](flat, hist, n, BLOCK=block, EP=ep, num_warps=nw)
    _expert_prefix_kernel[(ep // eb,)](
        hist,
        base,
        splits,
        cumsum,
        n_blocks,
        num_experts,
        EP=ep,
        EB=eb,
        CHUNK_T=min(nb_p2, max(1, 8192 // ep)),
        CHUNK=min(nb_p2, 8192 // eb),
        num_warps=8 if nb_p2 * ep >= 8192 else 4,
    )
    _expert_scatter_kernel[(n_blocks,)](flat, base, sorted_order, scatter_index, n, BLOCK=block, EP=ep, num_warps=nw)
    return splits, cumsum, sorted_order, scatter_index.view(selected_experts.shape)


def _wrap_fused_moe_forward(veomni_forward):
    """Wrap VeOmni's bound ``_fused_moe_forward`` with the inference-only path above."""
    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.ops.kernels.moe import group_gemm as fused_moe_module
    from veomni.ops.kernels.moe._kernels.kernel.moe import moe_gather, moe_scatter

    from .moe_group_gemm_tuning import _OGS_MIN_ROWS, _load_matmul_ogs

    matmul_ogs = _load_matmul_ogs()

    def forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        fc1_1_2_weight=None,
        swiglu_limit=None,
    ):
        if (
            torch.is_grad_enabled()
            or fc1_1_2_weight is None
            or fc1_1_weight is not None
            or fc1_2_weight is not None
            or swiglu_limit is not None
            or hidden_states.dtype != torch.bfloat16
            or routing_weights.dtype != torch.bfloat16
            or get_parallel_state().ep_enabled
        ):
            return veomni_forward(
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_weight,
                fc1_2_weight,
                fc2_weight,
                fc1_1_2_weight,
                swiglu_limit=swiglu_limit,
            )

        group_gemm = fused_moe_module.group_gemm_same_nk  # tuned / matmul_ogs dispatch, VeOmni-identical
        topk = selected_experts.shape[-1]
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        # Stable counting sort: the same permutation as VeOmni's stable argsort, so every downstream bit
        # is unchanged; replaces expert_histogram + a radix sort + inverse scatter + cumsum.
        splits, cumsum, sorted_order, scatter_index = expert_counting_sort(selected_experts, num_experts)
        rows = sorted_order.shape[0]

        if rows > _OGS_MIN_ROWS:
            gather = (sorted_order.to(torch.int32), scatter_index.reshape(-1).to(torch.int32))
            fc1 = matmul_ogs(x, fc1_1_2_weight.transpose(1, 2), splits, topk, gather)
        else:
            fc1 = group_gemm(
                a=moe_scatter(x, scatter_index),
                b=fc1_1_2_weight,
                cumsum_M=cumsum,
                max_M=rows,
                transpose_a=False,
                transpose_b=True,
            )
        act = swiglu_gate(fc1, routing_weights.reshape(-1), sorted_order)
        fc2 = group_gemm(a=act, b=fc2_weight, cumsum_M=cumsum, max_M=rows, transpose_a=False, transpose_b=True)
        return moe_gather(fc2, scatter_index).reshape(hidden_states.shape)

    forward._vexact_rollout_fused = True
    return forward


def apply_rollout_fused_moe() -> None:
    """Wrap VeOmni's currently bound fused-MoE forward, which must be the Triton one. Idempotent."""
    import veomni.ops.kernels.moe as moe_ops
    from veomni.ops.kernels.moe.group_gemm import group_gemm_fused_moe_forward

    current = moe_ops._fused_moe_forward
    if getattr(current, "_vexact_rollout_fused", False):
        return
    if current is not group_gemm_fused_moe_forward:
        # Only the Triton grouped-GEMM forward is reproduced bit for bit.
        raise RuntimeError(f"expected VeOmni's Triton fused-MoE forward to be bound, got {current!r}")
    moe_ops._fused_moe_forward = _wrap_fused_moe_forward(current)
