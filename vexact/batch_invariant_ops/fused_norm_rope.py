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
"""Fused RMSNorm and RoPE that are bitwise identical to the eager HF ops under batch-invariant mode.

Eager HF RMSNorm (Qwen3, Llama-style) runs ~7 kernels:
    x.float(); var = x.pow(2).mean(-1, keepdim=True); y = x * rsqrt(var + eps); weight * y.to(bf16)
where `mean` is vexact's `mean_kernel` (batch-invariant mode). Eager RoPE runs ~8 kernels:
    q * cos + rotate_half(q) * sin    (bf16, rounded after every op)
Both fused kernels replay those operations and roundings exactly
(tests/batch_invariant_ops/test_fused_norm_rope.py):

- RMSNorm is written in Gluon so the bf16 row is loaded and summed in exactly the layout
  `mean_kernel` uses for its fp32 input (BLOCK 1024, 4 contiguous values per thread, 4 warps), which
  makes the reduction tree identical. A plain Triton kernel loads bf16 8-wide and sums in a different
  order (30% of fp32 variances differ).
- Intermediate bf16 roundings use integer bit operations (`_round_bf16`). Written as
  `.to(bf16).to(fp32)`, the compiler drops the round trip in front of an add, which changes results.
- `rsqrt` is libdevice `rsqrt`, which matches `torch.rsqrt` on CUDA bit for bit.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# Must match vexact.batch_invariant_ops.batch_invariant_ops.mean_kernel's launch (BLOCK_SIZE=1024,
# default 4 warps) and the blocked layout Triton assigns to its fp32 loads.
_MEAN_BLOCK = 1024
_MEAN_WARPS = 4


@gluon.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    row_stride,
    N,
    eps,
    BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4], threads_per_warp=[32], warps_per_cta=[NUM_WARPS], order=[0]
    )
    row = gl.program_id(0).to(gl.int64)
    x_row = x_ptr + row * row_stride
    o_row = out_ptr + row * N
    # mean(x^2) exactly as mean_kernel: acc = 0.0; acc += sum(block) per BLOCK; acc / N
    acc = 0.0
    for n0 in range(0, N, BLOCK):
        offs = n0 + gl.arange(0, BLOCK, layout=layout)
        m = offs < N
        x = gl.load(x_row + offs, mask=m, other=0.0).to(gl.float32)
        acc += gl.sum(x * x, axis=0)
    mean = acc / N
    r = gl.rsqrt(mean + eps)
    for n0 in range(0, N, BLOCK):
        offs = n0 + gl.arange(0, BLOCK, layout=layout)
        m = offs < N
        x = gl.load(x_row + offs, mask=m, other=0.0).to(gl.float32)
        w = gl.load(w_ptr + offs, mask=m, other=0.0).to(gl.float32)
        bits = (x * r).to(gl.uint32, bitcast=True)  # round to bf16 (nearest even) with integer ops
        y = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).to(gl.float32, bitcast=True)
        gl.store(o_row + offs, (w * y).to(gl.bfloat16), mask=m)


@gluon.jit
def _rms_norm_rows128_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_rows,
    row_stride,
    N,
    eps,
    ROWS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # For a 128-wide row, mean_kernel's 1024-wide layout puts the whole row in warp 0 (4 contiguous values per
    # lane); every other lane and warp contributes exact zeros. Its summation is therefore: 4 values per lane in
    # order, then a 32-lane butterfly. This layout keeps exactly that per row (one warp per row, 4 values per
    # lane) while packing ROWS rows per program and loading each row once.
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[1, 32], warps_per_cta=[NUM_WARPS, 1], order=[1, 0]
    )
    rows = gl.program_id(0).to(gl.int64) * ROWS + gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    m = (rows < n_rows)[:, None] & (cols < 128)[None, :]
    x = gl.load(x_ptr + rows[:, None] * row_stride + cols[None, :], mask=m, other=0.0).to(gl.float32)
    mean = gl.sum(x * x, axis=1) / N  # mean_kernel: (0.0 + sum) / N, and 0.0 + s == s
    r = gl.rsqrt(mean + eps)
    bits = (x * r[:, None]).to(gl.uint32, bitcast=True)  # round to bf16 (nearest even) with integer ops
    y = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).to(gl.float32, bitcast=True)
    w = gl.load(w_ptr + cols).to(gl.float32)
    gl.store(out_ptr + rows[:, None] * 128 + cols[None, :], (w[None, :] * y).to(gl.bfloat16), mask=m)


_ROWS128_PER_PROGRAM = 16  # 4 warps x 4 rows each


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Bitwise equal to HF eager RMSNorm (bf16 in/out) under vexact batch-invariant mode."""
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    rows = x2.shape[0]
    out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    if rows == 0:
        return out.reshape(x.shape)
    if n == 128:
        _rms_norm_rows128_kernel[(triton.cdiv(rows, _ROWS128_PER_PROGRAM),)](
            x2, weight, out, rows, x2.stride(0), n, eps, ROWS=_ROWS128_PER_PROGRAM, NUM_WARPS=4, num_warps=4
        )
    else:
        _rms_norm_kernel[(rows,)](
            x2, weight, out, x2.stride(0), n, eps, BLOCK=_MEAN_BLOCK, NUM_WARPS=_MEAN_WARPS, num_warps=_MEAN_WARPS
        )
    return out.reshape(x.shape)


@triton.jit
def _round_bf16(x):
    """fp32 -> nearest-even bf16 value, kept in fp32, via integer ops the optimizer cannot fold."""
    bits = x.to(tl.uint32, bitcast=True)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).to(tl.float32, bitcast=True)


@triton.jit
def _rope_token_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    n_heads,
    x_stride_t,
    x_stride_h,
    cs_stride_t,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)  # one token, all heads
    h = tl.arange(0, HEADS)[:, None]
    d = tl.arange(0, HALF)[None, :]
    hm = h < n_heads
    xb = x_ptr + t * x_stride_t + h * x_stride_h + d
    x1 = tl.load(xb, mask=hm, other=0.0).to(tl.float32)
    x2 = tl.load(xb + HALF, mask=hm, other=0.0).to(tl.float32)
    c1 = tl.load(cos_ptr + t * cs_stride_t + d).to(tl.float32)
    c2 = tl.load(cos_ptr + t * cs_stride_t + HALF + d).to(tl.float32)
    s1 = tl.load(sin_ptr + t * cs_stride_t + d).to(tl.float32)
    s2 = tl.load(sin_ptr + t * cs_stride_t + HALF + d).to(tl.float32)
    o1 = _round_bf16(x1 * c1) + _round_bf16(-x2 * s1)
    o2 = _round_bf16(x2 * c2) + _round_bf16(x1 * s2)
    ob = out_ptr + (t * n_heads + h) * 2 * HALF + d
    tl.store(ob, o1.to(tl.bfloat16), mask=hm)
    tl.store(ob + HALF, o2.to(tl.bfloat16), mask=hm)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x * cos + rotate_half(x) * sin for x [tokens, heads, D] and cos/sin [tokens, D] (bf16),
    bitwise equal to the eager expression. Returns a contiguous [tokens, heads, D] tensor."""
    t, h, d = x.shape
    out = torch.empty((t, h, d), dtype=x.dtype, device=x.device)
    if t * h:
        heads = triton.next_power_of_2(h)
        _rope_token_kernel[(t,)](
            x,
            cos,
            sin,
            out,
            h,
            x.stride(0),
            x.stride(1),
            cos.stride(0),
            HEADS=heads,
            HALF=d // 2,
            num_warps=4 if heads * d >= 2048 else 2,
        )
    return out


# --------------------------------------------------------------------------------------------
# Patching HF modules (inference only)
# --------------------------------------------------------------------------------------------


def _rms_supported(n: int) -> bool:
    """Row lengths whose blocks mean_kernel processes like the fused kernel: multiples of its 1024-wide
    block, or a single power-of-two block (checked in tests/batch_invariant_ops/test_fused_norm_rope.py)."""
    return n % _MEAN_BLOCK == 0 or (n <= _MEAN_BLOCK and n & (n - 1) == 0)


def _make_rms_forward(original_forward):
    def forward(self, hidden_states):
        if (
            torch.is_grad_enabled()
            or hidden_states.dtype != torch.bfloat16
            or self.weight.dtype != torch.bfloat16
            or not hidden_states.is_cuda
            or hidden_states.stride(-1) != 1
            or not _rms_supported(hidden_states.shape[-1])
        ):
            return original_forward(self, hidden_states)
        return rms_norm(hidden_states, self.weight, float(self.variance_epsilon))

    forward._vexact_fused = True
    return forward


def _make_rope_fn(original):
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1, **kwargs):
        # Fast path: packed batch (B == 1), q/k [1, heads, T, D], cos/sin [1, T, D] bf16.
        if (
            kwargs
            or torch.is_grad_enabled()
            or unsqueeze_dim != 1
            or q.dim() != 4
            or q.shape[0] != 1
            or k.shape[0] != 1
            or cos.dim() != 3
            or cos.shape[0] != 1
            or not (q.dtype == k.dtype == cos.dtype == sin.dtype == torch.bfloat16)
            or q.stride(-1) != 1
            or k.stride(-1) != 1
            or cos.stride(-1) != 1
            or sin.stride(-1) != 1
            or cos.stride(1) != sin.stride(1)
            or q.shape[-1] % 2
        ):
            return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim, **kwargs)

        def one(x):
            xt = x[0].transpose(0, 1)  # [T, heads, D] view
            out = rope(xt, cos[0], sin[0])  # contiguous [T, heads, D]
            return out.unsqueeze(0).transpose(1, 2)  # [1, heads, T, D], same layout as q from HF

        return one(q), one(k)

    apply_rotary_pos_emb._vexact_fused = True
    return apply_rotary_pos_emb


def apply_fused_norm_rope_patches() -> None:
    """Patch HF Qwen3 / Qwen3-MoE RMSNorm.forward and apply_rotary_pos_emb (no-grad calls only). Idempotent."""
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    for mod, norm_cls in (
        (modeling_qwen3, modeling_qwen3.Qwen3RMSNorm),
        (modeling_qwen3_moe, modeling_qwen3_moe.Qwen3MoeRMSNorm),
    ):
        if not getattr(norm_cls.forward, "_vexact_fused", False):
            norm_cls.forward = _make_rms_forward(norm_cls.forward)
        if not getattr(mod.apply_rotary_pos_emb, "_vexact_fused", False):
            mod.apply_rotary_pos_emb = _make_rope_fn(mod.apply_rotary_pos_emb)
