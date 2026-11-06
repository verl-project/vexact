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
"""Launch-config tuning for VeOmni's Triton grouped GEMM (the fused-MoE expert GEMMs).

VeOmni ships no pre-tuned configs, so every call uses the fallback tile (BLOCK_M=128, BLOCK_N=128,
BLOCK_K=32, 4 warps, 3 stages). On H100 that leaves the decode-time expert GEMMs 1.1-1.5x slower
than streaming the active experts' weights.

The wrapper below launches the *same* JIT kernel with a tile chosen from the number of routed rows.
Results are bitwise identical to the fallback: the kernel accumulates K sequentially from zero
for every tile shape, and on H100 neither the tile nor the emitted MMA instruction changes the bits
for a fixed K order (tests/batch_invariant_ops/test_moe_group_gemm_tuning.py). Training and
rollout therefore stay aligned whichever side applies it. Only the forward call pattern
(``transpose_b=True``, no activation, no accumulation, bf16, contiguous) is retuned; every other
call goes to VeOmni's original function unchanged.

Above ``_OGS_MIN_ROWS`` routed rows (prefill and mixed steps) the GEMM instead runs triton_kernels'
ragged ``matmul_ogs`` (persistent, TMA, warp-specialized), which reaches ~400-530 TFLOP/s where
VeOmni's kernel stays near 200. It is used with no gather/scatter/activation fusion and with split-K
pinned to 1 (its default split-K depends on the batch and would change the reduction tree), so it too
accumulates K sequentially from zero and returns the same bits (same test).
"""

import torch
import triton

from vexact.batch_invariant_ops.batch_invariant_ops import _max_shared_mem


# Routed rows (tokens x top-k) above which matmul_ogs is used. Below it both kernels stream expert
# weights at the same rate; above it matmul_ogs is 1.2-2.6x faster on H100.
_OGS_MIN_ROWS = 2048


def _load_matmul_ogs():
    """Return a callable (a, w_ekn, splits) -> out using triton_kernels' ragged matmul_ogs."""
    from triton_kernels.matmul_ogs import RoutingData, matmul_ogs
    from triton_kernels.matmul_ogs_details.opt_flags import update_opt_flags_constraints
    from triton_kernels.tensor import make_ragged_tensor_metadata

    # Batch-dependent split-K would change the reduction tree; keep one CTA per output tile.
    update_opt_flags_constraints({"split_k": 1})

    def run(a, w_ekn, splits):
        rows = a.shape[0]
        routing = RoutingData(None, splits, splits.shape[0], 1, make_ragged_tensor_metadata(splits, rows))
        return matmul_ogs(a, w_ekn, None, routing)

    return run


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages), fastest on H100 for routed rows (tokens x top-k)
# <= 128, <= 512, <= 4096 and above. VeOmni's own default tile fits any GPU and is tried last.
_CONFIGS = ((64, 256, 64, 8, 4), (64, 128, 128, 4, 4), (64, 128, 64, 8, 3), (128, 128, 32, 4, 4))
_VEOMNI_DEFAULT = (128, 128, 32, 4, 3)


def _ranked_configs(max_m: int) -> tuple:
    """Configs in order of preference for ``max_m`` routed rows."""
    best = 0 if max_m <= 128 else 1 if max_m <= 512 else 2 if max_m <= 4096 else 3
    return (_CONFIGS[best], *(c for i, c in enumerate(_CONFIGS) if i != best), _VEOMNI_DEFAULT)


def _launch_config(cfg: tuple) -> dict:
    block_m, block_n, block_k, warps, stages = cfg
    return dict(BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=warps, num_stages=stages)


def apply_moe_group_gemm_tuning() -> None:
    """Patch VeOmni's fused-MoE module to use tuned launches. Idempotent."""
    from veomni.ops.kernels.moe import group_gemm as fused_moe_module
    from veomni.ops.kernels.moe._kernels.kernel import group_gemm as kernel_module

    if getattr(fused_moe_module.group_gemm_same_nk, "_vexact_tuned", False):
        return
    original = kernel_module.group_gemm_same_nk
    jit_kernel = kernel_module.group_gemm_same_nk_kernel.fn  # below VeOmni's pretuned-config layer
    chosen: dict[tuple, tuple] = {}

    def choose_config(max_m, launch_kwargs):
        """The most preferred config whose compiled kernel fits the device's shared memory (the decode tile
        needs 120 KB on Ada, e.g. L20 with 99 KB). Compiled without launching, once per shape."""
        key = (_ranked_configs(max_m)[0], *(launch_kwargs[k] for k in ("G", "N", "K")), launch_kwargs["a_ptr"].device)
        if key not in chosen:
            limit = _max_shared_mem(launch_kwargs["a_ptr"].device.index)
            for cfg in _ranked_configs(max_m):
                kernel = jit_kernel.warmup(**launch_kwargs, **_launch_config(cfg), grid=(1,))
                if kernel.metadata.shared <= limit:
                    chosen[key] = cfg
                    break
            else:
                raise RuntimeError(f"No grouped-GEMM config fits the device's shared memory for {key}")
        return chosen[key]

    ragged_matmul = _load_matmul_ogs()

    def group_gemm_same_nk(
        a,
        b,
        cumsum_M,
        max_M,
        transpose_a=False,
        transpose_b=False,
        activation=None,
        save_activation=False,
        c=None,
    ):
        if (
            transpose_a
            or not transpose_b
            or activation is not None
            or save_activation
            or c is not None
            or a.dtype != torch.bfloat16
            or b.dtype != torch.bfloat16
            or not (a.is_contiguous() and b.is_contiguous())
            or len(cumsum_M) != b.shape[0]
        ):
            return original(
                a,
                b,
                cumsum_M,
                max_M,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                activation=activation,
                save_activation=save_activation,
                c=c,
            )
        G, N, K = b.shape
        if a.shape[0] == max_M and max_M > _OGS_MIN_ROWS:
            # Per-expert row counts from VeOmni's inclusive cumsum (device-side, CUDA-graph safe).
            splits = torch.diff(cumsum_M, prepend=cumsum_M.new_zeros(1)).to(torch.int32)
            return ragged_matmul(a, b.transpose(1, 2), splits)  # [E, K, N] column-major view
        out = torch.empty((a.shape[0], N), dtype=a.dtype, device=a.device)
        launch_kwargs = dict(
            a_ptr=a,
            b_ptr=b,
            c_ptr=out,
            act_ptr=None,
            cumsum_M=cumsum_M,
            max_M=max_M,
            total_M=a.shape[0],
            G=G,
            K=K,
            N=N,
            TRANSPOSE_A=False,
            TRANSPOSE_B=True,
            ACCUMULATE_TO_C=False,
            ACTIVATION=None,
            SAVE_ACTIVATION=False,
            GROUP=8,
        )
        with torch.cuda.device(a.device):
            cfg = choose_config(max_M, launch_kwargs)
            jit_kernel[(triton.cdiv(max_M, cfg[0]) * triton.cdiv(N, cfg[1]), G)](**launch_kwargs, **_launch_config(cfg))
        return out

    group_gemm_same_nk._vexact_tuned = True
    fused_moe_module.group_gemm_same_nk = group_gemm_same_nk
