# Adapted from https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from vexact.batch_invariant_ops.kv_cache_context import get_kv_cache_context, store_kvcache


def is_cuda():
    try:
        return triton.runtime.driver.active.get_current_target().backend == "cuda"
    except RuntimeError:
        return False


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    desc_k,
    desc_v,  #
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        v = v.to(p.dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


configs = [
    triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=1, num_warps=4, pre_hook=_host_descriptor_pre_hook),
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (
        is_cuda()
        and torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M * BLOCK_N < 128 * 128
        and conf.num_warps == 8
    )


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]

    # Filter out configs where BLOCK_M > N_CTX
    # Filter out configs where BLOCK_M < BLOCK_N when causal is True
    return [
        conf
        for conf in configs
        if conf.kwargs.get("BLOCK_M", 0) <= N_CTX
        and (conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0) or STAGE == 1)
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _attn_fwd(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    BF16_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    IS_HOPPER: tl.constexpr,  #
):
    if FP8_OUTPUT:
        dtype = tl.float8e5
    elif BF16_OUTPUT:
        dtype = tl.bfloat16
    else:
        dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM]
    )
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(
            desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1], block_shape=[HEAD_DIM, BLOCK_N]
        )
    else:
        desc_v = _maybe_make_tensor_desc(
            desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM]
        )
    desc_k = _maybe_make_tensor_desc(
        desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM]
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM]
    )

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,  #
            desc_k,
            desc_v,  #
            offset_y,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            IS_HOPPER,
        )
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,  #
            desc_k,
            desc_v,  #
            offset_y,
            dtype,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            IS_HOPPER,
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


@triton.jit
def _attn_bwd_preprocess(
    OUT,
    DO,  #
    Delta,  #
    Z,
    H,
    N_CTX,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(OUT + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    Q,
    k,
    v,
    sm_scale,  #
    DO,  #
    M,
    D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  #
    MASK: tl.constexpr,
    BF16_DOT: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        if BF16_DOT:
            ppT = ppT.to(tl.bfloat16)
        else:
            ppT = ppT.to(tl.float16)
        do_for_dv = do.to(ppT.dtype)
        dv += tl.dot(ppT, do_for_dv)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        do_for_dp = do.to(v.dtype)
        dpT = tl.dot(v, tl.trans(do_for_dp)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        if BF16_DOT:
            dsT = dsT.to(tl.bfloat16)
        else:
            dsT = dsT.to(tl.float16)
        qT = qT.to(dsT.dtype)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,  #
    do,
    m,
    D,
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    num_steps,  #
    MASK: tl.constexpr,
    BF16_DOT: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        do_for_dp = do.to(vT.dtype)
        dp = tl.dot(do_for_dp, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        if BF16_DOT:
            ds = ds.to(tl.bfloat16)
        else:
            ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        kT = kT.to(ds.dtype)
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,  #
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    D,
    # shared by Q/K/V/DO.
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    CAUSAL: tl.constexpr,
    BF16_DOT: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,  #
            Q,
            k,
            v,
            sm_scale,  #
            DO,  #
            M,
            D,  #
            stride_tok,
            stride_d,  #
            H,
            N_CTX,  #
            MASK_BLOCK_M1,
            BLOCK_N1,
            HEAD_DIM,  #
            start_n,
            start_m,
            num_steps,  #
            MASK=True,  #
            BF16_DOT=BF16_DOT,  #
        )

        start_m += num_steps * MASK_BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=False,  #
        BF16_DOT=BF16_DOT,  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    if CAUSAL:
        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _attn_bwd_dq, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(
            dq,
            q,
            K,
            V,  #
            do,
            m,
            D,  #
            stride_tok,
            stride_d,  #
            H,
            N_CTX,  #
            BLOCK_M2,
            MASK_BLOCK_N2,
            HEAD_DIM,  #
            start_m,
            end_n - num_steps * MASK_BLOCK_N2,
            num_steps,  #
            MASK=True,  #
            BF16_DOT=BF16_DOT,  #
        )
        end_n -= num_steps * MASK_BLOCK_N2
        # stage 2
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2

    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        start_n,
        num_steps,  #
        MASK=False,  #
        BF16_DOT=BF16_DOT,  #
    )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Use device_descriptor for Hopper + warpspec.
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(
                    v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1], block_shape=dummy_block
                )
            else:
                desc_v = TensorDescriptor(
                    v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block
                )
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        extra_kern_args = {}
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        _attn_fwd[grid](
            sm_scale,
            M,  #
            q.shape[0],
            q.shape[1],  #
            desc_q,
            desc_k,
            desc_v,
            desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            BF16_OUTPUT=q.dtype == torch.bfloat16,  #
            STAGE=stage,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 1
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o,
            do,  #
            delta,  #
            BATCH,
            N_HEAD,
            N_CTX,  #
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM,  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,  #
            M,
            delta,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            N_HEAD,
            N_CTX,  #
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
            CAUSAL=ctx.causal,  #
            BF16_DOT=q.dtype == torch.bfloat16,  #
        )

        return dq, dk, dv, None, None, None, None


def _require_cuda(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")


def _require_cuda_int32(name: str, tensor: torch.Tensor) -> None:
    _require_cuda(name, tensor)
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")


def _check_common_args(dropout_p: float, softcap: Optional[float], window_size: Optional[tuple[int, int]]) -> None:
    if dropout_p != 0.0:
        raise NotImplementedError("triton flash_attn_varlen_func only supports dropout_p=0.0")
    if softcap is not None:
        raise NotImplementedError("triton flash_attn_varlen_func does not support softcap")
    if window_size is not None and window_size != (-1, -1):
        raise NotImplementedError("triton flash_attn_varlen_func does not support sliding-window attention")


def _default_softmax_scale(q: torch.Tensor, softmax_scale: Optional[float]) -> float:
    if softmax_scale is None:
        return q.shape[-1] ** -0.5
    if isinstance(softmax_scale, torch.Tensor):
        raise ValueError("softmax_scale must be a Python float, not a tensor")
    return float(softmax_scale)


def _as_python_int(value, name: str) -> int:
    if isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a Python int, not a tensor")
    return int(value)


@triton.jit
def _varlen_attn_fwd_block_kernel(
    Q,
    K,
    V,
    OUT,
    LSE,
    CU_SEQLENS_Q,
    CU_SEQLENS_K,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_lt: tl.constexpr,
    stride_lh: tl.constexpr,
    softmax_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_head_dim: tl.constexpr,
    causal: tl.constexpr,
    BF16_OUTPUT: tl.constexpr,
    FP32_OUTPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    seq_id = tl.program_id(0)
    start_m = tl.program_id(1)
    q_head = tl.program_id(2)
    kv_head = q_head // (num_q_heads // num_kv_heads)

    q_start = tl.load(CU_SEQLENS_Q + seq_id)
    q_end = tl.load(CU_SEQLENS_Q + seq_id + 1)
    k_start = tl.load(CU_SEQLENS_K + seq_id)
    k_end = tl.load(CU_SEQLENS_K + seq_id + 1)
    q_len = q_end - q_start
    kv_len = k_end - k_start
    q_abs_start = kv_len - q_len
    if start_m * BLOCK_M >= kv_len or (start_m + 1) * BLOCK_M <= q_abs_start:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    valid_q = (offs_m >= q_abs_start) & (offs_m < kv_len)
    q_local = tl.where(valid_q, offs_m - q_abs_start, 0)
    q = tl.load(
        Q + (q_start + q_local)[:, None] * stride_qt + q_head * stride_qh + offs_d[None, :] * stride_qd,
        mask=valid_q[:, None] & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    if BF16_OUTPUT:
        dtype = tl.bfloat16
    elif FP32_OUTPUT:
        dtype = tl.float32
    else:
        dtype = tl.float16

    qk_scale = softmax_scale * 1.44269504
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    start_n = 0
    while start_n < start_m * BLOCK_M:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_n = offs_n < kv_len
        k_local = tl.where(valid_n, offs_n, 0)
        k = tl.load(
            K + (k_start + k_local)[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :] * stride_kd,
            mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k))
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)

        v_val = tl.load(
            V + (k_start + k_local)[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :] * stride_vd,
            mask=valid_n[:, None] & (offs_d[None, :] < value_head_dim),
            other=0.0,
        )
        p = p.to(dtype)
        v_val = v_val.to(p.dtype)
        acc = tl.dot(p, v_val, acc * alpha[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij
        start_n += BLOCK_N

    start_n = start_m * BLOCK_M
    while start_n < (start_m + 1) * BLOCK_M:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_n = offs_n < kv_len
        k_local = tl.where(valid_n, offs_n, 0)
        k = tl.load(
            K + (k_start + k_local)[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :] * stride_kd,
            mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k))
        if causal:
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        v_val = tl.load(
            V + (k_start + k_local)[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :] * stride_vd,
            mask=valid_n[:, None] & (offs_d[None, :] < value_head_dim),
            other=0.0,
        )
        p = p.to(dtype)
        v_val = v_val.to(p.dtype)
        acc = tl.dot(p, v_val, acc * alpha[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij
        start_n += BLOCK_N

    out = acc / l_i[:, None]
    tl.store(
        OUT + (q_start + q_local)[:, None] * stride_ot + q_head * stride_oh + offs_d[None, :] * stride_od,
        out.to(dtype),
        mask=valid_q[:, None] & (offs_d[None, :] < value_head_dim),
    )
    tl.store(
        LSE + (q_start + q_local) * stride_lt + q_head * stride_lh,
        (m_i + tl.math.log2(l_i)) * 0.6931471824645996,
        mask=valid_q,
    )


@triton.jit
def _varlen_attn_bwd_dq_kernel(
    Q,
    K,
    V,
    OUT,
    DO,
    LSE,
    DQ,
    CU_SEQLENS_Q,
    CU_SEQLENS_K,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_dt: tl.constexpr,
    stride_dh: tl.constexpr,
    stride_dd: tl.constexpr,
    stride_lt: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_dqt: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqd: tl.constexpr,
    softmax_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    seq_id = tl.program_id(0)
    start_m = tl.program_id(1)
    q_head = tl.program_id(2)
    kv_head = q_head // (num_q_heads // num_kv_heads)

    q_start = tl.load(CU_SEQLENS_Q + seq_id)
    q_end = tl.load(CU_SEQLENS_Q + seq_id + 1)
    k_start = tl.load(CU_SEQLENS_K + seq_id)
    k_end = tl.load(CU_SEQLENS_K + seq_id + 1)
    q_len = q_end - q_start
    kv_len = k_end - k_start
    q_abs_start = kv_len - q_len
    if start_m * BLOCK_M >= q_len:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    valid_q = offs_m < q_len
    q_abs = offs_m + q_abs_start
    q = tl.load(
        Q + (q_start + offs_m)[:, None] * stride_qt + q_head * stride_qh + offs_d[None, :] * stride_qd,
        mask=valid_q[:, None] & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    do = tl.load(
        DO + (q_start + offs_m)[:, None] * stride_dt + q_head * stride_dh + offs_d[None, :] * stride_dd,
        mask=valid_q[:, None] & (offs_d[None, :] < value_head_dim),
        other=0.0,
    )
    out = tl.load(
        OUT + (q_start + offs_m)[:, None] * stride_ot + q_head * stride_oh + offs_d[None, :] * stride_od,
        mask=valid_q[:, None] & (offs_d[None, :] < value_head_dim),
        other=0.0,
    )
    lse = tl.load(
        LSE + (q_start + offs_m) * stride_lt + q_head * stride_lh,
        mask=valid_q,
        other=0.0,
    )
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    start_n = 0
    while start_n < kv_len:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_n = offs_n < kv_len
        k_local = tl.where(valid_n, offs_n, 0)
        k = tl.load(
            K + (k_start + k_local)[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :] * stride_kd,
            mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        v_val = tl.load(
            V + (k_start + k_local)[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :] * stride_vd,
            mask=valid_n[:, None] & (offs_d[None, :] < value_head_dim),
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        mask = valid_q[:, None] & valid_n[None, :] & (q_abs[:, None] >= offs_n[None, :])
        p = tl.exp(qk - lse[:, None])
        p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, tl.trans(v_val))
        ds = p * (dp - delta[:, None]) * softmax_scale
        dq += tl.dot(ds.to(k.dtype), k)
        start_n += BLOCK_N

    tl.store(
        DQ + (q_start + offs_m)[:, None] * stride_dqt + q_head * stride_dqh + offs_d[None, :] * stride_dqd,
        dq,
        mask=valid_q[:, None] & (offs_d[None, :] < head_dim),
    )


@triton.jit
def _varlen_attn_bwd_dkv_kernel(
    Q,
    K,
    V,
    OUT,
    DO,
    LSE,
    DK,
    DV,
    CU_SEQLENS_Q,
    CU_SEQLENS_K,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_dt: tl.constexpr,
    stride_dh: tl.constexpr,
    stride_dd: tl.constexpr,
    stride_lt: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_dkt: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvt: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvd: tl.constexpr,
    softmax_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_head_dim: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    seq_id = tl.program_id(0)
    start_n_block = tl.program_id(1)
    kv_head = tl.program_id(2)

    q_start = tl.load(CU_SEQLENS_Q + seq_id)
    q_end = tl.load(CU_SEQLENS_Q + seq_id + 1)
    k_start = tl.load(CU_SEQLENS_K + seq_id)
    k_end = tl.load(CU_SEQLENS_K + seq_id + 1)
    q_len = q_end - q_start
    kv_len = k_end - k_start
    q_abs_start = kv_len - q_len
    start_n = start_n_block * BLOCK_N
    if start_n >= kv_len:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    valid_n = offs_n < kv_len
    k_local = tl.where(valid_n, offs_n, 0)
    k = tl.load(
        K + (k_start + k_local)[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :] * stride_kd,
        mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    v_val = tl.load(
        V + (k_start + k_local)[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :] * stride_vd,
        mask=valid_n[:, None] & (offs_d[None, :] < value_head_dim),
        other=0.0,
    )
    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for group_idx in range(GROUPS):
        q_head = kv_head * GROUPS + group_idx
        start_m = 0
        while start_m < q_len:
            offs_m = start_m + tl.arange(0, BLOCK_M)
            valid_q = offs_m < q_len
            q_abs = offs_m + q_abs_start
            q = tl.load(
                Q + (q_start + offs_m)[:, None] * stride_qt + q_head * stride_qh + offs_d[None, :] * stride_qd,
                mask=valid_q[:, None] & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            do = tl.load(
                DO + (q_start + offs_m)[:, None] * stride_dt + q_head * stride_dh + offs_d[None, :] * stride_dd,
                mask=valid_q[:, None] & (offs_d[None, :] < value_head_dim),
                other=0.0,
            )
            out = tl.load(
                OUT + (q_start + offs_m)[:, None] * stride_ot + q_head * stride_oh + offs_d[None, :] * stride_od,
                mask=valid_q[:, None] & (offs_d[None, :] < value_head_dim),
                other=0.0,
            )
            lse = tl.load(
                LSE + (q_start + offs_m) * stride_lt + q_head * stride_lh,
                mask=valid_q,
                other=0.0,
            )
            delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
            qk = tl.dot(q, tl.trans(k)) * softmax_scale
            mask = valid_q[:, None] & valid_n[None, :] & (q_abs[:, None] >= offs_n[None, :])
            p = tl.exp(qk - lse[:, None])
            p = tl.where(mask, p, 0.0)
            dp = tl.dot(do, tl.trans(v_val))
            ds = p * (dp - delta[:, None]) * softmax_scale
            dv += tl.dot(tl.trans(p).to(do.dtype), do)
            dk += tl.dot(tl.trans(ds).to(q.dtype), q)
            start_m += BLOCK_M

    tl.store(
        DK + (k_start + k_local)[:, None] * stride_dkt + kv_head * stride_dkh + offs_d[None, :] * stride_dkd,
        dk,
        mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
    )
    tl.store(
        DV + (k_start + k_local)[:, None] * stride_dvt + kv_head * stride_dvh + offs_d[None, :] * stride_dvd,
        dv,
        mask=valid_n[:, None] & (offs_d[None, :] < value_head_dim),
    )


@triton.jit
def _paged_attn_fwd_block_kernel(
    Q,
    K,
    V,
    OUT,
    CU_SEQLENS_Q,
    SEQUSED_K,
    PAGE_TABLE,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kp: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vp: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    page_table_stride_b: tl.constexpr,
    page_table_stride_p: tl.constexpr,
    softmax_scale: tl.constexpr,
    page_size: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_head_dim: tl.constexpr,
    causal: tl.constexpr,
    BF16_OUTPUT: tl.constexpr,
    FP32_OUTPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    seq_id = tl.program_id(0)
    start_m = tl.program_id(1)
    q_head = tl.program_id(2)
    kv_head = q_head // (num_q_heads // num_kv_heads)

    q_start = tl.load(CU_SEQLENS_Q + seq_id)
    q_end = tl.load(CU_SEQLENS_Q + seq_id + 1)
    q_len = q_end - q_start
    kv_len = tl.load(SEQUSED_K + seq_id)
    q_abs_start = kv_len - q_len
    if start_m * BLOCK_M >= kv_len or (start_m + 1) * BLOCK_M <= q_abs_start:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    valid_q = (offs_m >= q_abs_start) & (offs_m < kv_len)
    q_local = tl.where(valid_q, offs_m - q_abs_start, 0)
    q = tl.load(
        Q + (q_start + q_local)[:, None] * stride_qt + q_head * stride_qh + offs_d[None, :] * stride_qd,
        mask=valid_q[:, None] & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    if BF16_OUTPUT:
        dtype = tl.bfloat16
    elif FP32_OUTPUT:
        dtype = tl.float32
    else:
        dtype = tl.float16

    qk_scale = softmax_scale * 1.44269504
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Stage 1: off-band blocks, identical to the OAI fused-attention forward.
    start_n = 0
    while start_n < start_m * BLOCK_M:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_n = offs_n < kv_len
        logical_page = tl.where(valid_n, offs_n // page_size, 0)
        page_offset = tl.where(valid_n, offs_n - logical_page * page_size, 0)
        physical_page = tl.load(
            PAGE_TABLE + seq_id * page_table_stride_b + logical_page * page_table_stride_p,
            mask=valid_n,
            other=-1,
        )
        k = tl.load(
            K
            + physical_page[:, None] * stride_kb
            + page_offset[:, None] * stride_kp
            + kv_head * stride_kh
            + offs_d[None, :] * stride_kd,
            mask=valid_n[:, None] & (physical_page[:, None] >= 0) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k))
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)

        v_val = tl.load(
            V
            + physical_page[:, None] * stride_vb
            + page_offset[:, None] * stride_vp
            + kv_head * stride_vh
            + offs_d[None, :] * stride_vd,
            mask=valid_n[:, None] & (physical_page[:, None] >= 0) & (offs_d[None, :] < value_head_dim),
            other=0.0,
        )
        p = p.to(dtype)
        v_val = v_val.to(p.dtype)
        acc = tl.dot(p, v_val, acc * alpha[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij
        start_n += BLOCK_N

    # Stage 2: diagonal/on-band block with causal masking.
    start_n = start_m * BLOCK_M
    while start_n < (start_m + 1) * BLOCK_M:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_n = offs_n < kv_len
        logical_page = tl.where(valid_n, offs_n // page_size, 0)
        page_offset = tl.where(valid_n, offs_n - logical_page * page_size, 0)
        physical_page = tl.load(
            PAGE_TABLE + seq_id * page_table_stride_b + logical_page * page_table_stride_p,
            mask=valid_n,
            other=-1,
        )
        k = tl.load(
            K
            + physical_page[:, None] * stride_kb
            + page_offset[:, None] * stride_kp
            + kv_head * stride_kh
            + offs_d[None, :] * stride_kd,
            mask=valid_n[:, None] & (physical_page[:, None] >= 0) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k))
        if causal:
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        v_val = tl.load(
            V
            + physical_page[:, None] * stride_vb
            + page_offset[:, None] * stride_vp
            + kv_head * stride_vh
            + offs_d[None, :] * stride_vd,
            mask=valid_n[:, None] & (physical_page[:, None] >= 0) & (offs_d[None, :] < value_head_dim),
            other=0.0,
        )
        p = p.to(dtype)
        v_val = v_val.to(p.dtype)
        acc = tl.dot(p, v_val, acc * alpha[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij
        start_n += BLOCK_N

    out = acc / l_i[:, None]
    tl.store(
        OUT + (q_start + q_local)[:, None] * stride_ot + q_head * stride_oh + offs_d[None, :] * stride_od,
        out.to(dtype),
        mask=valid_q[:, None] & (offs_d[None, :] < value_head_dim),
    )


class _PagedFlashAttentionVarlen(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        page_table: torch.Tensor,
        softmax_scale: float,
        causal: bool,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        del max_seqlen_q
        _require_cuda("q", q)
        _require_cuda("k", k)
        _require_cuda("v", v)
        _require_cuda_int32("page_table", page_table)
        _require_cuda_int32("cu_seqlens_q", cu_seqlens_q)
        _require_cuda_int32("seqused_k", seqused_k)

        if k.ndim != 4 or v.ndim != 4:
            raise ValueError("paged flash_attn_varlen_func expects k/v with shape (num_blocks, page_size, heads, dim)")
        if q.ndim != 3:
            raise ValueError("paged flash_attn_varlen_func expects q with shape (total_q, heads, dim)")
        if q.shape[-1] != k.shape[-1]:
            raise ValueError(f"q and k head_dim must match, got {q.shape[-1]} and {k.shape[-1]}")
        if q.shape[1] % k.shape[2] != 0:
            raise ValueError(f"query heads ({q.shape[1]}) must be divisible by KV heads ({k.shape[2]})")

        if cu_seqlens_q.device != q.device:
            raise ValueError(f"cu_seqlens_q must be on {q.device}, got {cu_seqlens_q.device}")
        if seqused_k.device != q.device:
            raise ValueError(f"seqused_k must be on {q.device}, got {seqused_k.device}")
        if page_table.device != q.device:
            raise ValueError(f"page_table must be on {q.device}, got {page_table.device}")
        if not causal:
            raise NotImplementedError("paged triton flash_attn_varlen_func only supports causal=True")
        o = torch.empty((q.shape[0], q.shape[1], v.shape[-1]), device=q.device, dtype=q.dtype)
        block_d = triton.next_power_of_2(max(q.shape[-1], v.shape[-1]))
        if block_d > 256:
            raise ValueError(f"head_dim must be <= 256, got {q.shape[-1]}")

        max_seqlen_k = page_table.shape[1] * k.shape[1]
        grid = (seqused_k.numel(), triton.cdiv(max_seqlen_k, 128), q.shape[1])
        _paged_attn_fwd_block_kernel[grid](
            q,
            k,
            v,
            o,
            cu_seqlens_q,
            seqused_k,
            page_table,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            page_table.stride(0),
            page_table.stride(1),
            softmax_scale,
            k.shape[1],
            q.shape[1],
            k.shape[2],
            q.shape[-1],
            v.shape[-1],
            causal,
            BF16_OUTPUT=q.dtype == torch.bfloat16,
            FP32_OUTPUT=q.dtype == torch.float32,
            BLOCK_M=128,
            BLOCK_N=64,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=3,
        )
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        del ctx, do
        raise NotImplementedError("paged triton flash_attn_varlen_func backward is not implemented")


class _NonpagedFlashAttentionVarlen(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        softmax_scale: float,
        causal: bool,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        _require_cuda_int32("cu_seqlens_q", cu_seqlens_q)
        _require_cuda_int32("cu_seqlens_k", cu_seqlens_k)
        if cu_seqlens_q.device != q.device:
            raise ValueError(f"cu_seqlens_q must be on {q.device}, got {cu_seqlens_q.device}")
        if cu_seqlens_k.device != q.device:
            raise ValueError(f"cu_seqlens_k must be on {q.device}, got {cu_seqlens_k.device}")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError(
                "non-paged flash_attn_varlen_func expects q/k/v with shape (total_tokens, heads, head_dim)"
            )
        if q.shape[-1] != k.shape[-1]:
            raise ValueError(f"q and k head_dim must match, got {q.shape[-1]} and {k.shape[-1]}")
        if k.shape[-2] != v.shape[-2]:
            raise ValueError(f"k/v must have the same number of KV heads, got {k.shape[-2]} and {v.shape[-2]}")
        if q.shape[-2] % k.shape[-2] != 0:
            raise ValueError(f"query heads ({q.shape[-2]}) must be divisible by KV heads ({k.shape[-2]})")
        if not causal:
            raise NotImplementedError("non-paged triton flash_attn_varlen_func only supports causal=True")

        out = torch.zeros((q.shape[0], q.shape[1], v.shape[-1]), device=q.device, dtype=q.dtype)
        lse = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
        block_d = triton.next_power_of_2(max(q.shape[-1], v.shape[-1]))
        if block_d > 128:
            raise ValueError(f"head_dim must be <= 128, got {q.shape[-1]}")

        grid = (cu_seqlens_q.numel() - 1, triton.cdiv(max(max_seqlen_q, max_seqlen_k), 128), q.shape[1])
        _varlen_attn_fwd_block_kernel[grid](
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            softmax_scale,
            q.shape[1],
            k.shape[1],
            q.shape[-1],
            v.shape[-1],
            causal,
            BF16_OUTPUT=q.dtype == torch.bfloat16,
            FP32_OUTPUT=q.dtype == torch.float32,
            BLOCK_M=128,
            BLOCK_N=64,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=3,
        )

        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        if not ctx.causal:
            raise NotImplementedError("non-paged triton flash_attn_varlen_func backward only supports causal=True")
        if do.stride(-1) != 1:
            do = do.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        block_d = triton.next_power_of_2(max(q.shape[-1], v.shape[-1]))
        if block_d > 128:
            raise ValueError(f"head_dim must be <= 128, got {q.shape[-1]}")
        block_m = 64 if block_d <= 64 else 32
        block_n = 64 if block_d <= 64 else 32
        groups = q.shape[1] // k.shape[1]
        num_warps = 4 if block_d <= 64 else 8

        dq_grid = (cu_seqlens_q.numel() - 1, triton.cdiv(ctx.max_seqlen_q, block_m), q.shape[1])
        _varlen_attn_bwd_dq_kernel[dq_grid](
            q,
            k,
            v,
            out,
            do,
            lse,
            dq,
            cu_seqlens_q,
            cu_seqlens_k,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            lse.stride(0),
            lse.stride(1),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            ctx.softmax_scale,
            q.shape[1],
            k.shape[1],
            q.shape[-1],
            v.shape[-1],
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=3,
        )

        dkv_grid = (cu_seqlens_k.numel() - 1, triton.cdiv(ctx.max_seqlen_k, block_n), k.shape[1])
        _varlen_attn_bwd_dkv_kernel[dkv_grid](
            q,
            k,
            v,
            out,
            do,
            lse,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            lse.stride(0),
            lse.stride(1),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            ctx.softmax_scale,
            q.shape[1],
            k.shape[1],
            q.shape[-1],
            v.shape[-1],
            groups,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=3,
        )
        return dq, dk, dv, None, None, None, None, None, None


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[tuple[int, int]] = None,
    softcap: Optional[float] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """FlashAttention-compatible varlen interface.

    ``page_table`` and ``block_table`` are aliases.  Without either alias, this
    function uses the oai Triton autograd kernel.  With a page table it uses
    the Triton paged forward kernel and raises on backward.
    """
    del alibi_slopes, deterministic, return_attn_probs, kwargs
    _check_common_args(dropout_p, softcap, window_size)
    if page_table is not None and block_table is not None:
        raise ValueError("pass only one of page_table or block_table")

    scale = _default_softmax_scale(q, softmax_scale)
    active_page_table = page_table if page_table is not None else block_table
    if active_page_table is None:
        if cu_seqlens_q is None:
            raise ValueError("cu_seqlens_q is required when page_table/block_table is not provided")
        if cu_seqlens_k is None:
            raise ValueError("cu_seqlens_k is required when page_table/block_table is not provided")
        if max_seqlen_k is None:
            raise ValueError("max_seqlen_k is required when page_table/block_table is not provided")
        if max_seqlen_q is None:
            raise ValueError("max_seqlen_q is required when page_table/block_table is not provided")
        out = _NonpagedFlashAttentionVarlen.apply(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            scale,
            causal,
            _as_python_int(max_seqlen_q, "max_seqlen_q"),
            _as_python_int(max_seqlen_k, "max_seqlen_k"),
        )
        return out, None

    active_seqused_k = seqused_k if seqused_k is not None else cache_seqlens
    if active_seqused_k is None:
        raise ValueError("seqused_k or cache_seqlens is required when page_table/block_table is provided")
    if max_seqlen_q is None:
        max_seqlen_q = 0

    out = _PagedFlashAttentionVarlen.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        active_seqused_k,
        active_page_table,
        scale,
        causal,
        _as_python_int(max_seqlen_q, "max_seqlen_q"),
    )
    return out, None


def flash_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Open-VeXact attention wrapper that delegates to ``flash_attn_varlen_func``."""
    kv_context = get_kv_cache_context()

    query = query.transpose(1, 2).contiguous()
    if key is not None:
        key = key.transpose(1, 2).contiguous()
    if value is not None:
        value = value.transpose(1, 2).contiguous()

    _, _, num_attention_heads, head_dim = query.shape
    layer_idx = _as_python_int(
        kwargs.get("layer_idx") if "layer_idx" in kwargs else getattr(module, "layer_idx", 0),
        "layer_idx",
    )

    if kv_context is None:
        if key is None or value is None:
            raise ValueError("key and value are required for non-paged triton attention")
        batch_size, q_len = query.shape[:2]
        cu_seqlens_q = kwargs.get(
            "cu_seq_lens_q",
            kwargs.get("cu_seqlens_q", kwargs.get("cu_seq_lens", kwargs.get("cu_seqlens", None))),
        )
        cu_seqlens_k = kwargs.get(
            "cu_seq_lens_k",
            kwargs.get("cu_seqlens_k", kwargs.get("cu_seq_lens", kwargs.get("cu_seqlens", None))),
        )
        max_seqlen_q = kwargs.get("max_length_q", kwargs.get("max_seqlen_q", kwargs.get("max_seqlen", None)))
        max_seqlen_k = kwargs.get("max_length_k", kwargs.get("max_seqlen_k", kwargs.get("max_seqlen", None)))
        if cu_seqlens_q is None:
            raise ValueError("cu_seq_lens_q/cu_seqlens_q is required for non-paged triton-invariant attention")
        if cu_seqlens_k is None:
            raise ValueError("cu_seq_lens_k/cu_seqlens_k is required for non-paged triton-invariant attention")
        if max_seqlen_q is None:
            raise ValueError("max_length_q/max_seqlen_q is required for non-paged triton-invariant attention")
        if max_seqlen_k is None:
            raise ValueError("max_length_k/max_seqlen_k is required for non-paged triton-invariant attention")

        query_flat = query.reshape(-1, num_attention_heads, head_dim)
        key_flat = key.reshape(-1, key.shape[2], key.shape[-1])
        value_flat = value.reshape(-1, value.shape[2], value.shape[-1])
        attn_out, _ = flash_attn_varlen_func(
            query_flat,
            key_flat,
            value_flat,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=kwargs.get("is_causal", getattr(module, "is_causal", True)),
        )
        return attn_out.reshape(batch_size, q_len, num_attention_heads, -1), None

    if kv_context.is_paged_attn:
        key_cache_blocks = kv_context.key_cache[layer_idx]
        value_cache_blocks = kv_context.value_cache[layer_idx]
        if key is not None and value is not None:
            num_key_value_heads = key.shape[2]
            key_flat = key.reshape(-1, num_key_value_heads, head_dim)
            value_flat = value.reshape(-1, num_key_value_heads, value.shape[-1])
            store_kvcache(key_flat, value_flat, key_cache_blocks, value_cache_blocks, kv_context.slot_mapping)

        query_flat = query.reshape(-1, num_attention_heads, head_dim)
        return flash_attn_varlen_func(
            query_flat,
            key_cache_blocks,
            value_cache_blocks,
            cu_seqlens_q=kv_context.query_start_loc,
            cu_seqlens_k=None,
            max_seqlen_q=kv_context.max_seqlen_q,
            max_seqlen_k=None,
            seqused_k=kv_context.context_lens,
            page_table=kv_context.block_tables,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=True,
        )

    if key is None or value is None:
        raise ValueError("key and value are required for non-paged triton attention")
    query_flat = query.reshape(-1, num_attention_heads, head_dim)
    key_flat = key.reshape(-1, key.shape[2], key.shape[-1])
    value_flat = value.reshape(-1, value.shape[2], value.shape[-1])
    return flash_attn_varlen_func(
        query_flat,
        key_flat,
        value_flat,
        cu_seqlens_q=kv_context.query_start_loc,
        cu_seqlens_k=kv_context.query_start_loc,
        max_seqlen_q=kv_context.max_seqlen_q,
        max_seqlen_k=kv_context.max_seqlen_q,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True,
    )


__all__ = ["flash_attn_varlen_func", "flash_attention_forward"]
