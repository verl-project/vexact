# Batch-Invariant Ops

Deterministic, batch-invariant operators and Triton kernels for VeXact. These ops guarantee that the output for a given sequence is identical regardless of what other sequences share the same batch,
a requirement for true on-policy RL training where rollout log-probabilities must exactly match actor log-probabilities.

## How It Works

`enable_batch_invariant_mode()` patches PyTorch's ATen dispatch table to replace nondeterministic CUDA kernels with deterministic Triton equivalents:

| ATen op              | Replacement                     | Notes                                                                |
| -------------------- | ------------------------------- | -------------------------------------------------------------------- |
| `aten::mm`           | `matmul_persistent`             | Persistent Triton GEMM with fixed tile schedule                      |
| `aten::addmm`        | `matmul_persistent` (with bias) | Same kernel, bias fused                                              |
| `aten::_log_softmax` | `log_softmax`                   | Row-wise Triton kernel                                               |
| `aten::mean.dim`     | `mean_dim`                      | Single-dim reduction via Triton; multi-dim falls back to `torch.sum` |

It also sets `torch.use_deterministic_algorithms(True)`, disables reduced-precision reductions, and configures `CUBLAS_WORKSPACE_CONFIG`.

### Usage

```python
from vexact.batch_invariant_ops import set_batch_invariant_mode

with set_batch_invariant_mode():
    output = model(input_ids)
```

Or enable/disable explicitly:

```python
from vexact.batch_invariant_ops import enable_batch_invariant_mode, disable_batch_invariant_mode

enable_batch_invariant_mode()
# ... inference ...
disable_batch_invariant_mode()
```

## Module Structure

```
batch_invariant_ops/
├── __init__.py                  # Public API re-exports
├── batch_invariant_ops.py       # Core: ATen overrides, Triton matmul/softmax/mean/bmm/rmsnorm
├── flash_attention.py           # Paged flash attention (FA3 on SM90, FA4/cute on SM90+/SM100)
├── flex_attention.py            # Paged + packed flex attention via torch.nn.attention.flex_attention
├── fused_moe.py                 # Fused MoE with group GEMM (Triton on SM90, quack on SM100)
├── kv_cache_context.py          # Thread-local KV cache context, block manager, Triton store kernel
├── standalone_logprobs.py       # Efficient log-prob computation via flash_attn cross-entropy
├── group_gemm/                  # Triton group GEMM kernels and pre-tuned configs
│   ├── kernel/
│   │   ├── group_gemm.py        # group_gemm_same_mn, group_gemm_same_nk
│   │   ├── moe.py               # expert_histogram, moe_gather, moe_scatter
│   │   └── triton_utils/        # Triton JIT helpers (activation, memory, pid mapping)
│   └── utils/                   # Config loading, device detection, pre-tuned hyperparameters
└── README.md
```

## Key Components

### Attention (`flash_attention.py`, `flex_attention.py`)

Two attention backends, both supporting paged KV cache:

- **Flash Attention** — Uses `flash_attn_with_kvcache` (FA3, SM90) or `flash_attn.cute` (FA4, SM90+/SM100). Requires `is_paged_attn=True` in the KV cache context. Forces `num_splits=1` for determinism.
- **Flex Attention** — Uses `torch.nn.attention.flex_attention` with custom `mask_mod` for causal masking. Supports both packed (non-paged) and paged modes. Paged mode builds a physical-to-logical block mapping and caches the block mask across layers.

Both expect the KV cache context to be set via `set_kv_cache_context()` before the forward pass.

### KV Cache (`kv_cache_context.py`)

Thread-local context system for passing KV cache metadata to attention kernels:

- **`KVCacheContext`** — Dataclass holding cache tensors, block tables, slot mappings, and sequence metadata.
- **`KVCacheManager`** — Block allocator for continuous batching (allocate/free physical blocks).
- **`KVCacheStore`** — Owns per-layer key/value cache tensors. Supports different K/V head dims (e.g., MLA models with `qk_head_dim=192`, `v_head_dim=128`).
- **`store_kvcache`** — Triton kernel for scatter-storing KV pairs into paged cache via slot mapping.

### Fused MoE (`fused_moe.py`)

Batch-invariant Mixture-of-Experts with full forward+backward autograd support:

- **SM90 (Hopper)**: `FusedMoeExpertFunction` — Uses Triton group GEMM kernels from `group_gemm/`.
- **SM100 (Blackwell)**: `QuackFusedMoeExpertFunction` — Uses quack GEMM with `cu_seqlens_m` for variable-length expert groups.

Both paths: dispatch tokens to experts via `moe_scatter`, run gated SiLU MLP (split fc1), gather results via `moe_gather`.

### Standalone Log-Probs (`standalone_logprobs.py`)

Efficient per-token log-probability computation using `flash_attn.ops.triton.cross_entropy`
