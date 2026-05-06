# VeXact

Transformer-based bitwise-aligned rollout for FSDP with VeRL integration.

## Key Features

- 🎯 **Bitwise-aligned training & inference** — VeOmni FSDP actor and VeXact rollout engine produce identical logprobs for dense and MoE models with verl (the legacy FSDP engine is not supported)
- ⚡ **Fast and aligned kernels** — Fused MoE, fused linear cross-entropy, Flash Attention with paged KV cache, all numerically consistent between training and inference
- 🧩 **Simple model definitions** — Transformer model code is self-contained and easy to audit, so training and inference model definitions stay in sync
- 📖 **Readable codebase** — Clean implementation with chunked prefill, pipeline parallelism, and CUDA graph support

## Installation

VeXact uses [uv](https://docs.astral.sh/uv/) for environment management. Pick
the extras that match your use case:

```bash
# End-to-end RL training (verl trainer + VeOmni FSDP actor + VeXact rollout):
uv sync --extra gpu --extra verl --extra veomni

# Rollout-only (no trainer, no FSDP actor):
uv sync --extra gpu

# Add the dev extra (pytest, pre-commit) when contributing:
uv sync --extra gpu --extra verl --extra veomni --extra dev
```

What each extra does:

- `gpu` — PyTorch (CUDA 12.9), FlashAttention 2/3/4, quack-kernels, NVML.
- `verl` — pulls verl from `verl-project/verl` (pinned by commit in
  `[tool.uv.sources]`) plus FastAPI/uvicorn/cachetools used by the trainer.
- `veomni` — pulls VeOmni from `ByteDance-Seed/VeOmni` (pinned by commit).
- `vllm` — vLLM 0.18 if you prefer it as the rollout engine instead of
  VeXact's native one.
- `dev` — `pytest`, `pytest-asyncio`, `pre-commit` for development.

### Working on verl or VeOmni locally

`verl` and `veomni` are pinned by git commit in `pyproject.toml`'s
`[tool.uv.sources]` block, so contributors and CI all resolve to the same
upstream. To hack on either upstream against your local checkout, swap the
relevant entry to `editable = true` (the file has inline hints):

```toml
[tool.uv.sources]
verl = { path = "./verl", editable = true }
veomni = { path = "./VeOmni", editable = true }
```

Then `uv sync --extra gpu --extra verl --extra veomni` re-resolves the venv
to your local tree.

## Components

- [`vexact/batch_invariant_ops/`](vexact/batch_invariant_ops/README.md) — batch-invariant operators/kernels for true on-policy RL training.

## Contribution Guide

See [contributions guide](CONTRIBUTING.md).

## Acknowledgements

Besides VeRL and VeOmni, VeXact builds on and is inspired by the following projects:

- [vLLM](https://github.com/vllm-project/vllm) — We refer to vLLM model runner-v2 design and reuse its sampler.
- [batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) — Batch-invariant operators for deterministic inference
- [Torch Memory Saver](https://github.com/fzyzcjy/torch_memory_saver) - Model param and KV cache offloads.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - We support FA4 for SM90+ (including SM100) GPU, including MLA shape for DeepSeek-V3 model architecture.
