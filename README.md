# VeXact

Transformer-based bitwise-aligned rollout for FSDP with VeRL integration.

## Key Features

- 🎯 **Bitwise-aligned training & inference** — VeOmni FSDP actor and VeXact rollout engine produce identical logprobs for dense and MoE models with verl (the legacy FSDP engine is not supported)
- ⚡ **Fast and aligned kernels** — Fused MoE, fused linear cross-entropy, Flash Attention with paged KV cache, all numerically consistent between training and inference
- 🧩 **Simple model definitions** — Transformer model code is self-contained and easy to audit, so training and inference model definitions stay in sync
- 📖 **Readable codebase** — Clean implementation with chunked prefill, pipeline parallelism, and CUDA graph support

## Installation

```bash
# uv
uv sync --extra gpu --extra dev
```

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
