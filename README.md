# VeXact

Transformer-based bitwise-aligned rollout for FSDP with VeRL integration.

## Key Features

- 🎯 **Bitwise-aligned training & inference** — VeOmni FSDP actor and VeXact rollout engine produce identical logprobs for dense and MoE models with verl (the legacy FSDP engine is not supported)
- ⚡ **Fast and aligned kernels** — Fused MoE, fused linear cross-entropy, Flash Attention with paged KV cache, all numerically consistent between training and inference
- 🧩 **Simple model definitions** — Transformer model code is self-contained and easy to audit, so training and inference model definitions stay in sync
- 📖 **Readable codebase** — Clean implementation with chunked prefill, pipeline parallelism, and CUDA graph support

## Installation

```bash
# pip
pip install -e ".[gpu]"

# uv
uv sync --extra gpu
```

## Quick Start

### Standalone Inference

```python
import asyncio
from vexact.config import ModelConfig, VeXactConfig
from vexact.vexact import VExact
from vexact.request import DriverRequest, GenerationConfig

engine = VExact(VeXactConfig(model=ModelConfig(model_path="/path/to/model")))

prompt_ids = engine.tokenizer.encode("Hello, veXact!")
result = asyncio.run(engine.generate(DriverRequest(
    request_id="req_0",
    generation_config=GenerationConfig(max_new_tokens=128),
    input_ids_list=prompt_ids,
)))

print(engine.tokenizer.decode(result.new_token_ids, skip_special_tokens=True))
engine.close()
```

### Training with VeRL

Example: Qwen3-30B-A3B (MoE) GRPO on 16x H100.

```bash
export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register

python3 -m verl.trainer.main_ppo \
    model_engine=veomni \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.external_lib=vexact.integrations.verl.fsdp_enable_invariant \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.rollout.name=vexact \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=fa-invariant \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    ...
```

See `examples/` for complete recipes (dense and MoE).

## Components

- [`vexact/batch_invariant_ops/`](vexact/batch_invariant_ops/README.md) — batch-invariant operators/kernels for true on-policy RL training.

## Contribution Guide

See [contributions guide](CONTRIBUTING.md).

## Acknowledgements

Besides VeRL and VeOmni, VeXact builds on and is inspired by the following projects:

- [vLLM](https://github.com/vllm-project/vllm) — We refer to vLLM model runner-v2 design and reuse its sampler.
- [batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) — Batch-invariant operators for deterministic inference
- [slime](https://github.com/THUDM/slime) — We refer to its work on determnistic chunked prefill. 
