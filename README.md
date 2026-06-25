# VeXact

Transformer-based bitwise-aligned rollout for VeOmni FSDP with VeRL integration.

VeXact is our zero-mismatch rollout engine for LLM reinforcement learning. See our paper **[Diagnosing Training-Inference Mismatch in LLM Reinforcement Learning](http://arxiv.org/abs/2605.14220)** for its use as a TIM-free diagnostic baseline.

## Key Features

- 🎯 **Bitwise-aligned training & inference** — VeOmni FSDP actor and VeXact rollout engine produce identical logprobs for dense and MoE models with verl (the legacy FSDP engine is not supported for MoE models).
    - All the dense model should work out-of-the-box if they are not using ops that are different between training and inference like linear attention.
    - MoE models need to patch the model with Fused MoE kernel like our Qwen3-MoE and DeepSeek-V3 example.
- ⚡ **Fast and aligned kernels** — Fused MoE, fused linear cross-entropy, Flash Attention 3/4 with paged KV cache, all numerically consistent between training and inference
- 🧩 **Simple model definitions** — Transformer model code is self-contained and easy to audit, so training and inference model definitions stay in sync
- 📖 **Readable codebase** — Clean implementation with chunked prefill, pipeline parallelism, and CUDA graph support

## Effectiveness

> **Qwen3-30B-A3B · REINFORCE · DAPO dataset**

Off-policy logprob bias from vLLM causes the rollout-correction KL to explode after ~300 steps, which triggers gradient norm blow-up and ultimately training collapse. VeXact's bitwise-aligned rollout keeps the KL at exactly zero throughout, yielding stable training and a ~2× higher final AIME 2024 score.

<table>
  <tr>
    <td align="center"><b>Training reward</b></td>
    <td align="center"><b>AIME 2024 (mean@32)</b></td>
  </tr>
  <tr>
    <td><img src="assets/figures/train_reward.png" width="360"/></td>
    <td><img src="assets/figures/aime24.png" width="360"/></td>
  </tr>
  <tr>
    <td align="center"><b>Rollout-correction K3 KL (log scale)</b></td>
    <td align="center"><b>Gradient norm (log scale)</b></td>
  </tr>
  <tr>
    <td><img src="assets/figures/k3_kl.png" width="360"/></td>
    <td><img src="assets/figures/grad_norm_logy.png" width="360"/></td>
  </tr>
</table>

## Example Recipes

End-to-end RL training scripts live under [`examples/`](examples/README.md),
covering Qwen3-1.7B/30B-A3B and Moonlight-16B-A3B (GRPO/DAPO/REINFORCE) plus
the dense `verify/` pair, on H100/B200. Run any script from the repo root:

```bash
bash examples/getting_started/run_qwen3_1b7.sh
# override paths via env vars
model_dir=/path/to/model data_dir=/path/to/data bash examples/moe/run_qwen3_30B_A3B_dapo.sh
```

See [`examples/README.md`](examples/README.md) for the full recipe list, path
configuration, attention backend selection, and an explanation of the
`verify/` pair.

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

Extras: `gpu` (PyTorch + FlashAttention + kernels), `verl` and `veomni`
(upstreams pinned by commit), `vllm` (alternative rollout engine), `dev`
(test/lint tooling). `verl` and `veomni` are pinned in `pyproject.toml`'s
`[tool.uv.sources]` block; to develop against a local checkout, set
`editable = true` there (inline hints included).

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

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{zhong2026diagnosing,
  title={Diagnosing Training Inference Mismatch in LLM Reinforcement Learning},
  author={Zhong, Tianle and Ling, Neiwen and Pi, Yifan and Wei, Zijun and Yu, Tianshu and Fox, Geoffrey and Wu, Peng and Yu, Xiao},
  journal={arXiv preprint arXiv:2605.14220},
  year={2026}
}
```
