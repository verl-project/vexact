# Examples

End-to-end RL training recipes using VeXact as the rollout engine (with
[verl](https://github.com/volcengine/verl) as the RL framework, and optionally
[VeOmni](https://github.com/ByteDance-Seed/VeOmni) as the training engine).

## Layout

```
examples/
├── getting_started/      # minimum viable run
├── moe/                  # MoE training recipes (algorithm / hardware variants)
├── verify/               # vexact vs vllm determinism check
└── math_reward_model/    # reward functions used by the recipes above
```

## Recipe index

| Recipe                               | Model                         | Dataset (train / val)                                                                          | Hardware | Algorithm     | Notes                               |
| ------------------------------------ | ----------------------------- | ---------------------------------------------------------------------------------------------- | -------- | ------------- | ----------------------------------- |
| `getting_started/run_qwen3_1b7.sh`   | Qwen3-1.7B (dense)            | gsm8k                                                                                          | 1× 8H100 | GRPO          | Smallest entry point                |
| `moe/run_qwen3_30B_A3B_dapo.sh`      | Qwen3-30B-A3B                 | DAPO-Math-17k / AIME 2025                                                                      | 1× 8H100 | DAPO          | Long-context math RL (20k response) |
| `moe/run_qwen3_30B_A3B_16H100.sh`    | Qwen3-30B-A3B                 | gsm8k                                                                                          | 2× 8H100 | GRPO          | Multi-node test                     |
| `moe/run_qwen3_30B_A3B_8B200.sh`     | Qwen3-30B-A3B                 | gsm8k                                                                                          | 1× 8B200 | GRPO          | FA4 + fused_quack MoE kernel        |
| `moe/run_qwen3_30B_A3B_reinforce.sh` | Qwen3-30B-A3B-Base            | DAPO-Math-17k / AIME 2024                                                                      | 8× 8H100 | REINFORCE++   | Algorithm diversity                 |
| `verify/run_dense_vexact.sh`         | DeepSeek-R1-Distill-Qwen-1.5B | MATH ([math_1460](https://huggingface.co/datasets/sail/Sanity-Test-R1D-1.5B)) / AIME 2024+2025 | 1× 8H100 | GRPO (vexact) | Pair with the vllm one below        |
| `verify/run_dense_vllm.sh`           | DeepSeek-R1-Distill-Qwen-1.5B | MATH ([math_1460](https://huggingface.co/datasets/sail/Sanity-Test-R1D-1.5B)) / AIME 2024+2025 | 1× 8H100 | GRPO (vllm)   | Baseline for determinism check      |

## Running a recipe

All vexact scripts assume they are launched **from the repo root** (`vexact/`)

```bash
cd /path/to/vexact
bash examples/getting_started/run_qwen3_1b7.sh
```

### Setting model and dataset paths

The defaults point to the Arnold-style mount `/mnt/hdfs/model_path` and
`/mnt/hdfs/data_path`. Override via environment variables or by editing the
script:

```bash
model_dir=/path/to/Qwen3-1.7B \
data_dir=/path/to/gsm8k \
bash examples/getting_started/run_qwen3_1b7.sh
```

Datasets are expected in parquet format. See each recipe for the specific
splits it loads.

### Picking an attention backend

For H100 (SM90), the batch-invariant FA3 kernel is the default:

```bash
export INFER_FA_IMPL=fa-invariant   # FA3, H100
```

On B200 (SM100+) the FA4 CUTE kernel is used:

```bash
export INFER_FA_IMPL=fa-invariant-cute   # FA4, B200
```

## `verify/` — batch-invariant check

`run_dense_vexact.sh` and `run_dense_vllm.sh` share identical hyperparameters;
they differ only in `actor_rollout_ref.rollout.name` (`vexact` vs `vllm`). The
vllm recipe uses **native vLLM**, not in batch-invariant mode.

Running both side by side and comparing the wandb figures illustrates why a
batch-invariant rollout engine matters for on-policy RL:

- **`run_dense_vllm.sh`** — training reward **collapses** mid-run, and
    `rollout_probs_diff_mean` (the gap between rollout log-probs and the actor's
    recomputed log-probs on the same tokens) stays noticeably high. Native vLLM
    is not batch-invariant, so its rollout distribution drifts from what the
    actor would have produced.
- **`run_dense_vexact.sh`** — reward keeps improving and
    `rollout_probs_diff_mean` stays zero.

## `math_reward_model/`

Custom reward functions used by the math-RL recipes:

- `math_grader.py` — dense models.
- `math_grader_moe.py` — MoE models.
- `math_utils.py` — shared math expression normalization / comparison.

Both graders expose `compute_math_score`, which the recipes wire up via
`custom_reward_function.name=compute_math_score`.

## Acknowledgements

The dense experiments under `verify/` and the reward model in
`math_reward_model/` are adapted from
[sail-sg/Precision-RL-verl](https://github.com/sail-sg/Precision-RL-verl).
