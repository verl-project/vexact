#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?Usage: run_verl_smoke.sh MODEL_PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK_ROOT="${RUNNER_TEMP:-/tmp}/vexact-verl-smoke"
DATA_DIR="${WORK_ROOT}/gsm8k"
LOG_FILE="${WORK_ROOT}/verl-smoke.log"

mkdir -p "${WORK_ROOT}"

if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/test.parquet" ]]; then
  rm -rf "${DATA_DIR}"
  env -u HF_ENDPOINT uv run --frozen hf download verl-team/gsm8k-v0.4.1 \
    --repo-type dataset \
    --include '*.parquet' \
    --local-dir "${DATA_DIR}"
fi

cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  source "${REPO_ROOT}/.venv/bin/activate"
fi

export MODEL_PATH
export DATA_PATH="${DATA_DIR}"
export INFER_FA_IMPL="${VEXACT_TESTS_ATTN_IMPL:-triton-invariant}"
export VEOMNI_ATTN_IMPL="${VEOMNI_ATTN_IMPL:-flash_attention_2}"
export VEXACT_MAX_CACHE_BLOCKS="${VEXACT_MAX_CACHE_BLOCKS:-512}"

PIPELINE_PARALLELISM="${VEXACT_VERL_PIPELINE_PARALLELISM:-8}"

MODEL_PATH="${MODEL_PATH}" DATA_PATH="${DATA_PATH}" RAY_DEDUP_LOGS=1 PYTHONUNBUFFERED=1 \
  bash examples/moe/run_moonlight_gsm8k.sh \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=128 \
    data.max_response_length=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=512 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size="${PIPELINE_PARALLELISM}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=512 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    trainer.log_val_generations=0 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=1 \
    trainer.val_before_train=False \
    trainer.logger=[console] 2>&1 | tee "${LOG_FILE}"

uv run --frozen python - "${LOG_FILE}" <<'PY'
import math
import re
import sys

log_file = sys.argv[1]
text = open(log_file, encoding="utf-8").read()
matches = re.findall(r"training/rollout_probs_diff_max['\"]?\s*[:=]\s*([-+0-9.eE]+|nan)", text)
if not matches:
    raise SystemExit("training/rollout_probs_diff_max was not found in the VeRL smoke log")

value = float(matches[-1])
if math.isnan(value) or value != 0.0:
    raise SystemExit(f"Expected training/rollout_probs_diff_max == 0.0, got {value}")

print("VeRL smoke metric check passed: training/rollout_probs_diff_max == 0.0")
PY
