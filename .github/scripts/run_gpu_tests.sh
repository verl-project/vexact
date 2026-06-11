#!/usr/bin/env bash
set -euo pipefail

export VEXACT_TESTS_ATTN_IMPL="${VEXACT_TESTS_ATTN_IMPL:-triton-invariant}"
export TOKENIZERS_PARALLELISM=false

MODEL_ROOT="${RUNNER_TEMP:-/tmp}/vexact-models"
QWEN_MODEL_PATH="${MODEL_ROOT}/Qwen3-1.7B"
MOONLIGHT_MODEL_PATH="${MODEL_ROOT}/Moonlight-16B-A3B-Instruct"

show_disk_usage() {
  local label="$1"
  echo "Disk usage ${label}:"
  df -h "${RUNNER_TEMP:-/tmp}"
  du -sh "${MODEL_ROOT}" 2>/dev/null || true
}

download_model() {
  local model_id="$1"
  local target_dir="$2"
  mkdir -p "${target_dir}"
  env -u HF_ENDPOINT uv run --frozen hf download "${model_id}" --local-dir "${target_dir}"
}

run_verifier_pair() {
  local model_path="$1"
  local output_dir="$2"
  local max_length="$3"
  local max_new_tokens="$4"
  local simulate_requests="$5"
  local max_cache_blocks="${6:-}"
  local skip_backward="${7:-false}"
  local logprobs_from_logits="${8:-false}"

  rm -rf "${output_dir}"
  local inference_cmd=(
    uv run --frozen python tests/scripts/hf_inference.py
    --model_path "${model_path}"
    --attn_impl "${VEXACT_TESTS_ATTN_IMPL}"
    --simulate_requests "${simulate_requests}"
    --request_interval 0.01
    --max_length "${max_length}"
    --max_new_tokens "${max_new_tokens}"
    --max_num_batched_tokens 512
    --enable_batch_invariant
    --enable_chunked_prefill
    --use_fp32_logits
    --output_dir "${output_dir}"
    --seed 1234
  )
  if [[ -n "${max_cache_blocks}" ]]; then
    inference_cmd+=(--max_cache_blocks "${max_cache_blocks}")
  fi
  "${inference_cmd[@]}"

  local verify_cmd=(
    uv run --frozen python tests/scripts/verify_logits_vs_native_hf.py
    --model_path "${model_path}"
    --data_dir "${output_dir}"
    --attn_impl "${VEXACT_TESTS_ATTN_IMPL}"
    --model_backend veomni
    --enable_batch_invariant
    --use_remove_padding
    --rtol 0
    --atol 0
  )
  if [[ "${logprobs_from_logits}" == "true" ]]; then
    verify_cmd+=(--logprobs_from_logits)
  else
    verify_cmd+=(--use_fused_lce)
  fi
  if [[ "${skip_backward}" == "true" ]]; then
    verify_cmd+=(--skip_backward)
  fi
  "${verify_cmd[@]}"
}

show_disk_usage "before model downloads"

echo "Downloading Qwen3-1.7B to ${QWEN_MODEL_PATH}"
download_model "Qwen/Qwen3-1.7B" "${QWEN_MODEL_PATH}"
show_disk_usage "after Qwen3-1.7B download"

export VEXACT_TESTS_MODEL_PATH="${QWEN_MODEL_PATH}"

echo "Running all unit and smoke tests with ${VEXACT_TESTS_ATTN_IMPL}"
uv run --frozen pytest -s -x tests --ignore=tests/perf

echo "Running Qwen3-1.7B VExact/VeOmni bitwise verifier"
run_verifier_pair "${QWEN_MODEL_PATH}" "${RUNNER_TEMP:-/tmp}/qwen-vexact-triton-outputs" 128 16 4

echo "Downloading Moonlight-16B-A3B-Instruct to ${MOONLIGHT_MODEL_PATH}"
download_model "moonshotai/Moonlight-16B-A3B-Instruct" "${MOONLIGHT_MODEL_PATH}"
show_disk_usage "after Moonlight-16B-A3B-Instruct download"

echo "Running Moonlight-16B-A3B-Instruct VExact/VeOmni bitwise verifier"
VEXACT_TESTS_MOE_IMPL=fused_triton run_verifier_pair \
  "${MOONLIGHT_MODEL_PATH}" "${RUNNER_TEMP:-/tmp}/moonlight-vexact-triton-outputs" 64 4 2 32 true true

if [[ "${VEXACT_RUN_VERL_SMOKE:-1}" == "1" ]]; then
  echo "Running Moonlight-16B-A3B-Instruct VeRL smoke test"
  VEOMNI_MOE_IMPLEMENTATION=fused_triton bash .github/scripts/run_verl_smoke.sh "${MOONLIGHT_MODEL_PATH}"
fi
