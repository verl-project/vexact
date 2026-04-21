#!/bin/bash

# Nightly batch-invariance CI entry point. Invoke from the repo root with
# DENSE_MODEL_PATH and MOE_MODEL_PATH set. Set USE_FAKE_MODEL=0 to run the
# e2e verification directly against the original checkpoints (skip the
# 4-layer random-weights truncation used by default).

set -ex

export NCCL_DEBUG=ERROR
export UV_HTTP_TIMEOUT=300

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit
echo "Current directory: $(pwd)"

if [[ ! -f "pyproject.toml" ]]; then
    echo "Error: script must be run from the repo root (pyproject.toml not found in $PWD)"
    exit 1
fi

nvidia-smi

if [[ ! -d verl ]]; then
    git clone https://github.com/verl-project/verl.git verl
fi

uv sync --frozen --extra gpu --extra dev
source .venv/bin/activate

# Unit-level batch invariance kernels.
uv run pytest -s tests/batch_invariant_ops/

# End-to-end batch invariance on dense + MoE backbones (Case 1/2/3 of
# scripts/run_batch_invariant_tests.sh: decode-only, chunked prefill, PP=2).
: "${DENSE_MODEL_PATH:?DENSE_MODEL_PATH must be set to a Qwen3 dense checkpoint}"
: "${MOE_MODEL_PATH:?MOE_MODEL_PATH must be set to a Qwen3 MoE checkpoint}"

USE_FAKE_MODEL=${USE_FAKE_MODEL:-1}
if [[ "$USE_FAKE_MODEL" == "1" ]]; then
    FAKE_ROOT="$(mktemp -d)"
    trap 'rm -rf "$FAKE_ROOT"' EXIT

    DENSE_DIR="$FAKE_ROOT/dense-fake"
    MOE_DIR="$FAKE_ROOT/moe-fake"

    python tests/scripts/init_random_weights.py \
        --model-dir "$DENSE_MODEL_PATH" \
        --output-dir "$DENSE_DIR" \
        --num-layers 4 --verify

    python tests/scripts/init_random_weights.py \
        --model-dir "$MOE_MODEL_PATH" \
        --output-dir "$MOE_DIR" \
        --num-layers 4 --verify
else
    echo "USE_FAKE_MODEL=0 — running e2e against original checkpoints"
    DENSE_DIR="$DENSE_MODEL_PATH"
    MOE_DIR="$MOE_MODEL_PATH"
fi

echo "Running batch invariant tests for dense backbone (FA3)"
export model_dir="$DENSE_DIR"
unset ATTN_IMPL INFER_FA_IMPL
. scripts/run_batch_invariant_tests.sh
echo "✅ Dense FA3 e2e passed"

echo "Running batch invariant tests for MoE backbone (FA3)"
export model_dir="$MOE_DIR"
unset ATTN_IMPL INFER_FA_IMPL
. scripts/run_batch_invariant_tests.sh
echo "✅ MoE FA3 e2e passed"

echo "Running batch invariant tests for dense backbone (FA4 cute)"
export model_dir="$DENSE_DIR"
export ATTN_IMPL=flash_attention_4
export INFER_FA_IMPL=fa-invariant-cute
. scripts/run_batch_invariant_tests.sh
echo "✅ Dense FA4 e2e passed"

echo "Running batch invariant tests for MoE backbone (FA4 cute)"
export model_dir="$MOE_DIR"
export ATTN_IMPL=flash_attention_4
export INFER_FA_IMPL=fa-invariant-cute
. scripts/run_batch_invariant_tests.sh
echo "✅ MoE FA4 e2e passed"
