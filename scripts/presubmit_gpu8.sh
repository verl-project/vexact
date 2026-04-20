#!/bin/bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 8xGPU presubmit suite: unit tests + batch-invariant e2e verification on
# dense (Qwen3-1.7B) and MoE (Qwen3-30B-A3B) backbones with FA3 and FA4.
#
# Required environment variables:
#   DENSE_MODEL_PATH - path to a Qwen3 dense checkpoint (e.g. Qwen3-1.7B)
#   MOE_MODEL_PATH   - path to a Qwen3 MoE checkpoint   (e.g. Qwen3-30B-A3B-Instruct-2507)
#
# Usage:
#   DENSE_MODEL_PATH=/path/to/Qwen3-1.7B \
#   MOE_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507 \
#   bash scripts/presubmit_gpu8.sh

set -ex

# Suppress prolonged NCCL logs.
export NCCL_DEBUG=ERROR
# Disable Liger kernels for VeOmni.
export VEOMNI_USE_LIGER_KERNEL=0

: "${DENSE_MODEL_PATH:?must be set to a dense Qwen3 checkpoint}"
: "${MOE_MODEL_PATH:?must be set to a Qwen3 MoE checkpoint}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1
if [[ ! -f "pyproject.toml" ]]; then
    echo "Error: script must be run from the repo root (pyproject.toml not found in $PWD)"
    exit 1
fi

uv sync --frozen --extra gpu --extra dev
source .venv/bin/activate

nvidia-smi | head -5

# ---------------------------------------------------------------------------
# 1. Unit tests
# ---------------------------------------------------------------------------
export VEXACT_TESTS_MODEL_PATH="$DENSE_MODEL_PATH"

rm -f .coverage
pytest -s --cov=vexact --cov-append tests/test_*
pytest -s --cov=vexact --cov-append tests/batch_invariant_ops/
echo "✅ Unit tests passed"

# ---------------------------------------------------------------------------
# 2. Model-level batch-invariance verification
# ---------------------------------------------------------------------------
FAKE_ROOT="$(mktemp -d)"
trap 'rm -rf "$FAKE_ROOT"' EXIT

DENSE_FAKE="$FAKE_ROOT/dense-fake"
MOE_FAKE="$FAKE_ROOT/moe-fake"

python tests/scripts/init_random_weights.py \
    --model-dir "$DENSE_MODEL_PATH" \
    --output-dir "$DENSE_FAKE" \
    --num-layers 4 --verify

python tests/scripts/init_random_weights.py \
    --model-dir "$MOE_MODEL_PATH" \
    --output-dir "$MOE_FAKE" \
    --num-layers 4 --verify

echo "Running batch invariant tests for dense backbone (FA3)"
export model_dir="$DENSE_FAKE"
unset ATTN_IMPL INFER_FA_IMPL
. scripts/run_batch_invariant_tests.sh
echo "✅ Dense FA3 e2e passed"

echo "Running batch invariant tests for MoE backbone (FA3)"
export model_dir="$MOE_FAKE"
unset ATTN_IMPL INFER_FA_IMPL
. scripts/run_batch_invariant_tests.sh
echo "✅ MoE FA3 e2e passed"

echo "Running batch invariant tests for dense backbone (FA4 cute)"
export model_dir="$DENSE_FAKE"
export ATTN_IMPL=flash_attention_4
export INFER_FA_IMPL=fa-invariant-cute
. scripts/run_batch_invariant_tests.sh
echo "✅ Dense FA4 e2e passed"

# ---------------------------------------------------------------------------
# 3. Coverage
# ---------------------------------------------------------------------------
coverage report || true
