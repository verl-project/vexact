#!/bin/bash

# This script is triggered by .codebase/pipelines/nightly.yaml

set -ex

export NCCL_DEBUG=ERROR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit
echo "Current directory: $(pwd)"

if [[ ! "$PWD" == */vexact ]]; then
    echo "Error: Script must be run from vexact directory"
    exit 1
fi

nvidia-smi

uv sync --frozen --extra gpu --extra dev

uv run pytest -s --cov=vexact tests/batch_invariant_ops/
