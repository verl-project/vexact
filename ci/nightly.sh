#!/bin/bash

# This script is triggered by .codebase/pipelines/nightly.yaml

set -ex

export NCCL_DEBUG=ERROR
export UV_HTTP_TIMEOUT=300

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit
echo "Current directory: $(pwd)"

if [[ ! "$PWD" == */vexact ]]; then
    echo "Error: Script must be run from vexact directory"
    exit 1
fi

nvidia-smi

if [[ ! -d verl ]]; then
    git clone https://github.com/verl-project/verl.git verl
fi

uv sync --frozen --extra gpu --extra dev

uv run pytest -s tests/batch_invariant_ops/
