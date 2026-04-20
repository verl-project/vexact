#!/usr/bin/env python3
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

"""
Shared pytest fixtures for VeXact tests.
"""

import os
from pathlib import Path

import pytest

from vexact.config import ModelConfig, ParallelConfig, SchedulerConfig, VeXactConfig
from vexact.engine import VeXact


@pytest.fixture(scope="module")
def vexact_engine(request):
    """Shared VeXact engine fixture. Use indirect parameterization to set pipeline_parallel_size."""
    model_path = os.environ["VEXACT_TESTS_MODEL_PATH"]
    pp_size = getattr(request, "param", 1)

    config = VeXactConfig(
        model=ModelConfig(
            model_path=model_path,
            attn_impl="fa-invariant",
            enable_batch_invariant=True,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=pp_size,
        ),
        scheduler=SchedulerConfig(
            max_num_batched_tokens=8,
            enable_chunked_prefill=True,
        ),
    )

    engine = VeXact(config)

    yield engine

    engine.close()


@pytest.fixture(scope="session")
def repo_root() -> str:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session", autouse=True)
def _init_memory_saver_adapter():
    from vexact.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

    TorchMemorySaverAdapter.create(enable=False)
