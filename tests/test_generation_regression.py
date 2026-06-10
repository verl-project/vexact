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

import json
import os
import subprocess
from pathlib import Path

import pytest

from vexact.core.request import RequestStatus
from vexact.utils.subprocess_utils import get_sys_executable


_COMPLETED_METADATA_STATUS = {"completed", RequestStatus.FINISHED.value}


def _load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_generated_texts(metadata_path: Path) -> list[str]:
    metadata = _load_metadata(metadata_path)

    generated_texts = []
    for request_id, entry in metadata.items():
        if "generated_text" not in entry:
            raise AssertionError(f"Missing generated_text for request {request_id}")
        generated_texts.append(entry["generated_text"])

    return generated_texts


def _run_simulation_and_compare(tmp_path: Path, pipeline_parallel_size: int):
    repo_root = Path(__file__).resolve().parents[1]
    baseline_path = repo_root / "tests/ref_data" / "metadata.json"
    if not baseline_path.exists():
        pytest.skip("Baseline metadata.json not found; run the manual command once to record it.")

    output_dir = tmp_path / f"inference_outputs_pp{pipeline_parallel_size}"
    output_dir.mkdir()

    model_path = os.environ["VEXACT_TESTS_MODEL_PATH"]
    attn_impl = os.environ.get("VEXACT_TESTS_ATTN_IMPL", "fa-invariant")
    cmd = get_sys_executable() + [
        str(repo_root / "tests/scripts/hf_inference.py"),
        "--model_path",
        model_path,
        "--attn_impl",
        attn_impl,
        "--pipeline_parallel_size",
        str(pipeline_parallel_size),
        "--simulate_requests",
        "4",
        "--request_interval",
        "0.01",
        "--max_length",
        "256",
        "--max_new_tokens",
        "256",
        "--enable_batch_invariant",
        "--enable_memory_saver",
        "--enable_chunked_prefill",
        "--output_dir",
        str(output_dir),
    ]

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    result = subprocess.run(
        cmd,
        env=env,
        check=False,
        text=True,
    )

    if result.returncode != 0:
        if result.returncode in [-11, -6]:
            print("(--enable_memory_saver is provided, return code can be SIGSEGV or ABORT.)")
        else:
            raise AssertionError(
                f"hf_inference simulation command failed: return code {result.returncode}, check log above."
            )

    new_metadata_path = output_dir / "metadata.json"
    assert new_metadata_path.exists(), "metadata.json was not produced by the simulation run"

    print(f"{baseline_path=}, {new_metadata_path=}")
    baseline_outputs = _load_generated_texts(baseline_path)
    new_outputs = _load_generated_texts(new_metadata_path)

    assert baseline_outputs, "Baseline metadata.json contains no generated_text entries"
    assert len(new_outputs) == len(baseline_outputs), (
        f"Expected {len(baseline_outputs)} generated texts, got {len(new_outputs)}"
    )

    if attn_impl != "fa-invariant":
        new_metadata = _load_metadata(new_metadata_path)
        for request_id, entry in new_metadata.items():
            assert entry.get("status") in _COMPLETED_METADATA_STATUS, f"{request_id} did not complete: {entry}"
            assert entry.get("response_len", 0) > 0, f"{request_id} produced no response tokens"
        return

    baseline_texts = sorted(baseline_outputs)
    new_texts = sorted(new_outputs)
    assert baseline_texts == new_texts, (
        f"Generated texts differ from baseline metadata.json\n{baseline_texts=}\n{new_texts=}"
    )


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_simulated_generation_matches_baseline(tmp_path: Path, pipeline_parallel_size: int):
    _run_simulation_and_compare(tmp_path, pipeline_parallel_size)
