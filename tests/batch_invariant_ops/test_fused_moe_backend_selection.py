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

"""Backend selection tests for fused MoE."""

import pytest
import torch


if not torch.cuda.is_available():
    pytest.skip("CUDA is required to import fused MoE kernels", allow_module_level=True)

import vexact.batch_invariant_ops.fused_moe as fused_moe
import vexact.utils.device as device_utils


class _SentinelExpert:
    called = False

    @classmethod
    def apply(cls, *args):
        cls.called = True
        return args[3]


def _run_selection(monkeypatch, device_major: int):
    class TritonExpert(_SentinelExpert):
        called = False

    class QuackExpert(_SentinelExpert):
        called = False

    monkeypatch.setattr(device_utils, "DEVICE_MAJOR", device_major)
    monkeypatch.setattr(fused_moe, "FusedMoeExpertFunction", TritonExpert)
    monkeypatch.setattr(fused_moe, "QuackFusedMoeExpertFunction", QuackExpert)

    hidden_states = torch.ones(1, 4, dtype=torch.bfloat16)
    result = fused_moe.fused_moe_forward(
        1,
        torch.ones(1, 1, dtype=torch.bfloat16),
        torch.zeros(1, 1, dtype=torch.int32),
        hidden_states,
        torch.ones(1, 4, 4, dtype=torch.bfloat16),
        torch.ones(1, 4, 4, dtype=torch.bfloat16),
        torch.ones(1, 4, 4, dtype=torch.bfloat16),
    )

    assert result is hidden_states
    return TritonExpert.called, QuackExpert.called


def test_hopper_uses_triton_fused_moe(monkeypatch):
    triton_called, quack_called = _run_selection(monkeypatch, device_major=9)

    assert triton_called
    assert not quack_called


def test_blackwell_or_newer_uses_quack_fused_moe(monkeypatch):
    triton_called, quack_called = _run_selection(monkeypatch, device_major=10)

    assert not triton_called
    assert quack_called


def test_sm11_uses_quack_fused_moe(monkeypatch):
    triton_called, quack_called = _run_selection(monkeypatch, device_major=11)

    assert not triton_called
    assert quack_called
