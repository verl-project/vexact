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

"""Check current VeOmni initialization in standalone spawned rollout workers."""

import os
import sys
from datetime import timedelta
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from vexact.utils.veomni_parallel_state import init_veomni_parallel_state


def _install_veomni():
    class AcceleratorConfig:
        def __init__(self):
            self.world_size = self.dp_size = self.dp_shard_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.dp_replicate_size = self.pp_size = self.tp_size = self.cp_size = self.ulysses_size = 1
            self.ep_size = 1

    veomni = ModuleType("veomni")
    arguments = ModuleType("veomni.arguments")
    arguments.AcceleratorConfig = AcceleratorConfig
    distributed = ModuleType("veomni.distributed")
    parallel_state = ModuleType("veomni.distributed.parallel_state")
    distributed.parallel_state = parallel_state
    veomni.arguments = arguments
    veomni.distributed = distributed
    return {
        "veomni": veomni,
        "veomni.arguments": arguments,
        "veomni.distributed": distributed,
        "veomni.distributed.parallel_state": parallel_state,
    }


@pytest.mark.parametrize("inherited_world_size", ["1", "16"])
def test_uses_rollout_group_topology(monkeypatch, inherited_world_size):
    modules = _install_veomni()
    initializer = Mock()
    modules["veomni.distributed.parallel_state"].init_parallel_state_from_config = initializer
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setenv("WORLD_SIZE", inherited_world_size)

    assert init_veomni_parallel_state() is initializer.return_value
    initializer.assert_called_once()
    accelerator = initializer.call_args.args[0]
    assert accelerator.world_size == accelerator.dp_size == accelerator.dp_shard_size == 4
    assert accelerator.dp_replicate_size == accelerator.pp_size == accelerator.tp_size == 1
    assert accelerator.cp_size == accelerator.ulysses_size == accelerator.ep_size == 1
    assert initializer.call_args.kwargs == {"name": None}
    assert os.environ["WORLD_SIZE"] == inherited_world_size


def test_uninitialized_group_does_not_import_veomni(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setitem(sys.modules, "veomni", None)
    assert init_veomni_parallel_state() is None


def _spawned_worker(rank, world_size, init_method):
    assert "verl" not in sys.modules
    assert os.environ["WORLD_SIZE"] == "16"
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        from unittest.mock import patch

        from veomni.distributed import parallel_state

        # Exercise a real VeOmni mesh on CPU even on the GPU CI runner.
        with patch.object(parallel_state, "get_device_type", return_value="cpu"):
            first = init_veomni_parallel_state()
            assert first is init_veomni_parallel_state()
        assert first is parallel_state.get_parallel_state()
        assert first.dp_size == first.dp_shard_size == first.world_size == world_size
        assert first.dp_replicate_size == first.pp_size == first.tp_size == 1
        assert first.cp_size == first.ulysses_size == 1
        assert not first.ep_enabled
        assert first.device_mesh.device_type == "cpu"
        assert first.dp_group.size() == world_size
        assert "verl" not in sys.modules
        assert os.environ["WORLD_SIZE"] == "16"
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_spawned_rollout_group_uses_current_api(monkeypatch, tmp_path):
    pytest.importorskip("veomni")
    monkeypatch.setenv("WORLD_SIZE", "16")
    mp.spawn(_spawned_worker, args=(4, f"file://{tmp_path / 'rendezvous'}"), nprocs=4, join=True)
