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

"""Test WorkerProcManager subprocess launching and CUDA_VISIBLE_DEVICES setting."""

import os
from dataclasses import dataclass

import torch


class MockProxy:
    """Mock proxy that verifies CUDA_VISIBLE_DEVICES is set correctly."""

    def __init__(self, config, rank: int):
        self.rank = rank

        # Verify CUDA_VISIBLE_DEVICES is set to rank
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        assert cuda_env == str(rank), f"CUDA_VISIBLE_DEVICES={cuda_env}, expected {rank}"

        # Verify torch sees at most 1 device (the one specified by CUDA_VISIBLE_DEVICES)
        device_count = torch.cuda.device_count()
        assert device_count <= 1, f"torch.cuda.device_count()={device_count}, expected <= 1"

        print(f"MockProxy rank={rank}: CUDA_VISIBLE_DEVICES={cuda_env}, device_count={device_count}")

    def start(self):
        pass

    def stop(self):
        pass


@dataclass
class MockParallelConfig:
    world_size: int


@dataclass
class MockConfig:
    parallel: MockParallelConfig


def test_cuda_visible_devices():
    """Test that CUDA_VISIBLE_DEVICES is set correctly for each worker subprocess."""
    from vexact.worker.worker_proc_manager import WorkerProcManager

    world_size = 4
    config = MockConfig(parallel=MockParallelConfig(world_size=world_size))
    manager = WorkerProcManager(config=config, proxy_cls=MockProxy)

    # If we get here, all workers started successfully (assertions passed in MockProxy.__init__)
    print(f"✓ All {world_size} workers verified CUDA_VISIBLE_DEVICES correctly")

    manager.close()
    print("✓ All workers shutdown")


if __name__ == "__main__":
    test_cuda_visible_devices()
