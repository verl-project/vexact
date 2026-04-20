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

"""Unit tests for WorkerProxy."""

import multiprocessing
import time

import pytest

from vexact.ipc.control_channel import ControlChannelClient


class MockWorker:
    """Mock Worker for testing proxy without loading real models."""

    def __init__(self, rank: int = 1):
        self.rank = rank
        self.sleep_count = 0
        self.wake_up_count = 0

        # Mock pp_info
        class MockPPInfo:
            def __init__(self, rank):
                self.rank = rank

        self.pp_info = MockPPInfo(rank)

    def start(self):
        """Start mock worker (blocking)."""
        while True:
            time.sleep(0.1)

    def sleep(self, tag: str = None):
        """Mock sleep."""
        self.sleep_count += 1
        return f"worker {self.rank} slept with tag={tag}"

    def wake_up(self, tag: str = None):
        """Mock wake_up."""
        self.wake_up_count += 1
        return f"worker {self.rank} woke up with tag={tag}"


def run_mock_worker_proxy(rank: int, control_address: str, ready_queue: multiprocessing.Queue):
    """Run a mock worker proxy in a separate process."""
    from vexact.ipc.control_channel import ControlChannelServer

    worker = MockWorker(rank)

    control_server = ControlChannelServer(
        address=control_address,
        target=worker,
        allowed_methods=["sleep", "wake_up"],
    )

    control_server.start()
    ready_queue.put("ready")

    # Run worker (blocking)
    try:
        worker.start()
    except KeyboardInterrupt:
        pass
    finally:
        control_server.stop()


def start_mock_worker_proxy(rank: int, control_address: str):
    """Start mock worker proxy and wait for it to be ready."""
    ready_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=run_mock_worker_proxy,
        args=(rank, control_address, ready_queue),
    )
    process.start()

    status = ready_queue.get(timeout=5)
    assert status == "ready"
    time.sleep(0.1)

    return process


def cleanup_process(process: multiprocessing.Process):
    """Terminate process."""
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()


class TestWorkerProxy:
    """Test WorkerProxy functionality."""

    def test_single_worker_control(self):
        """Test control channel RPC to single worker."""
        control_address = "ipc:///tmp/vexact_test_worker_ctrl.sock"

        process = start_mock_worker_proxy(rank=1, control_address=control_address)

        try:
            client = ControlChannelClient([control_address])

            # Test sleep
            results = client.collective_rpc("sleep", tag="test_tag")
            assert len(results) == 1
            assert results[0] == "worker 1 slept with tag=test_tag"

            # Test wake_up
            results = client.collective_rpc("wake_up")
            assert len(results) == 1
            assert results[0] == "worker 1 woke up with tag=None"

            client.close()

        finally:
            cleanup_process(process)

    def test_multiple_workers_collective_rpc(self):
        """Test collective RPC to multiple workers."""
        control_addresses = [
            "ipc:///tmp/vexact_test_worker_ctrl_1.sock",
            "ipc:///tmp/vexact_test_worker_ctrl_2.sock",
        ]

        processes = []
        for i, addr in enumerate(control_addresses):
            process = start_mock_worker_proxy(rank=i + 1, control_address=addr)
            processes.append(process)

        try:
            client = ControlChannelClient(control_addresses)

            # Test collective sleep
            results = client.collective_rpc("sleep", tag="collective_test")
            assert len(results) == 2
            assert results[0] == "worker 1 slept with tag=collective_test"
            assert results[1] == "worker 2 slept with tag=collective_test"

            # Test collective wake_up
            results = client.collective_rpc("wake_up", tag="collective_test")
            assert len(results) == 2
            assert results[0] == "worker 1 woke up with tag=collective_test"
            assert results[1] == "worker 2 woke up with tag=collective_test"

            client.close()

        finally:
            for process in processes:
                cleanup_process(process)


@pytest.fixture(scope="session", autouse=True)
def setup_multiprocessing():
    """Set up multiprocessing start method for the test session."""
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set
