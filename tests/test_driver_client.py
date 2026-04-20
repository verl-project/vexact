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

"""Unit tests for DriverClient."""

import multiprocessing

import pytest
from transformers import GenerationConfig

from vexact.config import DriverConfig
from vexact.core.request import DriverRequest, DriverRequestOutput, RequestStatus
from vexact.worker.driver_client import DriverClient


def run_mock_driver_worker(request_address: str, control_address: str, result_queue: multiprocessing.Queue):
    """Run mock DriverWorkerProxy in a separate process."""
    from vexact.ipc.control_channel import ControlChannelServer
    from vexact.ipc.request_channel import RequestChannelServer

    # Mock worker state
    sleep_count = 0
    wake_up_count = 0

    # Control channel target methods
    class MockWorker:
        def sleep(self, tag: str = None):
            nonlocal sleep_count
            sleep_count += 1
            return f"slept_{sleep_count}"

        def wake_up(self, tag: str = None):
            nonlocal wake_up_count
            wake_up_count += 1
            return f"woke_up_{wake_up_count}"

    worker = MockWorker()

    # Request channel handlers
    received_requests = []
    outputs_to_send = []
    num_outputs = 3

    def on_request(request: DriverRequest):
        print(f"[Server] Received request: {request.request_id}")
        received_requests.append(request)
        # Auto-respond with outputs, last one marked as finished
        for i in range(num_outputs):
            is_last = i == num_outputs - 1
            output = DriverRequestOutput(
                request_id=request.request_id,
                new_token_ids=[100 + i],
                new_logprobs=None,
                status=RequestStatus.FINISHED if is_last else RequestStatus.RUNNING,
            )
            outputs_to_send.append(output)

    def poll_outputs(timeout=None):
        if outputs_to_send:
            return [outputs_to_send.pop(0)]
        return []

    # Create servers
    request_server = RequestChannelServer(
        address=request_address,
        on_request=on_request,
        poll_outputs=poll_outputs,
    )
    control_server = ControlChannelServer(
        address=control_address,
        target=worker,
        allowed_methods=["sleep", "wake_up"],
    )

    request_server.start()
    control_server.start()

    # Signal ready
    result_queue.put("ready")
    print(f"[Server] Ready on {request_address}, {control_address}")

    # Keep servers running
    try:
        import time

        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        request_server.stop()
        control_server.stop()


def start_mock_server(
    request_address: str, control_address: str
) -> tuple[multiprocessing.Process, multiprocessing.Queue]:
    """Start mock server and wait for it to be ready."""
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=run_mock_driver_worker, args=(request_address, control_address, result_queue)
    )
    process.start()

    # Wait for server to be ready
    status = result_queue.get(timeout=50)
    assert status == "ready"

    import time

    time.sleep(0.1)

    return process, result_queue


def cleanup_server(process: multiprocessing.Process):
    """Terminate server process."""
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()


@pytest.mark.asyncio
class TestDriverClient:
    """Test DriverClient functionality."""

    async def test_generate_streaming(self):
        """Test streaming generation via request channel."""
        request_address = "ipc:///tmp/vexact_test_driver_request.sock"
        control_address = "ipc:///tmp/vexact_test_driver_ctrl.sock"
        process, _ = start_mock_server(request_address, control_address)

        try:
            config = DriverConfig(
                request_address=request_address,
                control_addresses=[control_address],
            )
            client = DriverClient(config)

            # Submit request (auto-starts)
            request = DriverRequest(
                request_id="stream_test",
                generation_config=GenerationConfig(max_new_tokens=5),
                input_ids_list=[1, 2, 3],
            )

            # Collect streaming outputs - loop auto-terminates on is_finished
            outputs = []
            async for output in client.generate(request):
                outputs.append(output)

            # Verify all outputs received and is_finished behavior
            assert len(outputs) == 3
            assert outputs[0].request_id == "stream_test"
            assert outputs[0].new_token_ids == [100]
            assert outputs[0].is_finished is False
            assert outputs[1].new_token_ids == [101]
            assert outputs[1].is_finished is False
            assert outputs[2].new_token_ids == [102]
            assert outputs[2].is_finished is True

            client.close()

        finally:
            cleanup_server(process)

    async def test_sleep_wake_up(self):
        """Test sleep and wake_up control methods."""
        request_address = "ipc:///tmp/vexact_test_sleep_req.sock"
        control_address = "ipc:///tmp/vexact_test_sleep_ctrl.sock"
        process, _ = start_mock_server(request_address, control_address)

        try:
            config = DriverConfig(
                request_address=request_address,
                control_addresses=[control_address],
            )
            client = DriverClient(config)

            # Test sleep
            results = client.sleep(tag="test_tag")
            assert len(results) == 1
            assert results[0] == "slept_1"

            # Test wake_up
            results = client.wake_up(tag="test_tag")
            assert len(results) == 1
            assert results[0] == "woke_up_1"

            client.close()

        finally:
            cleanup_server(process)


@pytest.fixture(scope="session", autouse=True)
def setup_multiprocessing():
    """Set up multiprocessing start method for the test session."""
    multiprocessing.set_start_method("spawn", force=True)
