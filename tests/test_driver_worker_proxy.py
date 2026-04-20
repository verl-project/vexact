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

"""Unit tests for DriverWorkerProxy."""

import multiprocessing
import time

import pytest
from transformers import GenerationConfig

from vexact.core.request import DriverRequest, DriverRequestOutput, InferenceRequest
from vexact.ipc.control_channel import ControlChannelClient
from vexact.ipc.request_channel import RequestChannelClient


class MockDriverWorker:
    """Mock DriverWorker for testing proxy without loading real models."""

    def __init__(self):
        self._pending_requests: list[InferenceRequest] = []
        self._finished_requests: list[InferenceRequest] = []
        self._running = False
        self.sleep_count = 0
        self.wake_up_count = 0

    def start(self):
        """Start mock worker."""
        self._running = True

    def stop(self):
        """Stop mock worker."""
        self._running = False

    def submit_request(self, request: InferenceRequest) -> bool:
        """Submit request - immediately mark as finished with mock tokens."""
        # Simulate generation by adding mock tokens
        request.generated_tokens = [100, 101, 102]
        request.generated_logprobs = [-0.1, -0.2, -0.3]
        self._finished_requests.append(request)
        return True

    def poll_results(self) -> list[InferenceRequest]:
        """Return and clear finished requests."""
        finished = self._finished_requests[:]
        self._finished_requests.clear()
        return finished

    def sleep(self, tag: str = None):
        """Mock sleep."""
        self.sleep_count += 1
        return f"slept with tag={tag}"

    def wake_up(self, tag: str = None):
        """Mock wake_up."""
        self.wake_up_count += 1
        return f"woke up with tag={tag}"


def run_mock_proxy_server(request_address: str, control_address: str, ready_queue: multiprocessing.Queue):
    """Run a mock proxy server in a separate process."""
    from vexact.core.request import DriverRequest, InferenceRequest
    from vexact.ipc.control_channel import ControlChannelServer
    from vexact.ipc.request_channel import RequestChannelServer

    worker = MockDriverWorker()

    def on_request(request: DriverRequest):
        inference_request = InferenceRequest.from_driver_request(request)
        worker.submit_request(inference_request)

    def poll_outputs(timeout: float = None) -> list[DriverRequestOutput]:
        outputs = []
        for req in worker.poll_results():
            output = DriverRequestOutput(
                request_id=req.request_id,
                new_token_ids=req.generated_tokens,
                new_logprobs=req.generated_logprobs if req.generated_logprobs else None,
            )
            outputs.append(output)
        return outputs

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

    control_server.start()
    request_server.start()
    worker.start()

    ready_queue.put("ready")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        request_server.stop()
        control_server.stop()


def start_mock_proxy(request_address: str, control_address: str):
    """Start mock proxy and wait for it to be ready."""
    # Clean up stale IPC sockets from prior runs
    import pathlib

    for addr in (request_address, control_address):
        sock_path = addr.replace("ipc://", "")
        for suffix in ("", ".in", ".out"):
            p = pathlib.Path(sock_path + suffix)
            p.unlink(missing_ok=True)

    ready_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=run_mock_proxy_server,
        args=(request_address, control_address, ready_queue),
    )
    process.start()

    status = ready_queue.get(timeout=30)
    assert status == "ready"
    time.sleep(0.1)

    return process


def cleanup_process(process: multiprocessing.Process):
    """Terminate process."""
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()


class TestDriverWorkerProxy:
    """Test DriverWorkerProxy functionality."""

    @pytest.mark.asyncio
    async def test_request_response(self):
        """Test submitting request and receiving response."""
        request_address = "ipc:///tmp/vexact_test_proxy_req.sock"
        control_address = "ipc:///tmp/vexact_test_proxy_ctrl.sock"

        process = start_mock_proxy(request_address, control_address)

        try:
            client = RequestChannelClient(request_address)

            request = DriverRequest(
                request_id="test_req_1",
                generation_config=GenerationConfig(max_new_tokens=10),
                input_ids_list=[1, 2, 3],
            )

            outputs = []
            async for output in client.submit(request):
                outputs.append(output)
                break  # Get first output

            assert len(outputs) == 1
            assert outputs[0].request_id == "test_req_1"
            assert outputs[0].new_token_ids == [100, 101, 102]
            assert outputs[0].new_logprobs == [-0.1, -0.2, -0.3]

            client.close()

        finally:
            cleanup_process(process)

    def test_control_channel_sleep_wake_up(self):
        """Test control channel RPC calls."""
        request_address = "ipc:///tmp/vexact_test_proxy_req2.sock"
        control_address = "ipc:///tmp/vexact_test_proxy_ctrl2.sock"

        process = start_mock_proxy(request_address, control_address)

        try:
            client = ControlChannelClient([control_address])

            # Test sleep
            results = client.collective_rpc("sleep", tag="test_tag")
            assert len(results) == 1
            assert results[0] == "slept with tag=test_tag"

            # Test wake_up
            results = client.collective_rpc("wake_up", tag="test_tag")
            assert len(results) == 1
            assert results[0] == "woke up with tag=test_tag"

            client.close()

        finally:
            cleanup_process(process)


@pytest.fixture(scope="session", autouse=True)
def setup_multiprocessing():
    """Set up multiprocessing start method for the test session."""
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set
