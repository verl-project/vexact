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

"""Unit tests for request channel."""

import multiprocessing

import msgspec
import pytest
from transformers import GenerationConfig

from vexact.core.request import DriverRequest, DriverRequestOutput, RequestStatus
from vexact.ipc.request_channel import RequestChannelClient


def run_simple_server(address: str, result_queue: multiprocessing.Queue, num_outputs: int = 1):
    """Run a simple request channel server in a separate process."""
    from vexact.ipc.request_channel import RequestChannelServer

    outputs_to_send = []

    def on_request(request: DriverRequest):
        """Handle incoming request."""
        print(f"[Server] Received request: {request.request_id}")
        # Auto-respond with multiple outputs, last one marked as finished
        for i in range(num_outputs):
            is_last = i == num_outputs - 1
            output = DriverRequestOutput(
                request_id=request.request_id,
                new_token_ids=[100 + i],
                new_logprobs=None,
                status=RequestStatus.FINISHED if is_last else RequestStatus.RUNNING,
            )
            outputs_to_send.append(output)

    def poll_outputs(timeout: float = None):
        """Poll for available outputs."""
        if outputs_to_send:
            return [outputs_to_send.pop(0)]
        return []

    server = RequestChannelServer(
        address=address,
        on_request=on_request,
        poll_outputs=poll_outputs,
    )
    server.start()

    # Signal ready
    result_queue.put("ready")
    print(f"[Server] Ready on {address}")

    # Keep server running
    try:
        import time

        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def start_server(address: str, num_outputs: int = 1) -> tuple[multiprocessing.Process, multiprocessing.Queue]:
    """Start server process and wait for it to be ready."""
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_simple_server, args=(address, result_queue, num_outputs))
    process.start()

    # Wait for server to be ready
    status = result_queue.get(timeout=20)
    assert status == "ready"

    import time

    time.sleep(0.1)  # Extra time for socket binding

    return process, result_queue


def cleanup_server(process: multiprocessing.Process):
    """Terminate server process."""
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()


def test_driver_request_preserves_generation_config_seed():
    generation_config = GenerationConfig(max_new_tokens=10)
    generation_config.seed = 12345
    request = DriverRequest(
        request_id="seeded_req",
        generation_config=generation_config,
        input_ids_list=[1, 2, 3],
    )

    encoder = msgspec.msgpack.Encoder(enc_hook=DriverRequest.enc_hook)
    decoder = msgspec.msgpack.Decoder(DriverRequest, dec_hook=DriverRequest.dec_hook)
    decoded = decoder.decode(encoder.encode(request))

    assert decoded.request_id == request.request_id
    assert decoded.input_ids_list == request.input_ids_list
    assert decoded.generation_config.max_new_tokens == 10
    assert decoded.generation_config.seed == 12345


class TestRequestChannel:
    """Test request channel server and client."""

    @pytest.mark.asyncio
    async def test_single_request_output(self):
        """Test basic request-output flow across processes."""
        address = "ipc:///tmp/vexact_test_req_single.sock"
        process, _ = start_server(address, num_outputs=1)

        try:
            # Create client
            client = RequestChannelClient(address)

            # Submit request (auto-starts)
            request = DriverRequest(
                request_id="req_1",
                generation_config=GenerationConfig(max_new_tokens=10),
                input_ids_list=[1, 2, 3],
            )

            # Collect outputs - loop terminates when is_finished=True
            outputs = []
            async for output in client.submit(request):
                outputs.append(output)

            assert len(outputs) == 1
            assert outputs[0].request_id == "req_1"
            assert outputs[0].new_token_ids == [100]
            assert outputs[0].is_finished is True

            client.close()

        finally:
            cleanup_server(process)

    @pytest.mark.asyncio
    async def test_multiple_outputs_with_is_finished(self):
        """Test that client properly terminates when is_finished=True."""
        address = "ipc:///tmp/vexact_test_req_multi.sock"
        process, _ = start_server(address, num_outputs=3)

        try:
            client = RequestChannelClient(address)

            request = DriverRequest(
                request_id="req_multi",
                generation_config=GenerationConfig(max_new_tokens=10),
                input_ids_list=[1, 2, 3],
            )

            # Collect all outputs - should auto-terminate on is_finished
            outputs = []
            async for output in client.submit(request):
                outputs.append(output)

            # Verify we got all 3 outputs
            assert len(outputs) == 3
            assert outputs[0].new_token_ids == [100]
            assert outputs[0].is_finished is False
            assert outputs[1].new_token_ids == [101]
            assert outputs[1].is_finished is False
            assert outputs[2].new_token_ids == [102]
            assert outputs[2].is_finished is True

            # Verify queue cleanup
            assert "req_multi" not in client._output_queues

            client.close()

        finally:
            cleanup_server(process)

    def test_ipc_only_validation(self):
        """Test that non-IPC addresses are rejected."""
        from vexact.ipc.request_channel import RequestChannelServer

        with pytest.raises(ValueError, match="only supports IPC addresses"):
            RequestChannelClient("tcp://localhost:5555")

        with pytest.raises(ValueError, match="only supports IPC addresses"):
            RequestChannelServer(
                address="tcp://*:5555",
                on_request=lambda _: None,
                poll_outputs=lambda timeout: [],
            )


@pytest.fixture(scope="session", autouse=True)
def setup_multiprocessing():
    """Set up multiprocessing start method for the test session."""
    multiprocessing.set_start_method("spawn", force=True)
