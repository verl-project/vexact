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

"""Request channel implementation for async bidirectional streaming.

Uses separate PUSH/PULL sockets for input and output (1:1 server-client).
Each server thread owns its own ZMQ context and socket (RAII pattern).
Only supports IPC addresses.
"""

import asyncio
import logging
import threading
from typing import AsyncIterator, Callable

import msgspec
import zmq
import zmq.asyncio

from vexact.core.request import DriverRequest, DriverRequestOutput


logger = logging.getLogger(__name__)


class RequestChannelClient:
    """Client for submitting requests and receiving outputs.

    Uses PUSH socket for sending requests and PULL socket for receiving outputs.

    Args:
        address: Base ZMQ IPC address (e.g., "ipc:///tmp/vexact.sock")
    """

    def __init__(self, address: str):
        if not address.startswith("ipc://"):
            raise ValueError(f"RequestChannelClient only supports IPC addresses, got: {address}")

        self._context = zmq.asyncio.Context()
        self._input_socket = self._context.socket(zmq.PUSH)
        self._input_socket.connect(f"{address}.in")
        self._output_socket = self._context.socket(zmq.PULL)
        self._output_socket.connect(f"{address}.out")

        self._encoder = msgspec.msgpack.Encoder(enc_hook=DriverRequest.enc_hook)
        self._decoder = msgspec.msgpack.Decoder(DriverRequestOutput)

        self._output_queues: dict[str, asyncio.Queue] = {}
        self._receive_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    async def submit(self, request: DriverRequest) -> AsyncIterator[DriverRequestOutput]:
        """Submit a request and stream outputs asynchronously."""
        if not self._receive_task:
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._monitor_task = asyncio.create_task(self._monitor_loop())

        queue: asyncio.Queue = asyncio.Queue()
        self._output_queues[request.request_id] = queue
        await self._input_socket.send(self._encoder.encode(request))

        try:
            while True:
                output: DriverRequestOutput = await queue.get()
                yield output
                if not output.is_running:
                    break
        finally:
            self._output_queues.pop(request.request_id, None)

    async def _receive_loop(self):
        """Background loop to receive outputs and route to queues."""
        while True:
            try:
                if await self._output_socket.poll(timeout=100):
                    output = self._decoder.decode(await self._output_socket.recv())
                    if queue := self._output_queues.get(output.request_id):
                        await queue.put(output)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")

    async def _monitor_loop(self):
        """Background loop to log unfinished requests every 10 seconds."""
        while True:
            try:
                await asyncio.sleep(120)
                if self._output_queues:
                    logger.info(f"[Driver Client Monitor] Running requests ({len(self._output_queues)})")
            except asyncio.CancelledError:
                break

    def close(self):
        """Close client and clean up resources."""
        if self._receive_task:
            self._receive_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()
        self._input_socket.close()
        self._output_socket.close()
        self._context.term()


class RequestChannelServer:
    """Server for handling requests and sending outputs.

    Uses PULL socket for receiving requests and PUSH socket for sending outputs.
    Each thread owns its own ZMQ context and socket (RAII pattern).

    Args:
        address: Base ZMQ IPC address (e.g., "ipc:///tmp/vexact.sock")
        on_request: Callback to handle incoming requests
        poll_outputs: Callback to poll for outputs (timeout: float) -> list[DriverRequestOutput]
    """

    def __init__(
        self,
        address: str,
        on_request: Callable[[DriverRequest], None],
        poll_outputs: Callable[[float], list[DriverRequestOutput]],
    ):
        if not address.startswith("ipc://"):
            raise ValueError(f"RequestChannelServer only supports IPC addresses, got: {address}")

        self._input_address = f"{address}.in"
        self._output_address = f"{address}.out"
        self._on_request = on_request
        self._poll_outputs = poll_outputs

        self._shutdown = threading.Event()
        self._input_thread: threading.Thread | None = None
        self._output_thread: threading.Thread | None = None

    def start(self):
        """Start server with separate input and output threads."""
        if self._input_thread and self._input_thread.is_alive():
            logger.warning("RequestChannelServer is already running")
            return

        self._shutdown.clear()
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._output_thread = threading.Thread(target=self._output_loop, daemon=True)
        self._input_thread.start()
        self._output_thread.start()
        logger.info(f"RequestChannelServer started on {self._input_address}")

    def stop(self):
        """Signal shutdown and wait for threads to finish."""
        if not self._input_thread or not self._input_thread.is_alive():
            return

        self._shutdown.set()
        self._input_thread.join(timeout=2.0)
        self._output_thread.join(timeout=2.0)
        logger.info("RequestChannelServer stopped")

    def _input_loop(self):
        """Input thread: owns context/socket, receives requests."""
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind(self._input_address)
        decoder = msgspec.msgpack.Decoder(DriverRequest, dec_hook=DriverRequest.dec_hook)

        try:
            while not self._shutdown.is_set():
                if socket.poll(timeout=100):
                    try:
                        request = decoder.decode(socket.recv())
                        self._on_request(request)
                    except Exception as e:
                        logger.error(f"Error handling request: {e}")
        finally:
            socket.close()
            context.term()

    def _output_loop(self):
        """Output thread: owns context/socket, polls and sends outputs."""
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.bind(self._output_address)
        encoder = msgspec.msgpack.Encoder()

        try:
            while not self._shutdown.is_set():
                try:
                    for output in self._poll_outputs(timeout=1.0):
                        socket.send(encoder.encode(output))
                except Exception as e:
                    logger.error(f"Error in output loop: {e}")
        finally:
            socket.close()
            context.term()
