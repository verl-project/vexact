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

"""Control channel implementation for sync RPC communication.

Supports IPC, TCP (IPv4), and TCP (IPv6) addresses.
For TCP addresses, dual-stack (IPv4+IPv6) is automatically enabled.
"""

import logging
import threading
from typing import Any

import msgspec
import zmq


logger = logging.getLogger(__name__)


class ControlRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    """Request message for control channel RPC.

    Attributes:
        method: The method name to call on the target object
        args: Positional arguments for the method call
        kwargs: Keyword arguments for the method call
    """

    method: str
    args: tuple = ()
    kwargs: dict = {}


class ControlResponse(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    """Response message for control channel RPC.

    Attributes:
        success: Whether the method call succeeded
        result: The return value from the method call (if successful)
        error: Error message (if failed)
    """

    success: bool
    result: Any = None
    error: str | None = None


class ControlChannelServer:
    """Control channel server using ZMQ REP socket.

    Handles synchronous RPC requests in a background thread by dispatching
    method calls to a target object with whitelist validation.
    Supports IPC, TCP (IPv4), and TCP (IPv6) addresses automatically.

    Args:
        address: ZMQ address to bind (e.g., "ipc:///tmp/vexact_ctrl_0.sock" or "tcp://*:5555")
        target: The object whose methods will be called
        allowed_methods: List of method names that are allowed to be called
    """

    def __init__(self, address: str, target: object, allowed_methods: list[str]):
        self.address = address
        self.target = target
        self.allowed_methods = set(allowed_methods)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        # Enable dual-stack (IPv4 + IPv6) for TCP addresses
        if address.startswith("tcp://"):
            self.socket.setsockopt(zmq.IPV6, 1)

        self.socket.bind(address)

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(ControlRequest)

        self.shutdown_event = threading.Event()
        self.server_thread = None

        logger.info(f"ControlChannelServer bound to {address}")

    def start(self):
        """Start the server in a background thread."""
        if self.server_thread and self.server_thread.is_alive():
            logger.warning("ControlChannelServer is already running")
            return

        self.shutdown_event.clear()
        self.server_thread = threading.Thread(target=self._run, daemon=True)
        self.server_thread.start()
        logger.info("ControlChannelServer started")

    def stop(self):
        """Stop the server and clean up resources."""
        if not self.server_thread or not self.server_thread.is_alive():
            return

        logger.info("Stopping ControlChannelServer")
        self.shutdown_event.set()

        # Wake up the blocking recv with a dummy request from a temp socket
        try:
            temp_context = zmq.Context()
            temp_socket = temp_context.socket(zmq.REQ)
            temp_socket.connect(self.address)
            temp_socket.send(b"")
            temp_socket.close()
            temp_context.term()
        except Exception as e:
            logger.debug(f"Error sending shutdown signal: {e}")

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)

        self.socket.close()
        self.context.term()
        logger.info("ControlChannelServer stopped")

    def _run(self):
        """Main server loop running in background thread."""
        logger.info("ControlChannelServer loop started")

        while not self.shutdown_event.is_set():
            try:
                # Receive request (blocking)
                message = self.socket.recv()

                if self.shutdown_event.is_set():
                    break

                # Decode and handle request
                try:
                    request = self.decoder.decode(message)
                    response = self._handle_request(request)
                except Exception as e:
                    logger.error(f"Error decoding/handling request: {e}")
                    response = ControlResponse(success=False, error=f"Server error: {str(e)}")

                # Send response
                self.socket.send(self.encoder.encode(response))

            except zmq.ZMQError as e:
                if not self.shutdown_event.is_set():
                    logger.error(f"ZMQ error in server loop: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in server loop: {e}")

        logger.info("ControlChannelServer loop ended")

    def _handle_request(self, request: ControlRequest) -> ControlResponse:
        """Handle a control request by calling the target method.

        Args:
            request: The control request to handle

        Returns:
            ControlResponse with success status, result, or error
        """
        # Validate method is allowed
        if request.method not in self.allowed_methods:
            return ControlResponse(success=False, error=f"Method not allowed: {request.method}")

        # Check if method exists on target
        if not hasattr(self.target, request.method):
            return ControlResponse(success=False, error=f"Method not found: {request.method}")

        # Call the method
        try:
            method = getattr(self.target, request.method)
            result = method(*request.args, **request.kwargs)
            return ControlResponse(success=True, result=result)
        except Exception as e:
            logger.error(f"Error calling {request.method}: {e}")
            return ControlResponse(success=False, error=str(e))


class ControlChannelClient:
    """Control channel client for one-to-many RPC.

    Manages multiple ZMQ REQ sockets to communicate with multiple proxies.
    Supports IPC, TCP (IPv4), and TCP (IPv6) addresses automatically.

    Args:
        proxy_addresses: List of ZMQ addresses to connect to
                        (e.g., ["ipc:///tmp/vexact_ctrl_0.sock", "tcp://localhost:5555"])
    """

    def __init__(self, proxy_addresses: list[str]):
        self.proxy_addresses = proxy_addresses
        self.context = zmq.Context()
        self.sockets = []

        # Create a REQ socket for each proxy
        for address in proxy_addresses:
            socket = self.context.socket(zmq.REQ)

            # Enable dual-stack (IPv4 + IPv6) for TCP addresses
            if address.startswith("tcp://"):
                socket.setsockopt(zmq.IPV6, 1)

            socket.connect(address)
            self.sockets.append(socket)
            logger.info(f"ControlChannelClient connected to {address}")

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(ControlResponse)

    def collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        """Execute RPC on all proxies and collect results.

        Sends requests to all proxies in parallel, then collects responses.
        This reduces latency compared to sequential send-recv cycles.

        Args:
            method: The method name to call
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of results from all proxies (in order)

        Raises:
            RuntimeError: If any proxy call fails
        """
        request = ControlRequest(method=method, args=args, kwargs=kwargs)
        message = self.encoder.encode(request)

        errors = []
        sent_successfully = [False] * len(self.sockets)

        # Phase 1: Send to all sockets (parallel dispatch)
        for i, socket in enumerate(self.sockets):
            try:
                socket.send(message)
                sent_successfully[i] = True
            except Exception as e:
                error_msg = f"Proxy {i} ({self.proxy_addresses[i]}): send failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        # Phase 2: Receive from all sockets that successfully sent
        results = []

        for i, socket in enumerate(self.sockets):
            if not sent_successfully[i]:
                continue

            try:
                response_message = socket.recv()
                response = self.decoder.decode(response_message)

                if response.success:
                    results.append(response.result)
                else:
                    error_msg = f"Proxy {i} ({self.proxy_addresses[i]}): {response.error}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            except Exception as e:
                error_msg = f"Proxy {i} ({self.proxy_addresses[i]}): recv failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        if errors:
            raise RuntimeError(f"Errors calling {method}: {'; '.join(errors)}")

        return results

    def close(self):
        """Close all sockets and terminate the context."""
        for socket in self.sockets:
            socket.close()
        self.context.term()
        logger.info("ControlChannelClient closed")
