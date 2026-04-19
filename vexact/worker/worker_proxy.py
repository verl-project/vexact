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

"""Worker proxies: wrap workers with IPC channels for remote access."""

import logging
from abc import ABC, abstractmethod

from vexact.config import VeXactConfig
from vexact.core.request import DriverRequest, DriverRequestOutput, InferenceRequest
from vexact.ipc.control_channel import ControlChannelServer
from vexact.ipc.request_channel import RequestChannelServer
from vexact.worker.driver_worker import DriverWorker
from vexact.worker.worker import Worker, WorkerBase


logger = logging.getLogger(__name__)


class ProxyBase(ABC):
    worker: WorkerBase

    @abstractmethod
    def start(self):
        """Start the proxy (non-blocking)."""

    @abstractmethod
    def stop(self):
        """Stop the proxy and clean up resources."""

    def join(self):
        """Block until the worker stops (for non-subprocess launch)."""
        self.worker.join()


class WorkerProxy(ProxyBase):
    """Proxy that wraps Worker with IPC control channel for remote access."""

    def __init__(self, config: VeXactConfig, rank: int):
        self.rank = rank
        self.worker = Worker(config, rank)
        self.control_server = ControlChannelServer(
            address=config.driver.control_addresses[rank],
            target=self.worker,
            allowed_methods=["sleep", "wake_up", "receive_weights"],
        )

    def start(self):
        self.control_server.start()
        self.worker.start()

    def stop(self):
        self.worker.stop()
        self.control_server.stop()
        logger.info(f"WorkerProxy (rank={self.rank}) stopped")


class DriverWorkerProxy(ProxyBase):
    """Proxy that wraps DriverWorker with IPC channels for remote access."""

    def __init__(self, config: VeXactConfig, rank: int):
        if rank != 0:
            raise ValueError(f"DriverWorkerProxy must have rank=0, got {rank}")
        self.rank = rank
        self.worker: DriverWorker = DriverWorker(config)
        self.request_server = RequestChannelServer(
            address=config.driver.request_address,
            on_request=self._on_request,
            poll_outputs=self._poll_outputs,
        )
        self.control_server = ControlChannelServer(
            address=config.driver.control_addresses[rank],
            target=self.worker,
            allowed_methods=["sleep", "wake_up", "receive_weights"],
        )

    def start(self):
        self.control_server.start()
        self.request_server.start()
        self.worker.start()

    def stop(self):
        self.worker.stop()
        self.request_server.stop()
        self.control_server.stop()
        logger.info("DriverWorkerProxy stopped")

    def _on_request(self, request: DriverRequest):
        """DriverRequest --> InferenceRequest."""
        inference_request = InferenceRequest.from_driver_request(request)
        return self.worker.submit_request(inference_request)

    def _poll_outputs(self, timeout: float = None) -> list[DriverRequestOutput]:
        """InferenceRequest --> DriverRequestOutput."""
        return [request.to_driver_request_output() for request in self.worker.poll_results(timeout=timeout)]
