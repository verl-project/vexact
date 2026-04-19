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

"""DriverClient for submitting requests and controlling workers via IPC."""

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from vexact.config import DriverConfig
from vexact.core.request import DriverRequest, DriverRequestOutput
from vexact.ipc.control_channel import ControlChannelClient
from vexact.ipc.request_channel import RequestChannelClient


logger = logging.getLogger(__name__)


class BaseDriverClient(ABC):
    """Abstract base class for driver clients."""

    @abstractmethod
    def close(self):
        """Close the client and clean up resources."""
        pass

    @abstractmethod
    async def generate(self, request: DriverRequest) -> AsyncIterator[DriverRequestOutput]:
        """Submit a request and stream outputs asynchronously."""
        pass

    @abstractmethod
    def sleep(self, tag: str = None) -> Any:
        """Pause memory-saving regions to offload GPU memory to CPU."""
        pass

    @abstractmethod
    def wake_up(self, tag: str = None) -> Any:
        """Resume memory-saving regions to restore GPU memory from CPU."""
        pass


class DriverClient(BaseDriverClient):
    """IPC-based driver client for remote driver workers.

    Uses ZMQ channels for communication with driver worker proxies.

    Args:
        config: DriverConfig with request_address and control_addresses
    """

    def __init__(self, config: DriverConfig):
        self._config = config
        self.request_channel = RequestChannelClient(config.request_address)
        self.control_channel = ControlChannelClient(config.control_addresses)
        logger.info(f"DriverClient initialized (remote mode) at {config.request_address}")

    def close(self):
        """Close the client and clean up resources."""
        self.request_channel.close()
        self.control_channel.close()
        logger.info("DriverClient closed")

    async def generate(self, request: DriverRequest) -> AsyncIterator[DriverRequestOutput]:
        """Submit a request and stream outputs asynchronously.

        Args:
            request: The DriverRequest to submit

        Yields:
            DriverRequestOutput objects as they become available
        """
        async for output in self.request_channel.submit(request):
            yield output

    def _execute(self, method: str, *args, **kwargs) -> list[Any]:
        """Execute a method on all workers collectively."""
        return self.control_channel.collective_rpc(method, *args, **kwargs)

    def sleep(self, tag: str = None) -> list[Any]:
        """Pause memory-saving regions on all workers to offload GPU memory to CPU."""
        return self._execute("sleep", tag=tag)

    def wake_up(self, tag: str = None) -> list[Any]:
        """Resume memory-saving regions on all workers to restore GPU memory from CPU."""
        return self._execute("wake_up", tag=tag)

    def receive_weights(self) -> list[Any]:
        """Receive model weights via IPC on all workers."""
        return self._execute("receive_weights")
