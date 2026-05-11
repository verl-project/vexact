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

"""VeXact - High-level inference engine with tokenization support."""

import asyncio
import logging

from vexact.config import VeXactConfig
from vexact.core.request import DriverRequest, DriverRequestOutput
from vexact.utils.tokenizer import load_tokenizer
from vexact.worker.driver_client import DriverClient
from vexact.worker.worker_proc_manager import WorkerProcManager


logger = logging.getLogger(__name__)

# Exported symbols
__all__ = ["VeXact", "DriverRequest", "DriverRequestOutput"]


class VeXact:
    def __init__(self, config: VeXactConfig):
        """
        Initialize VeXact engine.

        Args:
            config: VeXactConfig instance with engine configuration
        """
        self.config = config

        # Load tokenizer
        logger.info(f"Loading tokenizer from: {config.model.model_path}")
        self.tokenizer = self._load_tokenizer()

        # Create worker proc manager if managed mode
        if config.driver.is_worker_proc_managed:
            self._worker_proc_manager = WorkerProcManager(config)

        # Create driver client
        self.driver_client = DriverClient(config.driver)

        logger.info("VeXact initialized")

    def _load_tokenizer(self):
        """
        Load HuggingFace tokenizer using utility function.

        Returns:
            Tokenizer instance
        """
        return load_tokenizer(self.config.model.model_path)

    def close(self):
        """Close the engine and clean up resources."""
        self.driver_client.close()
        if hasattr(self, "_worker_proc_manager"):
            self._worker_proc_manager.close()
        logger.info("VeXact closed")

    def sleep(self, tag: str = None):
        """
        Pause memory-saving regions to offload GPU memory to CPU.
        Only works if enable_memory_saver=True.
        """
        return self.driver_client.sleep(tag=tag)

    def wake_up(self, tag: str = None):
        """
        Resume memory-saving regions to restore GPU memory from CPU.
        Only works if enable_memory_saver=True.
        """
        return self.driver_client.wake_up(tag=tag)

    def get_prefix_cache_stats(self) -> dict:
        """Snapshot of prefix-cache counters since the driver started (or last
        weight update / sleep, which resets the counters).

        Returns a dict with: prefix_cache_enabled, hit_tokens, miss_tokens,
        hit_ratio, cached_blocks, free_blocks. Returns
        {"prefix_cache_enabled": False} on PP>1 (the prefix cache is disabled
        in that configuration).
        """
        return self.driver_client.get_prefix_cache_stats()

    async def generate(
        self,
        request: DriverRequest,
        timeout: float | None = None,
    ) -> DriverRequestOutput:
        """
        Async generate interface that accepts a DriverRequest and returns DriverRequestOutput.

        This method is safe for concurrent calls: each request gets its own result.

        Args:
            request: DriverRequest containing request_id, generation_config, and input_ids_list
            timeout: Timeout in seconds for waiting for the result (default: 60.0). None for no timeout.

        Returns:
            DriverRequestOutput with new_token_ids and optional new_logprobs

        Raises:
            asyncio.TimeoutError: If the request times out
            RuntimeError: If no output is received

        Example:
            >>> request = DriverRequest(
            ...     request_id="req_1",
            ...     generation_config=GenerationConfig(max_new_tokens=50),
            ...     input_ids_list=[1, 2, 3, 4],
            ... )
            >>> result = await engine.generate(request)
        """

        async def get_result():
            async for output in self.driver_client.generate(request):
                return output
            raise RuntimeError(f"No output received for request {request.request_id}")

        if timeout is None:
            return await get_result()
        return await asyncio.wait_for(get_result(), timeout=timeout)
