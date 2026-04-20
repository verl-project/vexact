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

"""
CUDAGraph utils
Reference: https://github.com/vllm-project/vllm/blob/d49899732edd3e6c011ec9f922601d919250a4d2/vllm/v1/worker/gpu/cudagraph_utils.py#L1
"""

from typing import Any

import torch
import torch.nn as nn

from vexact.batch_invariant_ops.kv_cache_context import KVCacheStore, set_kv_cache_context
from vexact.config import CacheConfig
from vexact.core.runtime_data import InputBuffers


class CudaGraphManager:
    """
    Decode-only CUDA graph capture/replay helper for VeXact `model.forward`.

    - Captures graphs for decode batches (seq_len=1) at specified batch sizes.
    - Uses external InputBuffers for static inputs/context.
    - Caller manages real KV contents; the manager replays with updated buffers.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        input_buffers: InputBuffers,
        cache_config: CacheConfig,
        cache_store: KVCacheStore,
        max_size: int,
        pool: Any | None = None,
        forward_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.device = device
        self.cache_config = cache_config
        self.cache_store = cache_store
        self.capture_sizes = self.build_capture_sizes(max_size)
        # Always use a graph pool; allow an external one to be injected.
        self.pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, Any] = {}
        self.forward_kwargs = forward_kwargs or {}
        self.input_buffers = input_buffers

    @staticmethod
    def build_capture_sizes(max_size: int | None) -> list[int]:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive or None, got {max_size}")
        sizes: list[int] = []
        size = 1
        while size < max_size:
            sizes.append(size)
            size *= 2
        sizes.append(size)
        return sizes

    def select_capture_size(self, total_tokens: int) -> int | None:
        """
        public method for inferencer to select a proper graph size to replay
        """
        for size in self.capture_sizes:
            if size >= total_tokens:
                return size
        return None

    def capture_graphs(self) -> None:
        if not self.capture_sizes:
            raise RuntimeError("CudaGraphManger capture sizes are not set.")
        for capture_size in self.capture_sizes:
            self.capture(capture_size)

    @torch.inference_mode()
    def capture(self, capture_size: int) -> int:
        """
        Capture a CUDA graph for decode-only batch of size `capture_size`.
        Returns capture_size for replay.
        """

        static_inputs = {
            "input_ids": self.input_buffers.input_ids.gpu[:, :capture_size],
            "intermediate_tensors": self.input_buffers.intermediate_tensors.gpu[:, :capture_size, :],
            "position_ids": self.input_buffers.position_ids.gpu[:, :capture_size],
        }

        # slice kv context to capture_size size
        (
            block_tables,
            context_lens,
            slot_mapping,
            query_start_loc,
        ) = self.input_buffers.get_kv_context_tensors(capture_size)

        # we set context_lens to 0 instead so that no KV cache would be actually touched in capture
        context_lens.zero_()
        # prevent any KV writes during capture
        slot_mapping.fill_(-1)
        # with position_ids as all 0, legit query_start_loc is just [0, 1, 2, ...]
        query_start_loc.copy_(torch.arange(query_start_loc.numel(), device=self.device, dtype=torch.int32))

        # Warm up to ensure kernels are initialized before graph capture.
        set_kv_cache_context(
            is_paged_attn=True,
            key_cache=self.cache_store.key_cache,
            value_cache=self.cache_store.value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            max_seqlen_q=1,
        )
        _ = self.model(**static_inputs, **self.forward_kwargs)

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, self.pool):
            outputs = self.model(**static_inputs, **self.forward_kwargs)

        self.graphs[capture_size] = graph
        self.outputs[capture_size] = outputs
        return capture_size

    @torch.inference_mode()
    def replay(self, capture_size: int) -> Any:
        """
        Replay a captured graph after copying new inputs and KV context into static buffers.

        Args:
            capture_size: batch size used during capture.
            gen_ctx: decode-only generation context with packed inputs and KV metadata.
        """
        if capture_size not in self.graphs:
            raise KeyError(f"No captured graph for id {capture_size}. Captured: {list(self.graphs)}")

        self.graphs[capture_size].replay()
        return self.outputs[capture_size]
