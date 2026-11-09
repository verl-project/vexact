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

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vexact.batch_invariant_ops.kv_cache_context import KVCacheStore, set_kv_cache_context
from vexact.config import CacheConfig
from vexact.core.runtime_data import InputBuffers


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """
    Shape captured by a CUDA graph.

    `num_tokens` is the padded token bucket. `num_seqs` is the padded sequence
    count. A descriptor can replay any runtime batch with no more tokens or
    sequences than these limits.

    `max_seqlen_q` is the longest query run in any batch the graph may serve. Attention takes it as a host
    integer, so capture bakes it in; it must therefore be an upper bound for every such batch. The general
    descriptors use the whole token bucket. A pure decode step has one token per sequence, so the decode
    descriptors (`num_tokens == num_seqs`, dispatched only when the batch has one token per sequence) can use
    1, which is what lets vLLM's FA3 build pick its one-warpgroup decode tile.
    """

    num_tokens: int
    num_seqs: int
    max_seqlen_q: int


class CudaGraphManager:
    """
    CUDA graph capture/replay helper for VeXact `model.forward`.

    - Captures graphs keyed by padded sequence count and token bucket size.
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
        logits_for_last_tokens_only: bool = False,
        capture_decode_graphs: bool = False,
    ):
        self.model = model
        # When set, the model gets `logits_to_keep` from the static last_token_idx buffer, which the
        # caller refreshes from query_start_loc before every replay.
        self.logits_for_last_tokens_only = logits_for_last_tokens_only
        self.device = device
        self.cache_config = cache_config
        self.cache_store = cache_store
        self.capture_sizes = self.build_capture_sizes(max_size)
        # Always use a graph pool; allow an external one to be injected.
        self.pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[BatchExecutionDescriptor, Any] = {}
        self.forward_kwargs = forward_kwargs or {}
        self.input_buffers = input_buffers
        self._descriptors = [
            BatchExecutionDescriptor(
                num_tokens=capture_size,
                num_seqs=min(capture_size, self.input_buffers.max_num_seqs),
                max_seqlen_q=capture_size,
            )
            for capture_size in self.capture_sizes
        ]
        # One token per sequence. Size 1 is already covered by the general descriptor of the same shape.
        # These differ from the general descriptors only by the baked max_seqlen_q and capture the same
        # shapes, so they share the capture pool and cost far less memory than their count suggests
        # (13 -> 22 graphs on Qwen3-8B, no rise in reserved memory). They pay off only with an attention
        # kernel that acts on max_seqlen_q == 1: worth +4.9% end to end with vLLM's FA3 build and nothing
        # at all with the upstream wheel, so `capture_decode_graphs` is off unless asked for.
        self._decode_descriptors = (
            [
                BatchExecutionDescriptor(num_tokens=size, num_seqs=size, max_seqlen_q=1)
                for size in self.capture_sizes
                if 1 < size <= self.input_buffers.max_num_seqs
            ]
            if capture_decode_graphs
            else []
        )

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

    def capture_graphs(self) -> None:
        if not self.capture_sizes:
            raise RuntimeError("CudaGraphManger capture sizes are not set.")
        for desc in self._descriptors + self._decode_descriptors:
            self.capture(desc)

    def dispatch(self, num_seqs: int, total_tokens: int) -> BatchExecutionDescriptor | None:
        if total_tokens == num_seqs:  # one token per sequence: a pure decode step
            for desc in self._decode_descriptors:
                if desc.num_seqs >= num_seqs:
                    return desc
        for desc in self._descriptors:
            if desc.num_seqs >= num_seqs and desc.num_tokens >= total_tokens:
                return desc
        return None

    @torch.inference_mode()
    def capture(self, desc: BatchExecutionDescriptor) -> BatchExecutionDescriptor:
        """
        Capture a CUDA graph for a packed batch shape.
        Returns the graph key for replay.
        """
        num_seqs = desc.num_seqs
        capture_tokens = desc.num_tokens
        if num_seqs <= 0 or capture_tokens <= 0:
            raise ValueError(f"num_seqs and capture_tokens must be positive, got {num_seqs=}, {capture_tokens=}")
        if num_seqs > capture_tokens:
            raise ValueError(f"num_seqs cannot exceed capture_tokens, got {num_seqs=}, {capture_tokens=}")
        if num_seqs > self.input_buffers.max_num_seqs:
            raise ValueError(f"num_seqs exceeds input buffer capacity: {num_seqs} > {self.input_buffers.max_num_seqs}")
        if capture_tokens > self.input_buffers.max_num_batched_tokens:
            raise ValueError(
                "capture_tokens exceeds input buffer capacity: "
                f"{capture_tokens} > {self.input_buffers.max_num_batched_tokens}"
            )

        if desc in self.graphs:
            return desc

        static_inputs = {
            "input_ids": self.input_buffers.input_ids.gpu[:, :capture_tokens],
            "intermediate_tensors": self.input_buffers.intermediate_tensors.gpu[:, :capture_tokens, :],
            "position_ids": self.input_buffers.position_ids.gpu[:, :capture_tokens],
        }

        (
            block_tables,
            context_lens,
            slot_mapping,
            query_start_loc,
        ) = self.input_buffers.get_kv_context_tensors(num_seqs=num_seqs, num_tokens=capture_tokens)

        self.input_buffers.input_ids.gpu[:, :capture_tokens].zero_()
        self.input_buffers.position_ids.gpu[:, :capture_tokens].zero_()
        self.input_buffers.intermediate_tensors.gpu[:, :capture_tokens, :].zero_()
        block_tables.zero_()
        # prevent any KV writes during capture
        slot_mapping.fill_(-1)

        tokens_per_seq = torch.full((num_seqs,), capture_tokens // num_seqs, device=self.device, dtype=torch.int32)
        tokens_per_seq[: capture_tokens % num_seqs] += 1
        query_start_loc[0].zero_()
        query_start_loc[1:].copy_(tokens_per_seq.cumsum(0))
        context_lens.copy_(tokens_per_seq)
        if self.logits_for_last_tokens_only:
            static_inputs["logits_to_keep"] = self.input_buffers.fill_last_token_idx(num_seqs)

        # Warm up to ensure kernels are initialized before graph capture.
        set_kv_cache_context(
            is_paged_attn=True,
            key_cache=self.cache_store.key_cache,
            value_cache=self.cache_store.value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            max_seqlen_q=desc.max_seqlen_q,
        )
        _ = self.model(**static_inputs, **self.forward_kwargs)

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, self.pool):
            outputs = self.model(**static_inputs, **self.forward_kwargs)

        self.graphs[desc] = graph
        self.outputs[desc] = outputs
        return desc

    @torch.inference_mode()
    def replay(self, graph_key: BatchExecutionDescriptor) -> Any:
        """
        Replay a captured graph after copying new inputs and KV context into static buffers.

        Args:
            graph_key: sequence count and token bucket used during capture.
        """
        if graph_key not in self.graphs:
            raise KeyError(f"No captured graph for id {graph_key}. Captured: {list(self.graphs)}")

        self.graphs[graph_key].replay()
        return self.outputs[graph_key]
