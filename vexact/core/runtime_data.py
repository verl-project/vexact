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

from dataclasses import dataclass
from typing import NamedTuple, Optional, Union

import torch
from torch import Tensor
from transformers import GenerationConfig


class TensorBuffer(NamedTuple):
    """Paired CPU (pinned) and GPU buffer for efficient H2D transfers"""

    cpu: torch.Tensor
    gpu: torch.Tensor

    @classmethod
    def create(
        cls, *size: int, dtype: torch.dtype, device: torch.device, fill_value: Optional[Union[int, float]] = None
    ):
        """Factory method to create paired buffers

        Args:
            *size: Shape dimensions (e.g., 256, 512 for a 2D tensor)
            dtype: Data type for both tensors
            device: Target GPU device
            fill_value: Optional fill value for GPU tensor (None = empty)
        """
        cpu = torch.empty(size, dtype=dtype, pin_memory=True, device="cpu")
        if fill_value is not None:
            gpu = torch.full(size, fill_value, dtype=dtype, device=device)
        else:
            gpu = torch.empty(size, dtype=dtype, device=device)
        return cls(cpu=cpu, gpu=gpu)

    def to_device(self, size: Optional[Union[slice, tuple]] = None, non_blocking: bool = True) -> torch.Tensor:
        """Transfer CPU data to GPU and return GPU view

        Args:
            size: Optional slice/indexing (e.g., slice(None, batch_size) or (slice(None, 10), slice(None, 20)))
            non_blocking: Whether to use non-blocking transfer

        Returns:
            GPU tensor view
        """
        if size is None:
            self.gpu.copy_(self.cpu, non_blocking=non_blocking)
            return self.gpu
        else:
            cpu_view = self.cpu[size]
            gpu_view = self.gpu[size]
            gpu_view.copy_(cpu_view, non_blocking=non_blocking)
            return gpu_view


class InputBuffers:
    """Pre-allocated buffers for generation context preparation"""

    def __init__(
        self,
        device: torch.device,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_blocks_per_req: int,
        hidden_size: int,
    ):
        self.device = device
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_blocks_per_req = max_blocks_per_req

        # attn
        self.slot_mapping = TensorBuffer.create(max_num_batched_tokens, dtype=torch.int32, device=device)
        self.context_lens = TensorBuffer.create(max_num_seqs, dtype=torch.int32, device=device)
        self.query_start_loc = TensorBuffer.create(max_num_seqs + 1, dtype=torch.int32, device=device)
        self.block_tables = TensorBuffer.create(
            max_num_seqs, max_blocks_per_req, dtype=torch.int32, device=device, fill_value=-1
        )

        # model forward
        # we always expect input batch to be packed together so batch size is 1
        self.input_ids = TensorBuffer.create(1, max_num_batched_tokens, dtype=torch.long, device=device, fill_value=0)
        # intermediate forward inputs for PP rank > 0
        self.intermediate_tensors = TensorBuffer.create(
            1, max_num_batched_tokens, hidden_size, dtype=torch.bfloat16, device=device, fill_value=0
        )
        self.position_ids = TensorBuffer.create(
            1, max_num_batched_tokens, dtype=torch.long, device=device, fill_value=0
        )

    def get_kv_context_tensors(self, num_seqs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return KV context slices for a decode-only batch (num_tokens == num_seqs).

        For prefill / chunked-prefill batches, slot_mapping needs token-level slicing;
        do not reuse this helper without extending the API.
        """
        return (
            self.block_tables.gpu[:num_seqs, :],
            self.context_lens.gpu[:num_seqs],
            self.slot_mapping.gpu[:num_seqs],
            self.query_start_loc.gpu[: num_seqs + 1],
        )


@dataclass
class GenerationContext:
    batch_input_ids: Optional[Tensor]
    intermediate_tensors: Optional[Tensor]
    query_start_loc: Tensor
    batch_position_ids: Tensor
    block_tables: Tensor
    context_lens: Tensor
    slot_mapping: Tensor
    max_seqlen_q: int | None
    generation_configs: list[GenerationConfig]
    tokens_generated: list[int]
    is_decode_only: bool = False


@dataclass
class InferencerOutput:
    token_ids: Tensor
    logits: Tensor
    logprobs: Tensor
