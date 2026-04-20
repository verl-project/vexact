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

import logging
import os
from dataclasses import dataclass, fields
from datetime import timedelta

import torch

from vexact.config import ParallelConfig, PPInfo
from vexact.core.runtime_data import GenerationContext, InferencerOutput


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorMeta:
    shape: torch.Size
    dtype: torch.dtype


class PPMessager:
    def __init__(self, pp_info: PPInfo, parallel_config: ParallelConfig, device: torch.device):
        self._device = device
        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", parallel_config.torch_distributed_addr)
            os.environ.setdefault("MASTER_PORT", parallel_config.torch_distributed_port)
            torch.distributed.init_process_group(
                backend="nccl", rank=pp_info.pp_rank, world_size=pp_info.pp_size, timeout=timedelta(minutes=30)
            )
            # When we initialize our own process group, pp_rank == world_rank
            self._init_pp_groups_standalone(pp_info)
        else:
            # Distributed already initialized (e.g., by VeRL).
            # new_group is collective - ALL ranks must participate in creating ALL groups.
            # TODO: remove this mode when OSS-version of veRL fully compatible with standalone managed server.
            self._init_pp_groups_within_world(pp_info)

    def _init_pp_groups_standalone(self, pp_info: PPInfo):
        """Initialize PP groups when VeXact owns the process group (pp_rank == world_rank).

        Creates a ring topology where each rank communicates with its neighbors:

            pp_size=2:  [0] <--> [1]
            pp_size=3:  [0] --> [1] --> [2] --> [0]
        """
        self._create_pp_pair_groups(pp_info, list(range(pp_info.pp_size)))

    def _init_pp_groups_within_world(self, pp_info: PPInfo):
        """Initialize PP groups when distributed is already initialized (e.g., by VeRL).

        new_group is collective - ALL ranks must call it with the same arguments.
        Creates PP groups for ALL replicas; each rank stores only its own groups.

        Assumes contiguous world_ranks per replica: world_rank = replica * pp_size + pp_rank

            world_size=6, pp_size=2:
              replica 0: [0] <--> [1]
              replica 1: [2] <--> [3]
              replica 2: [4] <--> [5]

            world_size=8, pp_size=4:
              replica 0: [0] --> [1] --> [2] --> [3] --> [0]
              replica 1: [4] --> [5] --> [6] --> [7] --> [4]
        """
        world_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        num_replicas = world_size // pp_info.pp_size
        my_replica = world_rank // pp_info.pp_size

        for replica in range(num_replicas):
            base = replica * pp_info.pp_size
            world_ranks = [base + i for i in range(pp_info.pp_size)]
            self._create_pp_pair_groups(pp_info, world_ranks, store=(replica == my_replica))

    def _create_pp_pair_groups(self, pp_info: PPInfo, world_ranks: list[int], store: bool = True):
        """Create communication groups for adjacent PP rank pairs.

        Args:
            pp_info: Pipeline parallel configuration.
            world_ranks: World ranks for this PP group (world_ranks[i] = world rank of pp_rank i).
            store: Whether to store the groups (False when creating groups for other replicas).
        """
        for pp_rank in range(pp_info.pp_size):
            next_pp_rank = (pp_rank + 1) % pp_info.pp_size
            pair = [world_ranks[pp_rank], world_ranks[next_pp_rank]]
            cpu_group = torch.distributed.new_group(backend="gloo", ranks=pair, timeout=timedelta(days=7))
            gpu_group = torch.distributed.new_group(backend="nccl", ranks=pair, timeout=timedelta(days=7))

            if store and pp_info.pp_rank == pp_rank:
                self._next_msger_rank, self._next_msger_cpu_pair, self._next_msger_gpu_pair = (
                    pair[1],
                    cpu_group,
                    gpu_group,
                )
            elif store and pp_info.pp_rank == next_pp_rank:
                self._prev_msger_rank, self._prev_msger_cpu_pair, self._prev_msger_gpu_pair = (
                    pair[0],
                    cpu_group,
                    gpu_group,
                )

    def send_gen_ctx(self, gen_ctx: GenerationContext):
        ctx_meta = []
        ctx_tensors = []
        # Avoid dataclasses.asdict() because it recursively deep-copies tensors and nested objects.
        for field in fields(gen_ctx):
            name = field.name
            tensor_data = getattr(gen_ctx, name)
            if isinstance(tensor_data, torch.Tensor):
                ctx_meta.append((name, TensorMeta(tensor_data.shape, tensor_data.dtype)))
                ctx_tensors.append(tensor_data)
            else:
                ctx_meta.append((name, tensor_data))
        self._send_object_list(ctx_meta)
        self._send_batch_tensor(ctx_tensors)

    def recv_gen_ctx(self) -> GenerationContext:
        ctx_meta = [None] * len(fields(GenerationContext))
        self._recv_object_list(ctx_meta)
        ctx_tensors = []
        ctx_dict = {}
        for name, tensor_meta in ctx_meta:
            if isinstance(tensor_meta, TensorMeta):
                empty_tensor = torch.empty(tensor_meta.shape, dtype=tensor_meta.dtype, device=self._device)
                ctx_tensors.append(empty_tensor)
                ctx_dict[name] = empty_tensor
            else:
                ctx_dict[name] = tensor_meta

        self._recv_batch_tensor(ctx_tensors)
        return GenerationContext(**ctx_dict)

    def send_infer_out(self, infer_out: InferencerOutput):
        token_ids_meta = TensorMeta(infer_out.token_ids.shape, infer_out.token_ids.dtype)
        logits_meta = TensorMeta(infer_out.logits.shape, infer_out.logits.dtype)
        logprobs_meta = TensorMeta(infer_out.logprobs.shape, infer_out.logprobs.dtype)
        infer_out_meta = [token_ids_meta, logits_meta, logprobs_meta]
        infer_out_tensors = [infer_out.token_ids, infer_out.logits, infer_out.logprobs]
        self._send_object_list(infer_out_meta)
        self._send_batch_tensor(infer_out_tensors)

    def recv_infer_out(self) -> InferencerOutput:
        infer_out_meta = [None] * 3
        self._recv_object_list(infer_out_meta)
        token_ids = torch.empty(infer_out_meta[0].shape, dtype=infer_out_meta[0].dtype, device=self._device)
        logits = torch.empty(infer_out_meta[1].shape, dtype=infer_out_meta[1].dtype, device=self._device)
        logprobs = torch.empty(infer_out_meta[2].shape, dtype=infer_out_meta[2].dtype, device=self._device)
        infer_out_tensors = [token_ids, logits, logprobs]
        self._recv_batch_tensor(infer_out_tensors)
        return InferencerOutput(token_ids=token_ids, logits=logits, logprobs=logprobs)

    def _send_object_list(self, obj_lst: list):
        torch.distributed.send_object_list(
            obj_lst, dst=self._next_msger_rank, group=self._next_msger_cpu_pair, device=torch.device("cpu")
        )

    def _recv_object_list(self, obj_lst: list):
        torch.distributed.recv_object_list(
            obj_lst, src=self._prev_msger_rank, group=self._prev_msger_cpu_pair, device=torch.device("cpu")
        )

    def _send_batch_tensor(self, tensor_lst: list[torch.Tensor]):
        p2p_op_lst = [
            torch.distributed.P2POp(
                torch.distributed.isend, tensor, peer=self._next_msger_rank, group=self._next_msger_gpu_pair
            )
            for tensor in tensor_lst
        ]
        reqs = torch.distributed.batch_isend_irecv(p2p_op_lst)

        for req in reqs:
            req.wait()

    def _recv_batch_tensor(self, tensor_lst: list[torch.Tensor]):
        p2p_op_lst = [
            torch.distributed.P2POp(
                torch.distributed.irecv, tensor, peer=self._prev_msger_rank, group=self._prev_msger_gpu_pair
            )
            for tensor in tensor_lst
        ]
        reqs = torch.distributed.batch_isend_irecv(p2p_op_lst)

        for req in reqs:
            req.wait()

        return tensor_lst
