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

import torch
import torch.multiprocessing as mp
from transformers import GenerationConfig

from vexact.config import ParallelConfig, PPInfo
from vexact.core.runtime_data import GenerationContext, InferencerOutput
from vexact.distributed.pp_messager import PPMessager
from vexact.utils.sys import find_available_port


def send_recv_gen_ctx(rank, pp_size, port):
    device = torch.device(f"cuda:{rank}")
    parallel_config = ParallelConfig(pipeline_parallel_size=pp_size, torch_distributed_port=port)
    pp_messager = PPMessager(
        pp_info=PPInfo(pp_rank=rank, pp_size=pp_size), parallel_config=parallel_config, device=device
    )
    if rank == 0:
        pp_messager.send_gen_ctx(
            GenerationContext(
                batch_input_ids=None,
                intermediate_tensors=torch.tensor([[0.2, 1.3, -0.1]], dtype=torch.bfloat16, device=device),
                query_start_loc=torch.tensor([0], dtype=torch.long, device=device),
                batch_position_ids=torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device),
                block_tables=torch.tensor([[22, 31]], dtype=torch.long, device=device),
                context_lens=torch.tensor([[3, 4]], dtype=torch.long, device=device),
                slot_mapping=torch.tensor([[2], [3], [5]], dtype=torch.long, device=device),
                max_seqlen_q=7,
                generation_configs=[
                    GenerationConfig(
                        max_new_tokens=20,
                        max_length=100,
                        do_sample=False,
                    )
                ],
                tokens_generated=[5, 4],
            )
        )
    elif rank == 1:
        gen_ctx = pp_messager.recv_gen_ctx()
        assert gen_ctx.batch_input_ids is None
        assert torch.equal(
            gen_ctx.intermediate_tensors, torch.tensor([[0.2, 1.3, -0.1]], dtype=torch.bfloat16, device=device)
        )
        assert torch.equal(gen_ctx.query_start_loc, torch.tensor([0], dtype=torch.long, device=device))
        assert torch.equal(gen_ctx.batch_position_ids, torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device))
        assert torch.equal(gen_ctx.block_tables, torch.tensor([[22, 31]], dtype=torch.long, device=device))
        assert torch.equal(gen_ctx.context_lens, torch.tensor([[3, 4]], dtype=torch.long, device=device))
        assert torch.equal(gen_ctx.slot_mapping, torch.tensor([[2], [3], [5]], dtype=torch.long, device=device))
        assert gen_ctx.max_seqlen_q == 7
        assert gen_ctx.generation_configs[0].max_new_tokens == 20
        assert gen_ctx.tokens_generated == [5, 4]


def test_send_recv_gen_ctx():
    pp_size = 2
    port = str(find_available_port())
    mp.spawn(
        send_recv_gen_ctx,
        args=(
            pp_size,
            port,
        ),
        nprocs=pp_size,
        join=True,
    )


def send_recv_infer_out(rank, pp_size, port):
    device = torch.device(f"cuda:{rank}")
    parallel_config = ParallelConfig(pipeline_parallel_size=pp_size, torch_distributed_port=port)
    pp_messager = PPMessager(
        pp_info=PPInfo(pp_rank=rank, pp_size=pp_size), parallel_config=parallel_config, device=device
    )
    if rank == 1:
        pp_messager.send_infer_out(
            InferencerOutput(
                token_ids=torch.tensor([1992, 301, 124], dtype=torch.long, device=device),
                logits=torch.tensor(
                    [
                        [[0.1, 1.2], [1.3, 1.1]],
                        [[0.3, 1.1], [3.8, 0.1]],
                    ],
                    dtype=torch.bfloat16,
                    device=device,
                ),
                logprobs=torch.empty(0, dtype=torch.bfloat16, device=device),
            )
        )
    elif rank == 0:
        infer_out = pp_messager.recv_infer_out()
        assert torch.equal(infer_out.token_ids, torch.tensor([1992, 301, 124], dtype=torch.long, device=device))
        assert torch.equal(
            infer_out.logits[0], torch.tensor([[0.1, 1.2], [1.3, 1.1]], dtype=torch.bfloat16, device=device)
        )
        assert torch.equal(
            infer_out.logits[1], torch.tensor([[0.3, 1.1], [3.8, 0.1]], dtype=torch.bfloat16, device=device)
        )


def test_send_recv_infer_out():
    pp_size = 2
    port = str(find_available_port())
    mp.spawn(
        send_recv_infer_out,
        args=(
            pp_size,
            port,
        ),
        nprocs=pp_size,
        join=True,
    )


def verify_pp_group_attributes(rank, pp_size, port):
    """Verify that prev/next messenger attributes are set correctly for ring topology."""
    device = torch.device(f"cuda:{rank}")
    parallel_config = ParallelConfig(pipeline_parallel_size=pp_size, torch_distributed_port=port)
    pp_messager = PPMessager(
        pp_info=PPInfo(pp_rank=rank, pp_size=pp_size), parallel_config=parallel_config, device=device
    )
    # Ring topology: rank -> (rank+1) % pp_size, and prev is (rank-1) % pp_size
    expected_next = (rank + 1) % pp_size
    expected_prev = (rank - 1) % pp_size
    assert pp_messager._next_msger_rank == expected_next, f"rank {rank}: next should be {expected_next}"
    assert pp_messager._prev_msger_rank == expected_prev, f"rank {rank}: prev should be {expected_prev}"
    assert pp_messager._next_msger_cpu_pair is not None
    assert pp_messager._next_msger_gpu_pair is not None
    assert pp_messager._prev_msger_cpu_pair is not None
    assert pp_messager._prev_msger_gpu_pair is not None


def test_pp_group_attributes_pp2():
    """Test ring topology attributes with pp_size=2."""
    pp_size = 2
    port = str(find_available_port())
    mp.spawn(verify_pp_group_attributes, args=(pp_size, port), nprocs=pp_size, join=True)


def test_pp_group_attributes_pp3():
    """Test ring topology attributes with pp_size=3."""
    pp_size = 3
    port = str(find_available_port())
    mp.spawn(verify_pp_group_attributes, args=(pp_size, port), nprocs=pp_size, join=True)
