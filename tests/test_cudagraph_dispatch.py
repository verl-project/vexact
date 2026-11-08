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

"""A captured graph bakes in `max_seqlen_q`, so dispatch must never hand a batch a graph whose value is too
small for it."""

import pytest

from vexact.inferencer.cudagraph_utils import BatchExecutionDescriptor, CudaGraphManager


def _manager(max_num_batched_tokens: int, max_num_seqs: int) -> CudaGraphManager:
    manager = CudaGraphManager.__new__(CudaGraphManager)
    sizes = CudaGraphManager.build_capture_sizes(max_num_batched_tokens)
    manager.capture_sizes = sizes
    manager._descriptors = [
        BatchExecutionDescriptor(num_tokens=s, num_seqs=min(s, max_num_seqs), max_seqlen_q=s) for s in sizes
    ]
    manager._decode_descriptors = [
        BatchExecutionDescriptor(num_tokens=s, num_seqs=s, max_seqlen_q=1) for s in sizes if 1 < s <= max_num_seqs
    ]
    return manager


@pytest.mark.parametrize("max_num_batched_tokens, max_num_seqs", [(4096, 512), (2048, 256), (8, 8)])
def test_decode_batches_get_a_max_seqlen_q_of_one(max_num_batched_tokens, max_num_seqs):
    manager = _manager(max_num_batched_tokens, max_num_seqs)
    for num_seqs in range(1, max_num_seqs + 1):
        desc = manager.dispatch(num_seqs=num_seqs, total_tokens=num_seqs)  # one token per sequence
        assert desc is not None
        assert desc.max_seqlen_q == 1
        assert desc.num_seqs >= num_seqs and desc.num_tokens >= num_seqs


@pytest.mark.parametrize("max_num_batched_tokens, max_num_seqs", [(4096, 512), (2048, 256)])
def test_dispatched_max_seqlen_q_bounds_every_batch(max_num_batched_tokens, max_num_seqs):
    """For a batch of `num_seqs` sequences and `total_tokens` tokens, one sequence can hold at most
    `total_tokens - num_seqs + 1` of them; the graph's baked value must not be below that."""
    manager = _manager(max_num_batched_tokens, max_num_seqs)
    for num_seqs in (1, 2, 7, 64, 255, 256):
        for total_tokens in (num_seqs, num_seqs + 1, num_seqs * 2, 512, 4096):
            if total_tokens < num_seqs or total_tokens > max_num_batched_tokens or num_seqs > max_num_seqs:
                continue
            desc = manager.dispatch(num_seqs=num_seqs, total_tokens=total_tokens)
            assert desc is not None, (num_seqs, total_tokens)
            assert desc.max_seqlen_q >= total_tokens - num_seqs + 1, (num_seqs, total_tokens, desc)


def test_oversized_batches_have_no_graph():
    manager = _manager(1024, 128)
    assert manager.dispatch(num_seqs=129, total_tokens=129) is None
    assert manager.dispatch(num_seqs=4, total_tokens=2048) is None
