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

import pytest
import torch
from transformers import GenerationConfig

from vexact.batch_invariant_ops.kv_cache_context import KVCacheManager
from vexact.config import CacheConfig, PPInfo, SchedulerConfig
from vexact.core.request import InferenceRequest, RequestStatus
from vexact.core.runtime_data import InferencerOutput
from vexact.core.scheduler import Scheduler


@pytest.fixture
def generation_config():
    return GenerationConfig(
        max_length=32,
        max_new_tokens=8,
        eos_token_id=2,
        pad_token_id=0,
    )


@pytest.fixture
def make_request(generation_config):
    def _make_request(idx: int) -> InferenceRequest:
        return InferenceRequest(
            request_id=f"req-{idx}",
            generation_config=generation_config,
            input_ids_list=[203, 102, 234],
        )

    return _make_request


@pytest.fixture
def kv_cache_manager():
    def _builder(page_size=32, max_blocks=4):
        cache_config = CacheConfig(page_size=page_size, max_cache_blocks=max_blocks)
        return KVCacheManager(cache_config)

    return _builder


def test_scheduler_batches_and_completes_requests(make_request, kv_cache_manager):
    scheduler_config = SchedulerConfig(max_num_batched_tokens=4, max_queue_size=4, enable_chunked_prefill=False)
    scheduler = Scheduler(
        config=scheduler_config,
        kv_cache_manager=kv_cache_manager(),
        pp_info=PPInfo(1, 0),
    )

    requests = [None, None, None]
    requests[0] = make_request(0)
    requests[1] = make_request(1)
    requests[2] = make_request(2)
    assert scheduler.submit_request(requests[0])
    assert scheduler.submit_request(requests[1])
    assert scheduler.submit_request(requests[2])

    # First schedule: req-0 starts prefill (completes prefill since chunked_prefill=False)
    scheduler_output = scheduler.schedule()
    first_batch = scheduler_output.batch_to_infer
    assert {r.request_id for r in first_batch} == {requests[0].request_id}
    assert scheduler.queue_size() == 2
    for req in first_batch:
        assert req.status == RequestStatus.RUNNING
        assert req.block_ids

    # Second schedule: req-0 moves to decode, and req-1 joins from queue
    scheduler_output = scheduler.schedule()
    batch_to_update = scheduler_output.batch_to_update
    assert {r.request_id for r in batch_to_update} == {requests[0].request_id, requests[1].request_id}

    # Update with EOS token for req-0 and a regular token for req-1
    # req-0 finishes, req-1 continues
    infer_result = InferencerOutput(
        token_ids=torch.tensor([2, 999], dtype=torch.long, device=torch.device("cpu")),
        logits=torch.empty(0, dtype=torch.bfloat16, device=torch.device("cpu")),
        logprobs=torch.empty(0, dtype=torch.bfloat16, device=torch.device("cpu")),
    )
    scheduler.update(batch_to_update, infer_result)

    # Third schedule: req-1 is in decode, req-2 joins from queue
    scheduler_output = scheduler.schedule()
    next_batch = scheduler_output.batch_to_infer
    assert {r.request_id for r in next_batch} == {requests[1].request_id, requests[2].request_id}
    total_active_requests = scheduler.total_inflight_request_count()
    assert total_active_requests == 2

    # Fourth schedule: both in decode mode
    scheduler_output = scheduler.schedule()
    batch_to_update = scheduler_output.batch_to_update

    # Finish both req-1 and req-2 with EOS tokens
    infer_result = InferencerOutput(
        token_ids=torch.tensor([2, 2], dtype=torch.long, device=torch.device("cpu")),
        logits=torch.empty(0, dtype=torch.bfloat16, device=torch.device("cpu")),
        logprobs=torch.empty(0, dtype=torch.bfloat16, device=torch.device("cpu")),
    )
    scheduler.update(batch_to_update, infer_result)

    for req in requests:
        assert req.block_ids == []

    # All completed requests should be retrievable from the result queue.
    # Results are queued per update() call, so we need to poll multiple times.
    completed_results = []
    completed_results.extend(scheduler.poll_results(timeout=1.0))  # First batch: req-0
    completed_results.extend(scheduler.poll_results(timeout=1.0))  # Second batch: req-1, req-2
    assert len(completed_results) == 3
    completed_ids = {r.request_id for r in completed_results}
    assert completed_ids == {r.request_id for r in requests}


def test_scheduler_respects_queue_capacity(make_request, kv_cache_manager):
    scheduler_config = SchedulerConfig(max_num_batched_tokens=1, max_queue_size=1, enable_chunked_prefill=False)
    scheduler = Scheduler(
        config=scheduler_config,
        kv_cache_manager=kv_cache_manager(),
        pp_info=PPInfo(1, 0),
    )
    assert scheduler.submit_request(make_request(0))  # Should succeed
    failed_request = make_request(1)
    assert not scheduler.submit_request(failed_request)  # Should fail and put in result queue
    assert failed_request.status == RequestStatus.FAILED
    # Failed request should be in result queue
    failed_results = scheduler.poll_results(timeout=0.1)
    assert len(failed_results) == 1
    assert failed_results[0].request_id == failed_request.request_id


def test_scheduler_requeues_when_kv_cache_full(kv_cache_manager):
    """With incremental allocation, a request with many prompt tokens can exhaust blocks.

    Use page_size=2 so that 3-token prompts need 2 blocks each. With max_blocks=2,
    only one request fits at activation time.
    """
    generation_config = GenerationConfig(max_length=32, max_new_tokens=8, eos_token_id=2, pad_token_id=0)

    def _make(idx):
        return InferenceRequest(
            request_id=f"req-{idx}",
            generation_config=generation_config,
            input_ids_list=[203, 102, 234],
        )

    scheduler_config = SchedulerConfig(max_num_batched_tokens=6, max_queue_size=2, enable_chunked_prefill=False)
    scheduler = Scheduler(
        config=scheduler_config,
        kv_cache_manager=kv_cache_manager(page_size=2, max_blocks=2),
        pp_info=PPInfo(1, 0),
    )

    first_request = _make(0)
    second_request = _make(1)
    assert scheduler.submit_request(first_request)
    assert scheduler.submit_request(second_request)

    scheduler_output = scheduler.schedule()
    first_batch = scheduler_output.batch_to_infer
    assert {req.request_id for req in first_batch} == {first_request.request_id}
    assert scheduler.queue_size() == 1  # Second request requeued while cache is full.
    assert second_request.status == RequestStatus.PENDING
    assert second_request.block_ids == []

    # Free cache by completing the first request, then ensure the pending one is retried.
    # Second schedule: first_request moves to decode
    scheduler_output = scheduler.schedule()
    batch_to_update = scheduler_output.batch_to_update

    # Update with EOS token to finish first_request
    infer_result = InferencerOutput(
        token_ids=torch.tensor([2], dtype=torch.long, device=torch.device("cpu")),
        logits=torch.empty(0, dtype=torch.bfloat16, device=torch.device("cpu")),
        logprobs=torch.empty(0, dtype=torch.bfloat16, device=torch.device("cpu")),
    )
    scheduler.update(batch_to_update, infer_result)

    # Third schedule: second_request should now be able to get cache
    scheduler_output = scheduler.schedule()
    next_batch = scheduler_output.batch_to_infer
    assert {req.request_id for req in next_batch} == {second_request.request_id}
    assert second_request.status == RequestStatus.RUNNING
    assert second_request.block_ids


# for testing max prefill number and chunked prefill
def test_scheduler_chunked_prefill_progresses_and_switches_to_decode(make_request, kv_cache_manager):
    # Phase 1: Test with max_num_batched_tokens=8, max_num_prefill_seqs=2
    # Should accept 2 prefill requests with tokens_this_step=3 each
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=8, max_queue_size=4, enable_chunked_prefill=True, max_num_prefill_seqs=2
    )
    scheduler = Scheduler(
        config=scheduler_config,
        kv_cache_manager=kv_cache_manager(),
        pp_info=PPInfo(1, 0),
    )

    first_request = make_request(0)
    second_request = make_request(1)
    assert scheduler.submit_request(first_request)
    assert scheduler.submit_request(second_request)

    # First schedule: should add 2 requests with 3 tokens each (total 6 tokens)
    scheduler_output = scheduler.schedule()
    batch1 = scheduler_output.batch_to_infer
    assert len(batch1) == 2
    assert {r.request_id for r in batch1} == {first_request.request_id, second_request.request_id}
    assert first_request.tokens_this_step == 3
    assert second_request.tokens_this_step == 3
    assert first_request.num_computed_tokens == 3
    assert second_request.num_computed_tokens == 3

    # Phase 2: Test with max_num_batched_tokens=2, max_num_prefill_seqs=1
    # Should accept only 1 prefill request with tokens_this_step=2
    scheduler_config2 = SchedulerConfig(
        max_num_batched_tokens=2, max_queue_size=4, enable_chunked_prefill=True, max_num_prefill_seqs=1
    )
    scheduler2 = Scheduler(
        config=scheduler_config2,
        kv_cache_manager=kv_cache_manager(),
        pp_info=PPInfo(1, 0),
    )

    third_request = make_request(2)
    fourth_request = make_request(3)
    scheduler2.submit_request(third_request)
    scheduler2.submit_request(fourth_request)

    # First schedule: should add 1 request with 2 tokens
    scheduler_output = scheduler2.schedule()
    batch2 = scheduler_output.batch_to_infer
    assert len(batch2) == 1
    assert {r.request_id for r in batch2} == {third_request.request_id}
    assert third_request.tokens_this_step == 2
    assert third_request.num_computed_tokens == 2


def test_scheduler_update_processes_outputs_and_finishes_requests(kv_cache_manager):
    """Test that scheduler.update() correctly processes outputs and finishes requests."""
    scheduler_config = SchedulerConfig(max_num_batched_tokens=4, max_queue_size=4, enable_chunked_prefill=False)
    scheduler = Scheduler(
        config=scheduler_config,
        kv_cache_manager=kv_cache_manager(),
        pp_info=PPInfo(1, 0),
    )

    generation_config = GenerationConfig(eos_token_id=23, max_new_tokens=10, output_logits=True, output_scores=False)
    infer_requests = [
        InferenceRequest(
            request_id="req01",
            generation_config=generation_config,
            generated_tokens=[],
            input_ids_list=[234, 1897],
        ),
        InferenceRequest(
            request_id="req02",
            generation_config=generation_config,
            generated_tokens=[],
            input_ids_list=[15287, 344, 765],
        ),
    ]

    # Prepare requests by allocating KV cache and setting up state
    for request in infer_requests:
        block_hashes, num_prefix_hit_blocks = scheduler._kv_cache_manager.plan_prefix_cache(
            request.input_ids_list
        )
        scheduler._activate_request(request, block_hashes, num_prefix_hit_blocks)
        # Simulate that prefill is already complete so update() takes the decode path.
        # _activate_request sets num_computed_tokens to the cache-derived value (0
        # in this test); we want it at input_len for this unit test of update().
        request.num_computed_tokens = len(request.input_ids_list)

    # Set generated tokens after prepare (which resets them)
    infer_requests[1].generated_tokens = [78, 22]

    # Add to active requests
    for request in infer_requests:
        scheduler._inflight_batches[0].active_requests[request.request_id] = request

    infer_result = InferencerOutput(
        token_ids=torch.tensor([233, 23], dtype=torch.long, device=torch.device("cpu")),
        logits=torch.tensor([[0.01, 1.23, 0.13], [-0.3, 1.4, 3.1]], dtype=torch.bfloat16, device=torch.device("cpu")),
        logprobs=torch.empty(0, dtype=torch.bfloat16, device=torch.device("cpu")),
    )

    scheduler.update(infer_requests, infer_result)

    # req02 should be finished because it generated EOS token (23)
    assert infer_requests[1].is_finished
    assert infer_requests[1].generated_tokens == [78, 22, 23]

    # req01 should still be active with new token
    assert not infer_requests[0].is_finished
    assert infer_requests[0].generated_tokens == [233]

    # req02 should be removed from active requests
    assert "req02" not in scheduler._inflight_batches[0].active_requests
    assert "req01" in scheduler._inflight_batches[0].active_requests

    # req02 should be in result queue
    completed_results = scheduler.poll_results(timeout=1.0)
    assert len(completed_results) == 1
    assert completed_results[0] is infer_requests[1]


# test decode and prefill order in scheduler
def test_scheduler_decode_before_prefill_ordering(make_request, kv_cache_manager):
    """Test that scheduler always schedules decode requests before prefill requests."""
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=5, max_queue_size=10, enable_chunked_prefill=True, max_num_prefill_seqs=2
    )
    scheduler = Scheduler(
        config=scheduler_config,
        kv_cache_manager=kv_cache_manager(),
        pp_info=PPInfo(1, 0),
    )

    # Create 5 requests
    req0 = make_request(0)
    req1 = make_request(1)
    req2 = make_request(2)
    req3 = make_request(3)
    req4 = make_request(4)

    # Submit first 2 requests
    assert scheduler.submit_request(req0)
    assert scheduler.submit_request(req1)

    # Schedule 1: Should add req0 and req1 for prefill (2 prefill requests, max_num_prefill_seqs=2)
    scheduler_output = scheduler.schedule()
    batch1 = scheduler_output.batch_to_infer
    assert len(batch1) == 2
    assert batch1[0].request_id == req0.request_id
    assert batch1[1].request_id == req1.request_id
    assert len(req0.generated_tokens) == 0
    assert len(req1.generated_tokens) == 0

    # Simulate decode: update with generated tokens for req0 and req1
    # This will move them to decode phase
    req0.generated_tokens.append(10)

    # Submit 3 more requests (req2, req3, req4) - these will be in prefill phase
    assert scheduler.submit_request(req2)
    assert scheduler.submit_request(req3)
    assert scheduler.submit_request(req4)

    # Schedule 2: Should have decode requests (req0, req1) BEFORE prefill requests (req2, req3)
    scheduler_output3 = scheduler.schedule()
    batch2 = scheduler_output3.batch_to_infer
    assert len(batch2) == 3

    # Verify strict ordering: req0 -> req1 -> req2 -> req3
    assert batch2[0].request_id == req0.request_id, f"Expected req0 at index 0, got {batch2[0].request_id}"
    assert batch2[1].request_id == req1.request_id, f"Expected req1 at index 1, got {batch2[1].request_id}"
    assert batch2[2].request_id == req2.request_id, f"Expected req2 at index 2, got {batch2[2].request_id}"

    # Verify phase: first 2 are decode, last 2 are prefill
    assert len(batch2[0].generated_tokens) > 0  # req0 is in decode phase
    assert len(batch2[1].generated_tokens) == 0  # req1 is in decode phase
    assert len(batch2[2].generated_tokens) == 0  # req2 is in prefill phase


def test_scheduler_preempts_least_progress_request(kv_cache_manager):
    """When extending blocks for an active request fails, the scheduler preempts
    the request with least computed tokens and requeues it."""
    # page_size=4: 3 tokens = 1 block, 5 tokens = 2 blocks
    # max_blocks=2: enough for two 3-token requests (1+1=2), but extending one to 5 tokens
    # requires 2 blocks total for that request => needs 1 more => OOM (0 free)
    gen_config = GenerationConfig(max_length=32, max_new_tokens=8, eos_token_id=2, pad_token_id=0)

    req_a = InferenceRequest(request_id="req-a", generation_config=gen_config, input_ids_list=[1, 2, 3])
    req_b = InferenceRequest(request_id="req-b", generation_config=gen_config, input_ids_list=[4, 5, 6])

    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=8, max_queue_size=4, enable_chunked_prefill=False, max_num_prefill_seqs=4
    )
    scheduler = Scheduler(
        config=scheduler_config,
        kv_cache_manager=kv_cache_manager(page_size=4, max_blocks=2),
        pp_info=PPInfo(1, 0),
    )

    scheduler.submit_request(req_a)
    scheduler.submit_request(req_b)

    # Schedule 1: both requests prefill (3 tokens each, 1 block each = 2 blocks used)
    out1 = scheduler.schedule()
    assert len(out1.batch_to_infer) == 2
    assert req_a.status == RequestStatus.RUNNING
    assert req_b.status == RequestStatus.RUNNING
    assert len(req_a.block_ids) == 1
    assert len(req_b.block_ids) == 1

    # Schedule 2: both decode (1 token each).
    # req_a: num_computed=3 -> plan 1 token -> next_num_comp=4. Needs ceil(4/4)=1 block, has 1. OK.
    # req_b: num_computed=3 -> plan 1 token -> next_num_comp=4. Needs ceil(4/4)=1 block, has 1. OK.
    out2 = scheduler.schedule()
    batch_to_update = out2.batch_to_update

    # Update: req_a gets token 10 (decode), req_b gets token 20 (decode)
    infer_result = InferencerOutput(
        token_ids=torch.tensor([10, 20], dtype=torch.long, device=torch.device("cpu")),
        logits=torch.empty(0, dtype=torch.bfloat16, device=torch.device("cpu")),
        logprobs=torch.empty(0, dtype=torch.bfloat16, device=torch.device("cpu")),
    )
    scheduler.update(batch_to_update, infer_result)

    # After update: both have 1 generated token each, num_computed=4
    # Manipulate num_computed_tokens to give req_a more progress than req_b
    req_a.num_computed_tokens = 4
    req_b.num_computed_tokens = 3  # less progress, will be preemption victim

    # Schedule 3: both need decode (1 token each).
    # req_a: num_computed=4 -> plan 1 token -> next_num_comp=5. Needs ceil(5/4)=2 blocks, has 1.
    #   extend needs 1 more block => OOM (0 free blocks)
    #   Preemption: victim = req_b (least progress=3), frees 1 block
    #   Retry: allocate 1 block => succeeds
    out3 = scheduler.schedule()
    batch3 = out3.batch_to_infer

    # req_b should be preempted: generated tokens folded into input_ids_list but also kept in generated_tokens
    assert req_b.status == RequestStatus.PENDING
    assert req_b.block_ids == []
    assert req_b.input_ids_list == [4, 5, 6, 20]  # original prompt + folded token
    assert req_b.generated_tokens == [20]  # preserved for tracking
    assert req_b.num_computed_tokens == 0
    assert scheduler.queue_size() >= 1

    # req_a should still be active with 2 blocks now
    assert req_a.request_id in {r.request_id for r in batch3}
    assert req_a.status == RequestStatus.RUNNING
    assert len(req_a.block_ids) == 2


def test_scheduler_preempted_request_folds_back_generated_tokens(kv_cache_manager):
    """After preemption, generated tokens are folded into input_ids_list for re-prefill
    while also kept in generated_tokens to track the full generation history."""
    gen_config = GenerationConfig(max_length=32, max_new_tokens=4, eos_token_id=2, pad_token_id=0)

    scheduler_config = SchedulerConfig(max_num_batched_tokens=8, max_queue_size=4, enable_chunked_prefill=False)
    scheduler = Scheduler(
        config=scheduler_config,
        kv_cache_manager=kv_cache_manager(page_size=32, max_blocks=4),
        pp_info=PPInfo(1, 0),
    )

    req = InferenceRequest(request_id="req-preempt", generation_config=gen_config, input_ids_list=[10, 20, 30])

    # Set up request as if it had been running and generated 2 tokens
    block_hashes, num_prefix_hit_blocks = scheduler._kv_cache_manager.plan_prefix_cache(req.input_ids_list)
    scheduler._activate_request(req, block_hashes, num_prefix_hit_blocks)
    req.generated_tokens = [40, 50]
    req.generated_logprobs = [0.1, 0.2]
    req.num_computed_tokens = 5

    scheduler._active_requests[req.request_id] = req

    # Preempt via the real method
    scheduler._preempt_request(req)

    # Generated tokens folded into input_ids_list for re-prefill
    assert req.input_ids_list == [10, 20, 30, 40, 50]
    # But also kept in generated_tokens to track full generation history
    assert req.generated_tokens == [40, 50]
    assert req.generated_logprobs == [0.1, 0.2]  # preserved, still valid
    assert req.num_computed_tokens == 0
    assert req.block_ids == []
    assert req.status == RequestStatus.PENDING
    assert req.request_id not in scheduler._active_requests

    # After re-prefill, 2 more tokens to reach max_new_tokens=4
    req.generated_tokens.extend([60, 70])
    assert len(req.generated_tokens) == 4
    assert req.should_finish(70)  # 4 generated >= max_new_tokens=4


def test_scheduler_incremental_allocation_uses_fewer_blocks(kv_cache_manager):
    """With incremental allocation, a request with large max_length only allocates
    blocks for prompt tokens initially, not for the full max_length."""
    gen_config = GenerationConfig(max_length=1024, max_new_tokens=1000, eos_token_id=2, pad_token_id=0)

    req = InferenceRequest(request_id="req-inc", generation_config=gen_config, input_ids_list=[1, 2, 3])

    scheduler_config = SchedulerConfig(max_num_batched_tokens=8, max_queue_size=4, enable_chunked_prefill=False)
    # page_size=32: 3 tokens need only 1 block. max_length=1024 would need 32 blocks.
    # With only 4 blocks available, old code would fail; incremental allocation succeeds.
    scheduler = Scheduler(
        config=scheduler_config,
        kv_cache_manager=kv_cache_manager(page_size=32, max_blocks=4),
        pp_info=PPInfo(1, 0),
    )

    scheduler.submit_request(req)
    out = scheduler.schedule()
    assert len(out.batch_to_infer) == 1
    assert req.status == RequestStatus.RUNNING
    # Only 1 block allocated for 3 tokens with page_size=32
    assert len(req.block_ids) == 1


def test_scheduler_pp_fair_share_under_kv_pressure(kv_cache_manager):
    """PP=2, tight KV → seqs_budget derived from fair share of free blocks limits admission."""
    gen_config = GenerationConfig(max_length=32, max_new_tokens=8, eos_token_id=2, pad_token_id=0)

    def _make(idx):
        return InferenceRequest(
            request_id=f"req-{idx}",
            generation_config=gen_config,
            input_ids_list=[1, 2, 3],
        )

    # page_size=4, max_blocks=4: pre-allocate 3 blocks to simulate pressure from other slot.
    # No active seqs → avg_blocks_per_seq=max(1.0, 0/1)=1.0
    # free_blocks=1, fair_share=max(1,1//2)=1, seqs_budget=min(4, max(1, 1/1.0))=1
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=32, max_num_seqs=4, max_num_prefill_seqs=4, max_queue_size=8, enable_chunked_prefill=True
    )
    mgr = kv_cache_manager(page_size=4, max_blocks=4)
    scheduler = Scheduler(
        config=scheduler_config,
        kv_cache_manager=mgr,
        pp_info=PPInfo(2, 0),
    )

    # Simulate KV pressure: manually allocate 3 of 4 blocks
    mgr.allocate_blocks(12)  # 12 tokens / page_size=4 = 3 blocks

    for i in range(4):
        scheduler.submit_request(_make(i))

    out = scheduler.schedule()
    batch = out.batch_to_infer

    assert len(batch) == 1, f"Expected 1 new request under KV pressure, got {len(batch)}"


def test_scheduler_pp_no_limit_when_kv_plentiful(kv_cache_manager):
    """PP=2, abundant KV → fair share is large enough to admit all requests."""
    gen_config = GenerationConfig(max_length=32, max_new_tokens=8, eos_token_id=2, pad_token_id=0)

    def _make(idx):
        return InferenceRequest(
            request_id=f"req-{idx}",
            generation_config=gen_config,
            input_ids_list=[1, 2, 3],
        )

    # page_size=4, max_blocks=64: plenty of blocks. Each 3-token request needs 1 block.
    # No active seqs → avg_blocks_per_seq=1.0
    # free_blocks=64, fair_share=64//2=32, seqs_budget=min(4, max(1, 32/1.0))=4
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=32, max_num_seqs=4, max_num_prefill_seqs=4, max_queue_size=8, enable_chunked_prefill=True
    )
    mgr = kv_cache_manager(page_size=4, max_blocks=64)
    scheduler = Scheduler(
        config=scheduler_config,
        kv_cache_manager=mgr,
        pp_info=PPInfo(2, 0),
    )

    for i in range(4):
        scheduler.submit_request(_make(i))

    out = scheduler.schedule()
    batch = out.batch_to_infer

    assert len(batch) == 4, f"Expected 4 requests with plentiful KV, got {len(batch)}"


def test_preempt_resubmit_produces_same_results(kv_cache_manager):
    """Verify that a request preempted once or twice produces the same generated tokens
    and logprobs as an uninterrupted run through the full scheduler loop.

    Uses a deterministic mock token generator so we can compare final outputs.
    Tight KV cache (page_size=4, max_blocks=3) forces preemption of the least-progress
    request when blocks run out during decode extension.
    """
    gen_config = GenerationConfig(max_length=32, max_new_tokens=6, eos_token_id=-1, pad_token_id=0)

    prompt = [10, 20, 30]

    # --- Helper: deterministic "model" that picks token = position * 7 + 1 ---
    def mock_generate(requests: list[InferenceRequest]) -> InferencerOutput:
        """Return a fake token for each request in decode phase (prefill requests are skipped by update)."""
        tokens = []
        logprobs = []
        for req in requests:
            if req.num_computed_tokens >= len(req.input_ids_list):
                tok = req.num_computed_tokens * 7 + 1
            else:
                tok = 0  # prefill placeholder, won't be used
            tokens.append(tok)
            logprobs.append(float(tok) / 100.0)
        return InferencerOutput(
            token_ids=torch.tensor(tokens, dtype=torch.long),
            logits=torch.empty(0, dtype=torch.bfloat16),
            logprobs=torch.tensor(logprobs, dtype=torch.float32),
        )

    def run_to_completion(scheduler, req):
        """Drive the scheduler loop until the request finishes."""
        scheduler.submit_request(req)
        for _ in range(50):  # safety bound
            out = scheduler.schedule()
            batch_update = out.batch_to_update
            if batch_update:
                infer_result = mock_generate(batch_update)
                scheduler.update(batch_update, infer_result)
            if req.is_finished:
                break
        return req.generated_tokens[:], req.generated_logprobs[:]

    # --- Run 1: baseline with plenty of KV (no preemption) ---
    baseline_req = InferenceRequest(request_id="baseline", generation_config=gen_config, input_ids_list=list(prompt))
    sched_baseline = Scheduler(
        config=SchedulerConfig(
            max_num_batched_tokens=8, max_queue_size=4, enable_chunked_prefill=False, max_num_prefill_seqs=4
        ),
        kv_cache_manager=kv_cache_manager(page_size=4, max_blocks=64),
        pp_info=PPInfo(1, 0),
    )
    baseline_tokens, baseline_logprobs = run_to_completion(sched_baseline, baseline_req)
    assert len(baseline_tokens) == 6, f"Baseline should generate 6 tokens, got {len(baseline_tokens)}"

    # --- Run 2: tight KV forces preemption ---
    # page_size=4, max_blocks=3. Two concurrent requests with 3-token prompt each need
    # 1 block each for prefill. During decode extension to 5 tokens, one needs 2 blocks
    # but only 1 is free → preempt the other.
    preempt_req = InferenceRequest(
        request_id="preempt-target", generation_config=gen_config, input_ids_list=list(prompt)
    )
    # A competing request to create memory pressure
    competitor = InferenceRequest(
        request_id="competitor",
        generation_config=GenerationConfig(max_length=32, max_new_tokens=6, eos_token_id=-1, pad_token_id=0),
        input_ids_list=[40, 50, 60],
    )
    sched_tight = Scheduler(
        config=SchedulerConfig(
            max_num_batched_tokens=8, max_queue_size=4, enable_chunked_prefill=False, max_num_prefill_seqs=4
        ),
        kv_cache_manager=kv_cache_manager(page_size=4, max_blocks=3),
        pp_info=PPInfo(1, 0),
    )
    sched_tight.submit_request(preempt_req)
    sched_tight.submit_request(competitor)

    # Run until preempt_req finishes (competitor may or may not finish)
    preempted_count = 0
    for _ in range(100):
        out = sched_tight.schedule()
        if out.batch_to_update:
            infer_result = mock_generate(out.batch_to_update)
            sched_tight.update(out.batch_to_update, infer_result)
        # Track preemptions
        if (
            preempt_req.status == RequestStatus.PENDING
            and preempt_req.num_computed_tokens == 0
            and len(preempt_req.generated_tokens) > 0
        ):
            preempted_count += 1
        if preempt_req.is_finished:
            break

    assert preempt_req.is_finished, "preempt_req should have completed"
    assert preempted_count >= 1, "test premise broken: no preemption happened"

    # Core invariant: generated tokens and logprobs must match the baseline
    assert preempt_req.generated_tokens == baseline_tokens, (
        f"Tokens differ after preemption!\n  Baseline:  {baseline_tokens}\n  Preempted: {preempt_req.generated_tokens}"
    )
    assert preempt_req.generated_logprobs == baseline_logprobs, (
        f"Logprobs differ after preemption!\n"
        f"  Baseline:  {baseline_logprobs}\n"
        f"  Preempted: {preempt_req.generated_logprobs}"
    )


def test_repeated_preemption_no_token_duplication_in_input_ids():
    """Verify that multiple preemptions don't duplicate tokens in input_ids_list.

    Directly tests InferenceRequest.preempt() through 3 successive cycles.
    """
    gen_config = GenerationConfig(max_length=32, max_new_tokens=8, eos_token_id=2, pad_token_id=0)
    req = InferenceRequest(request_id="req-multi-preempt", generation_config=gen_config, input_ids_list=[10, 20, 30])

    # Round 1: generate 2 tokens → preempt
    req.generated_tokens = [40, 50]
    req.generated_logprobs = [0.4, 0.5]
    req.num_computed_tokens = 5
    req.preempt()
    assert req.input_ids_list == [10, 20, 30, 40, 50], "First preemption should fold [40,50]"
    assert req.generated_tokens == [40, 50]
    assert req.status == RequestStatus.PENDING
    assert req.num_computed_tokens == 0

    # Round 2: generate 1 more token → preempt again
    req.generated_tokens.append(60)
    req.generated_logprobs.append(0.6)
    req.num_computed_tokens = 6
    req.preempt()
    # Critical: should be [10,20,30,40,50,60], NOT [10,20,30,40,50,40,50,60]
    assert req.input_ids_list == [10, 20, 30, 40, 50, 60], (
        f"Second preemption duplicated tokens! Got {req.input_ids_list}"
    )
    assert req.generated_tokens == [40, 50, 60]
    assert req.generated_logprobs == [0.4, 0.5, 0.6]

    # Round 3: generate 2 more tokens → preempt a third time
    req.generated_tokens.extend([70, 80])
    req.generated_logprobs.extend([0.7, 0.8])
    req.num_computed_tokens = 8
    req.preempt()
    assert req.input_ids_list == [10, 20, 30, 40, 50, 60, 70, 80], (
        f"Third preemption duplicated tokens! Got {req.input_ids_list}"
    )
    assert req.generated_tokens == [40, 50, 60, 70, 80]
    assert len(req.generated_tokens) == 5
    assert req.num_computed_tokens == 0
