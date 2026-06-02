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
import queue
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import zip_longest

from vexact.batch_invariant_ops.kv_cache_context import KVCacheManager
from vexact.config import PPInfo, SchedulerConfig
from vexact.core.request import InferenceRequest
from vexact.core.runtime_data import InferencerOutput


logger = logging.getLogger(__name__)


@dataclass
class InFlightBatch:
    """
    Represents a batch of requests flowing through the pipeline.

    In pipeline parallelism, multiple batches are processed concurrently as they
    flow through different pipeline stages. This class tracks all requests in a
    single batch as it progresses through the pipeline.
    """

    # All active requests in this batch (persistent storage)
    active_requests: OrderedDict[str, InferenceRequest] = field(default_factory=OrderedDict)

    def get_snapshot(self) -> list[InferenceRequest]:
        """Get current batch as a list, excluding finished requests."""
        return [req for req in self.active_requests.values() if not req.is_finished]


@dataclass
class SchedulerOutput:
    # Batch to send to the inferencer for forward pass
    batch_to_infer: list[InferenceRequest]

    # Batch ready to receive outputs from the inferencer
    # Due to pipeline latency, this is different from batch_to_infer
    batch_to_update: list[InferenceRequest]


class Scheduler:
    """
    Owns inference request lifecycle, including queueing, activation, and completion.
    Also tracks per-slot KV cache ownership similar to vLLM's scheduler.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        kv_cache_manager: KVCacheManager,
        pp_info: PPInfo,
    ):
        self.config = config
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.total_requests = 0
        self._kv_cache_manager = kv_cache_manager

        self._request_queue: queue.Queue[InferenceRequest] = queue.Queue(maxsize=config.max_queue_size)

        # In-flight batches: one batch slot per PP rank in the circular buffer
        # In PP, different batches flow through pipeline stages concurrently
        self._inflight_batches: list[InFlightBatch] = [InFlightBatch() for _ in range(pp_info.pp_size)]

        # Circular index tracking which batch slot is currently being scheduled
        # Rotates through slots: 0 -> 1 -> 2 -> ... -> pp_size-1 -> 0
        self._current_batch_idx = 0

        self._result_queue: queue.Queue[InferenceRequest] = queue.Queue()
        self._enable_chunked_prefill = config.enable_chunked_prefill

    @property
    def _active_requests(self) -> OrderedDict[str, InferenceRequest]:
        """Active requests in the current in-flight batch slot."""
        return self._inflight_batches[self._current_batch_idx].active_requests

    def submit_request(self, request: InferenceRequest):
        """Queue a new inference request.

        On failure (queue full), marks request as FAILED and puts it in result queue.
        """
        try:
            self._request_queue.put(request, block=False)
            self.total_requests += 1
            return True
        except queue.Full:
            logger.warning(f"[Scheduler] Queue full, rejecting {request.request_id}")
            request.fail()
            self._result_queue.put([request])
            return False

    def schedule(self) -> SchedulerOutput:
        """
        Schedule the next batch step.

        Fills available capacity with queued requests and returns batches for:
        1. Inference: batch to send to the inferencer
        2. Update: batch ready to receive outputs (may be empty due to pipeline latency)

        Returns:
            SchedulerOutput with batch_to_infer and batch_to_update
        """
        # Plan tokens for existing active requests and extend KV cache blocks
        available_token_budget = self.max_num_batched_tokens
        for request in list(self._active_requests.values()):
            if request.request_id not in self._active_requests:
                # Request was preempted during a prior iteration
                continue
            tokens, next_num_comp = self._plan_tokens_for_request(request, available_token_budget)
            request.tokens_this_step = tokens
            request.num_computed_tokens = next_num_comp

            # Try to extend blocks for planned tokens; preempt if OOM
            if not self._extend_or_preempt(request):
                # This request was itself preempted; don't count its tokens
                continue

            available_token_budget -= tokens

        # Distribute new sequence admissions evenly across batch slots to fill pipeline bubbles.
        # Derive budget from free blocks and average per-sequence block usage so it self-adjusts dynamically.
        num_inflight_slots = len(self._inflight_batches)
        seqs_budget = self.config.max_num_seqs - len(self._active_requests)
        if num_inflight_slots > 1 and self.config.enable_pp_fair_share:
            active_seqs = self.total_inflight_request_count()
            avg_blocks_per_seq = max(1.0, self._kv_cache_manager.num_allocated_blocks() / max(1, active_seqs))
            fair_share_blocks = max(1, self._kv_cache_manager.num_free_blocks() // num_inflight_slots)
            seqs_budget = min(seqs_budget, max(1, int(fair_share_blocks / avg_blocks_per_seq)))

        # Try to fill remaining capacity with new requests from queue
        while available_token_budget > 0 and seqs_budget > 0:
            try:
                # If completely idle, block on queue to avoid busy-waiting
                if self.total_inflight_request_count() == 0:
                    request = self._request_queue.get(block=True, timeout=0.1)
                else:
                    request = self._request_queue.get_nowait()
            except queue.Empty:
                break

            try:
                tokens, next_num_comp = self._plan_tokens_for_request(request, available_token_budget)
                if available_token_budget >= tokens:
                    self._activate_request(request)

                    # Set scheduling fields and add to the active batch
                    available_token_budget -= tokens
                    request.tokens_this_step = tokens
                    request.num_computed_tokens = next_num_comp
                    # A naive strategy, if this request cannot be fit in, just quit
                    # It's possible that there's other request in the queue that can
                    # be fit in
                    seqs_budget -= 1
                else:
                    self._request_queue.put(request)
                    break
            except RuntimeError as e:
                # Allocation failed (e.g., out of KV cache); requeue and retry later
                logger.debug(f"[Scheduler] Allocation failed for {request.request_id}: {e}")
                self._request_queue.put(request)
                break
            except Exception as e:
                logger.error(f"[Scheduler] Error scheduling {request.request_id}: {e}")
                raise

        # Get snapshot of batch to send for inference
        batch_to_infer = self._inflight_batches[self._current_batch_idx].get_snapshot()

        # Advance circular buffer: rotate to next batch slot
        self._current_batch_idx = (self._current_batch_idx + 1) % len(self._inflight_batches)

        # Get batch ready to receive outputs (after rotation, due to pipeline latency)
        batch_to_update = self._inflight_batches[self._current_batch_idx].get_snapshot()

        return SchedulerOutput(batch_to_infer=batch_to_infer, batch_to_update=batch_to_update)

    def _plan_tokens_for_request(self, request: InferenceRequest, available_token_budget: int):
        num_comp = request.num_computed_tokens
        input_len = len(request.input_ids_list)
        prefill_remaining = max(0, input_len - num_comp)

        if prefill_remaining == 0:
            tokens = 1
            next_num_comp = num_comp + tokens
        elif self._enable_chunked_prefill:
            tokens = min(available_token_budget, prefill_remaining)
            # TODO: if enable split kv, need to align with split kv size
            next_num_comp = num_comp + tokens
        else:
            tokens = input_len
            next_num_comp = input_len

        return tokens, next_num_comp

    def update(self, requests: list[InferenceRequest], infer_result: InferencerOutput):
        """Process inference outputs and finalize completed requests."""
        finished_requests = []

        # Process outputs for each request
        # zip_longest is when output_logits or output_scores is off
        for request, token_tensor, logits, logprobs in zip_longest(
            requests, infer_result.token_ids.cpu(), infer_result.logits.cpu(), infer_result.logprobs.cpu()
        ):
            if request.is_finished:
                continue

            if request.num_computed_tokens < len(request.input_ids_list):
                continue

            token_id = self._process_generated_token(request, token_tensor, logits, logprobs)

            if request.should_finish(token_id):
                self._finalize_request(request)
                finished_requests.append(request)

        if finished_requests:
            # currently we only put the finished request state in to the queue
            # cuz we don't need to do streaming the partial states for now
            self._result_queue.put(finished_requests)

    def _activate_request(self, request: InferenceRequest) -> None:
        """Activate a request for processing: allocate resources incrementally.

        Only allocates blocks for the current input_ids_list length (prompt for new requests,
        prompt+generated for re-activated preempted requests), not the full max_length.
        """
        new_blocks = self._kv_cache_manager.allocate_blocks(len(request.input_ids_list))
        if new_blocks is None:
            raise RuntimeError(
                f"Not enough free blocks to activate request {request.request_id}: "
                f"need coverage for {len(request.input_ids_list)} tokens"
            )
        request.block_ids.extend(new_blocks)
        request.activate()
        self._active_requests[request.request_id] = request

    def _extend_or_preempt(self, request: InferenceRequest) -> bool:
        """Try to extend KV cache blocks for request's planned tokens. If OOM, preempt least-progress request.

        Returns:
            True if request has enough blocks to proceed, False if request was preempted itself.
        """
        # num_computed_tokens already includes tokens_this_step (set to next_num_comp)
        total_tokens = request.num_computed_tokens
        new_blocks = self._kv_cache_manager.allocate_blocks(total_tokens, len(request.block_ids))
        if new_blocks is not None:
            request.block_ids.extend(new_blocks)
            return True

        # OOM: preempt the least-progress active request
        victim = min(self._active_requests.values(), key=lambda r: r.num_computed_tokens)
        self._preempt_request(victim)
        if victim is request:
            return False

        # Retry allocation after freeing victim's blocks
        new_blocks = self._kv_cache_manager.allocate_blocks(total_tokens, len(request.block_ids))
        if new_blocks is not None:
            request.block_ids.extend(new_blocks)
            return True
        return False

    def _preempt_request(self, request: InferenceRequest) -> None:
        """Preempt a request: free blocks, reset state for re-prefill, and requeue."""
        logger.debug(
            f"[Scheduler] Preempting {request.request_id} "
            f"(computed={request.num_computed_tokens}, generated={len(request.generated_tokens)})"
        )
        self._kv_cache_manager.free_blocks(request.block_ids)
        request.preempt()
        self._active_requests.pop(request.request_id, None)
        self._request_queue.put(request)

    def _finalize_request(self, request: InferenceRequest) -> None:
        """Finalize a completed request: mark finished, release KV blocks, and remove from active set."""
        self._kv_cache_manager.free_blocks(request.block_ids)
        request.finish()
        self._active_requests.pop(request.request_id, None)
        logger.debug(f"Request {request.request_id} finished after {len(request.generated_tokens)} tokens")

    def _process_generated_token(self, request: InferenceRequest, token_tensor, logits, logprobs) -> int:
        """Process and store a newly generated token with its outputs.

        Args:
            request: The request being processed
            token_tensor: Tensor containing the generated token ID
            logits: Optional logits for the generated token
            logprobs: Optional log probabilities for the generated token

        Returns:
            The generated token ID as an integer
        """
        token_id = int(token_tensor.item())

        if request.generation_config.output_logits:
            assert logits is not None
            request.generated_logits.append(logits.clone().detach())
        if request.generation_config.output_scores:
            assert logprobs is not None
            request.generated_logprobs.append(logprobs.item())

        request.generated_tokens.append(token_id)
        return token_id

    def poll_results(self, timeout: float = None) -> list[InferenceRequest]:
        """Get next batch of finished requests.

        Args:
            timeout: Max seconds to wait. None = block forever, >0 = block up to timeout.
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return []

    def total_inflight_request_count(self) -> int:
        """Count total active requests across all in-flight batches."""
        return sum(len(batch.active_requests) for batch in self._inflight_batches)

    def queue_size(self) -> int:
        return self._request_queue.qsize()
