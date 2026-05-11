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
import time

from vexact.batch_invariant_ops.kv_cache_context import KVCacheManager
from vexact.config import VeXactConfig
from vexact.core.request import InferenceRequest
from vexact.core.scheduler import Scheduler
from vexact.worker.worker import Worker


logger = logging.getLogger(__name__)


class DriverWorker(Worker):
    """Driver worker for rank 0 with scheduler and request management."""

    def __init__(self, config: VeXactConfig):
        super().__init__(config, rank=0)  # CUDA graph capture happens here
        # Prefix cache only on pp_size==1 — coordinating per-block hash state
        # across PP ranks (and ensuring KV slots agree on every rank) is out of
        # scope for now. With it disabled, KVCacheManager behaves like the
        # original allocator (always-fresh blocks, no lookups).
        # VEXACT_DISABLE_PREFIX_CACHE=1 is a debug escape hatch — useful for
        # measuring the cache's contribution by running the same workload twice.
        enable_prefix_cache = (
            config.parallel.pipeline_parallel_size == 1
            and os.environ.get("VEXACT_DISABLE_PREFIX_CACHE", "") != "1"
        )
        self.kv_cache_manager = KVCacheManager(config.cache, enable_prefix_cache=enable_prefix_cache)
        self.scheduler = Scheduler(
            config=config.scheduler,
            kv_cache_manager=self.kv_cache_manager,
            pp_info=self.pp_info,
        )

    def submit_request(self, request: InferenceRequest) -> bool:
        """Submit a request for processing."""
        return self.scheduler.submit_request(request)

    def poll_results(self, timeout: float = None) -> list[InferenceRequest]:
        """Get next batch of finished requests (blocking with optional timeout)."""
        return self.scheduler.poll_results(timeout=timeout)

    def receive_weights(self):
        """Receive new model weights.

        Active requests are preempted (their KV is stale under the new weights
        and decode would read garbage) and the prefix cache index is dropped.
        Re-prefill happens automatically on the next schedule().
        """
        self.scheduler.reset_for_state_change()
        super().receive_weights()

    def sleep(self, tag: str = None):
        """Pause memory-saving regions.

        The KV cache region is allocated with enable_cpu_backup=False, so its
        bytes are dropped on sleep — any active block IDs become invalid. Active
        requests are preempted; the prefix cache index is dropped.
        """
        self.scheduler.reset_for_state_change()
        super().sleep(tag=tag)

    def load_state_dict(self, state_dict):
        """In-process weight load (e.g. tests). Same staleness as receive_weights()."""
        self.scheduler.reset_for_state_change()
        super().load_state_dict(state_dict)

    def get_prefix_cache_stats(self) -> dict:
        """Snapshot prefix-cache counters since process start (or last state reset).

        hit_tokens / miss_tokens are summed across every activated request; the
        ratio is hit_tokens / (hit_tokens + miss_tokens), guarded against div0.
        Counters are reset on weight update / sleep so the ratio always reflects
        the current model.
        """
        hit = self.scheduler.cache_hit_tokens_total
        miss = self.scheduler.cache_miss_tokens_total
        total = hit + miss
        return {
            "prefix_cache_enabled": self.kv_cache_manager.prefix_cache_enabled,
            "hit_tokens": hit,
            "miss_tokens": miss,
            "hit_ratio": (hit / total) if total > 0 else 0.0,
            "cached_blocks": self.kv_cache_manager.num_cached_blocks(),
            "free_blocks": self.kv_cache_manager.num_free_blocks(),
        }

    def _generation_loop(self):
        """Main continuous generation loop."""
        # Start profiler here (not in __init__) so that _enable_profiler() and
        # _disable_profiler() run on the same thread. PyTorch's profiler state is
        # thread-local; starting on the main thread and stopping on the worker thread
        # causes "Can't disable Kineto profiler when it's not running".
        self.profiler.start()
        logger.info("Starting driver generation loop")
        while not self._shutdown_event.is_set():
            try:
                # Schedule next batch step
                scheduler_output = self.scheduler.schedule()
                batch_to_infer = scheduler_output.batch_to_infer
                batch_to_update = scheduler_output.batch_to_update

                # If no active requests, skip this round. scheduler handles blocking.
                if not batch_to_infer and not batch_to_update:
                    time.sleep(0.001)
                    continue

                # Generate next token for all active requests
                recv_out = len(batch_to_update) > 0
                with self.profiler.annotate_context_manager(f"inferencer.infer_Nreq{len(batch_to_infer)}"):
                    infer_result = self.inferencer.infer(batch_to_infer, recv_out)

                # post_process
                if infer_result is not None:
                    # IMPORTANT: Defer request removal until after generation step to maintain
                    # deterministic batch composition throughout the generation cycle
                    self.scheduler.update(batch_to_update, infer_result)

                # Update profiler: increment step counter and notify runtime profiler
                self.profiler.step()

            except Exception as e:
                logger.error(f"Error in generation loop: {e}")
                import traceback

                logger.error(traceback.format_exc())
                time.sleep(0.1)
                # breakpoint()
                break

        self.profiler.stop()
        logger.info("Driver generation loop ended")
