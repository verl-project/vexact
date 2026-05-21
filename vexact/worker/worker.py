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

import abc
import argparse
import logging
import threading
from typing import Iterable

import torch

from vexact.config import ModelConfig, ParallelConfig, PPInfo, ProfilerConfig, VeXactConfig
from vexact.distributed.pp_messager import PPMessager
from vexact.inferencer.inferencer import Inferencer
from vexact.inferencer.model_loader import ModelCreator, load_weights_from_weight_iterator
from vexact.utils.profiler import ProfilerManager
from vexact.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


logger = logging.getLogger(__name__)


class WorkerBase(abc.ABC):
    """Base class with threading and lifecycle management for workers."""

    def __init__(self):
        self._shutdown_event = threading.Event()
        self._processing_thread = None

    def start(self):
        """Start generation loop in background thread (non-blocking)."""
        if self._processing_thread and self._processing_thread.is_alive():
            logger.warning(f"{self.__class__.__name__} is already running")
            return

        self._shutdown_event.clear()
        self._processing_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self._processing_thread.start()
        logger.info(f"{self.__class__.__name__} started")

    def stop(self):
        """Signal shutdown and wait for thread to finish."""
        if not self._processing_thread or not self._processing_thread.is_alive():
            return

        self._shutdown_event.set()
        self._processing_thread.join(timeout=0.5)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info(f"{self.__class__.__name__} stopped")

    def join(self):
        """Block until worker stops."""
        if self._processing_thread:
            self._processing_thread.join()

    @abc.abstractmethod
    def _generation_loop(self):
        """Main generation loop to be implemented by subclasses."""


class Worker(WorkerBase):
    """Worker for non-driver ranks in pipeline parallelism."""

    def __init__(self, config: VeXactConfig, rank: int):
        super().__init__()

        self.config = config
        self.rank = rank
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.pp_info = PPInfo(config.parallel.pipeline_parallel_size, rank)
        profiler_config = config.profiler
        if rank != 0 and not profiler_config.profile_all_ranks:
            # set backend to None to disable profiling when not profiling all ranks
            profiler_config = ProfilerConfig(
                backend=None,
                delay_iterations=profiler_config.delay_iterations,
                max_iterations=profiler_config.max_iterations,
                output_path=profiler_config.output_path,
                profile_all_ranks=profiler_config.profile_all_ranks,
            )
        self.profiler = ProfilerManager(
            profiler_config=profiler_config,
            rank=rank,
            driver_id=config.driver.driver_id,
        )

        TorchMemorySaverAdapter.create(config.model.enable_memory_saver)

        # Load model
        self.model = ModelCreator(
            self.config.model.hf_config, config.model.model_path, self.device, self.pp_info
        ).create_model()

        # Create PP messager if needed
        pp_messager = None
        if self.pp_info.pp_size > 1:
            pp_messager = PPMessager(pp_info=self.pp_info, parallel_config=config.parallel, device=self.device)

        # VeOmni's fused MoE kernels (group_gemm/quack/npu) consult
        # ``get_parallel_state().ep_enabled`` on each forward pass to decide
        # between the EP and non-EP code paths. The lazy default
        # ``ParallelState()`` asserts ``pp*dp*cp*ulysses*tp == world_size`` and
        # therefore raises whenever torch.distributed reports world_size > 1
        # (e.g. our PP>1 rollout). Bind a non-EP parallel state up front so
        # subsequent MoE forwards take the cheap non-EP path; we never run
        # expert parallelism inside the rollout worker. The helper is
        # idempotent (warns + early-return when state already exists).
        if torch.distributed.is_initialized():
            from veomni.distributed.parallel_state import init_parallel_state

            init_parallel_state(dp_size=torch.distributed.get_world_size())

        self.inferencer = Inferencer(
            model=self.model,
            config=config,
            pp_info=self.pp_info,
            pp_messager=pp_messager,
            device=self.device,
            enable_batch_invariant=config.model.enable_batch_invariant,
        )

        # Weight transfer address
        self.zmq_handle = f"ipc:///tmp/vexact-weight-{config.driver.driver_id}-{rank}.sock"
        logger.info(f"Receiver:{self.zmq_handle=}:{torch.cuda.get_device_properties(self.device.index).uuid}")

    def _generation_loop(self):
        """Simple generation loop for non-driver workers."""
        logger.info(f"Worker (rank={self.pp_info.pp_rank}) generation loop started")
        self.profiler.start()
        try:
            while not self._shutdown_event.is_set():
                with self.profiler.annotate_context_manager("inferencer.infer"):
                    self.inferencer.infer([])
                self.profiler.step()
        finally:
            self.profiler.stop()
            logger.info(f"Worker (rank={self.pp_info.pp_rank}) generation loop ended")

    def sleep(self, tag: str = None):
        """Offload GPU memory to CPU."""
        TorchMemorySaverAdapter.get_instance().pause(tag=tag)

    def wake_up(self, tag: str = None):
        """Restore GPU memory from CPU."""
        TorchMemorySaverAdapter.get_instance().resume(tag=tag)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        """Update model weights via direct parameter copy."""
        if not hasattr(self, "_named_parameters"):  # Cache Index
            self._named_parameters = dict(self.model.named_parameters())
        # convert state_dict into iterable
        weight_iterator: Iterable[tuple[str, torch.Tensor]] = state_dict.items()
        load_weights_from_weight_iterator(self.model, self.config.model.hf_config, weight_iterator)

    def receive_weights(self):
        """Receive model weights via IPC transfer from another proc on the SAME device."""
        from vexact.integrations.verl.bucketed_weight_transfer import BucketedWeightReceiver

        receiver = BucketedWeightReceiver(zmq_handle=self.zmq_handle, device=self.device)
        receiver.receive_weights(on_bucket_received=lambda weights: self.load_state_dict(dict(weights)))


def run_worker(config: VeXactConfig, rank: int):
    """Run a worker (blocking)."""
    worker = Worker(config, rank)
    worker.start()
    worker.join()


def parse_worker_args():
    parser = argparse.ArgumentParser(description="Worker process for pipeline parallel inference")
    parser.add_argument("--pp-rank", type=int, required=True, help="Pipeline parallel rank for this worker")
    parser.add_argument("--pp-size", type=int, required=True, help="Total pipeline parallel size")
    parser.add_argument("--model-path", type=str, required=True, help="Model path or HuggingFace model name")
    parser.add_argument("--attn-impl", default="fa-invariant", help="Attention implementation to use")
    parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1024,
        help="Maximum number of KV cache blocks available for continuous batching",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable cudagraph capture/replay and force eager model forward",
    )
    parser.add_argument(
        "--use-fp32-logits",
        action="store_true",
        help="Use fused linear cross entropy for logprob computation",
    )
    parser.add_argument(
        "--profile-backend",
        type=str,
        default=None,
        choices=["torch", "proton"],
        help="Profiler backend to use (default: disabled)",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default=None,
        help="Output file for profiler trace (default: auto-generated)",
    )
    parser.add_argument(
        "--profile-delay-iterations",
        type=int,
        default=0,
        help="Number of steps to wait before starting profiler (default: 0)",
    )
    parser.add_argument(
        "--profile-max-iterations",
        type=int,
        default=0,
        help="Number of steps to profile (0 = until manually stopped, default: 0)",
    )
    parser.add_argument(
        "--profile-all-ranks",
        action="store_true",
        help="Enable profiling on all pipeline-parallel ranks (default: rank 0 only)",
    )
    return parser.parse_args()


def main():
    args = parse_worker_args()

    config = VeXactConfig(
        model=ModelConfig(
            model_path=args.model_path,
            attn_impl=args.attn_impl,
            enforce_eager=args.enforce_eager,
            use_fp32_logits=args.use_fp32_logits,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=args.pp_size,
        ),
        profiler=ProfilerConfig(
            backend=args.profile_backend,
            delay_iterations=args.profile_delay_iterations,
            max_iterations=args.profile_max_iterations,
            output_path=args.profile_output,
            profile_all_ranks=args.profile_all_ranks,
        ),
    )

    run_worker(config, args.pp_rank)


if __name__ == "__main__":
    main()
