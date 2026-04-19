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
Profiler Manager.

This module provides the ProfilerManager class that handles all profiling concerns
for the continuous batching engine, supporting both PyTorch and Triton Proton profilers.
"""

import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Optional

import torch
import triton.profiler as proton
from torch.profiler import ProfilerActivity, profile

from vexact.config import ProfilerConfig


logger = logging.getLogger(__name__)
logger.setLevel("INFO")


class ProfilerBackend(ABC):
    """Abstract base class for profiler backends."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    @abstractmethod
    def start(self) -> None:
        """Start a profiling session."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop profiling and export trace."""
        pass

    def annotate_context_manager(self, name: str):
        """Return a context manager to annotate profiler traces."""
        return nullcontext()


class TorchProfilerBackend(ProfilerBackend):
    """PyTorch profiler backend."""

    def __init__(self, output_path: str):
        super().__init__(output_path)
        self._profiler = None
        # Initialize Kineto before any CUDA graphs are created. Without this, profiling
        # after CUDA graph capture raises "Can't disable Kineto profiler when it's not
        # running". See https://github.com/pytorch/pytorch/issues/75504
        from torch.profiler._utils import _init_for_cuda_graphs

        _init_for_cuda_graphs()

    def start(self) -> None:
        """Start PyTorch profiler session."""
        self._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=False,
        )
        self._profiler.__enter__()

    def stop(self) -> None:
        """Stop PyTorch profiler and export trace."""
        if self._profiler is None:
            return

        try:
            # __exit__ finalizes the trace and populates kineto_results,
            # which export_chrome_trace requires.
            self._profiler.__exit__(None, None, None)
            self._profiler.export_chrome_trace(self.output_path)
            logger.info(f"PyTorch trace saved to {self.output_path}")
            self._upload_trace()
        except Exception as e:
            logger.warning(f"Failed during profiler stop/export: {e}")
        finally:
            self._profiler = None

    def annotate_context_manager(self, name: str):
        """Return a context manager to annotate profiler traces."""
        return torch.profiler.record_function(name)

    def _upload_trace(self) -> None:
        """Upload trace using mlx asset upload if available."""
        fallback_mlx = "/opt/tiger/mlx_deploy/bin/mlx"
        try:
            subprocess.run(["mlx", "asset", "upload", self.output_path], check=True)
            logger.info(f"Uploaded trace via mlx: {self.output_path}")
        except FileNotFoundError:
            try:
                subprocess.run([fallback_mlx, "asset", "upload", self.output_path], check=True)
                logger.info(f"Uploaded trace via mlx fallback: {self.output_path}")
            except FileNotFoundError:
                logger.info("mlx command not found (including fallback path); skipping upload")
            except subprocess.CalledProcessError as e:
                logger.info(f"Failed to upload trace via mlx fallback: {e}")
        except subprocess.CalledProcessError as e:
            logger.info(f"Failed to upload trace via mlx: {e}")


class ProtonProfilerBackend(ProfilerBackend):
    """Triton Proton profiler backend."""

    def __init__(self, output_path: str):
        super().__init__(output_path)
        # Proton automatically appends `.hatchet`; strip it to avoid double suffix.
        self._proton_name = output_path[: -len(".hatchet")] if output_path.endswith(".hatchet") else output_path
        self._profiler_active = False

    def start(self) -> None:
        """Start Proton profiler session."""
        proton.start(name=self._proton_name, context="shadow", hook="triton")
        self._profiler_active = True

    def stop(self) -> None:
        """Stop Proton profiler and finalize trace."""
        if not self._profiler_active:
            return

        try:
            proton.finalize()
            logger.info(f"Proton trace saved to {self.output_path}")
        finally:
            self._profiler_active = False

    def annotate_context_manager(self, name: str):
        """Return a context manager to annotate profiler traces."""
        if self._profiler_active:
            return proton.cpu_timed_scope(name)
        return nullcontext()


class ProfilerManager:
    """
    Manages profiling for the continuous batching engine.

    Supports two profiling backends:
    - torch: PyTorch profiler with chrome trace output
    - proton: Triton Proton profiler with optimized Triton kernel profiling

    Uses delay_iterations and max_iterations to control profiling window.
    """

    def __init__(
        self,
        profiler_config: ProfilerConfig,
        rank: int,
        driver_id: str,
    ):
        """
        Initialize the profiler manager.

        Args:
            profiler_config: Profiler configuration (backend=None disables profiling)
            rank: Pipeline/process rank for trace naming
            driver_id: Driver/server identifier for trace naming
        """
        self._rank = rank
        self._driver_id = driver_id
        self._enabled = profiler_config.backend is not None

        # Track when the profiler gets triggered by start()
        self._active_iteration_count = 0
        self._active = False

        # Track when the profiler is actually running
        self._profiling_for_iters = 0
        self._running = False

        # Early return if disabled
        if not self._enabled:
            self.backend = None
            return

        self._delay_iters = profiler_config.delay_iterations
        self._max_iters = profiler_config.max_iterations

        if self._delay_iters > 0:
            logger.info(f"GPU profiling will start {self._delay_iters} steps after start profile.")

        if self._max_iters > 0:
            logger.info(f"GPU profiling will stop after {self._max_iters} steps, or when stop profile is called.")

        output_path = self._prepare_output_path(profiler_config.output_path, profiler_config.backend)

        self.backend = self._create_backend(profiler_config.backend, output_path)

    def _create_backend(self, backend_name: str, output_path: str) -> ProfilerBackend:
        """Create the appropriate profiler backend."""
        if backend_name == "torch":
            return TorchProfilerBackend(output_path)
        elif backend_name == "proton":
            return ProtonProfilerBackend(output_path)
        else:
            raise ValueError(f"Unknown profiler backend: {backend_name}")

    def _prepare_output_path(self, output_path: Optional[str], backend_name: str) -> str:
        """Add a per-process suffix and backend-specific extension to trace filenames."""
        if not output_path:
            output_path = "vexact_profile"
        os.makedirs(output_path, exist_ok=True)
        timestamp = int(time.time())
        pid = os.getpid()
        backend_suffix = ".trace.json.gz" if backend_name == "torch" else ".hatchet"
        filename = f"d{self._driver_id}_r{self._rank}_pid{pid}_{timestamp}{backend_suffix}"
        return os.path.join(output_path, filename)

    def _call_start(self) -> None:
        """Call backend.start() with error handling."""
        try:
            self.backend.start()
            self._running = True
            logger.info(f"Profiler started; writing to {self.backend.output_path}")
        except Exception as e:
            logger.warning(f"Failed to start profiler: {e}")

    def _call_stop(self) -> None:
        """Call backend.stop() with error handling."""
        try:
            self.backend.stop()
            logger.info("Profiler stopped successfully.")
        except Exception as e:
            logger.warning(f"Failed to stop profiler: {e}")
        self._running = False

    def start(self) -> None:
        """Activate profiling, accounting for delayed starts."""
        if not self._enabled:
            return

        if self._active:
            logger.debug("start profile called when profiler is already active. Ignoring.")
            return

        self._active = True
        if self._delay_iters == 0:
            self._call_start()

    def step(self) -> None:
        """
        Update profiler state at each generation step.

        Handles delayed starts and max iteration limits.
        Should be called once per generation step.
        """
        if not self._active:
            return

        self._active_iteration_count += 1

        # Handle delayed start
        if not self._running and self._delay_iters > 0 and self._active_iteration_count == self._delay_iters:
            logger.info("Starting profiler after delay...")
            self._call_start()

        # Track profiling iterations
        if self._running:
            self._profiling_for_iters += 1

        # Auto-stop after max iterations
        if self._max_iters > 0 and self._running and self._profiling_for_iters >= self._max_iters:
            logger.info("Max profiling iterations reached. Stopping profiler...")
            self._call_stop()

    def stop(self) -> None:
        """Deactivate profiling and stop if running."""
        if not self._enabled:
            return

        if not self._active:
            logger.debug("stop profile called when profiler is not active. Ignoring.")
            return

        self._active = False
        self._active_iteration_count = 0
        self._profiling_for_iters = 0

        if self._running:
            self._call_stop()

    def annotate_context_manager(self, name: str):
        """
        Return a context manager to annotate profiler traces.

        Only annotates when profiler is running.

        Args:
            name: Name for the profiler annotation

        Returns:
            Context manager for profiler annotation
        """
        if self._enabled and self._running:
            return self.backend.annotate_context_manager(name)
        return nullcontext()

    def is_active(self) -> bool:
        """Check if profiler is active (started but maybe not running yet)."""
        return self._active

    def is_running(self) -> bool:
        """Check if profiler is currently running (collecting traces)."""
        return self._running
