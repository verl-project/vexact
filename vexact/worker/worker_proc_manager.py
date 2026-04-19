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

"""WorkerProcManager: manages lifecycle of worker subprocesses."""

import logging
import multiprocessing
import os
import signal
import threading
import weakref
from contextlib import contextmanager
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

from vexact.config import VeXactConfig
from vexact.worker.worker_proxy import DriverWorkerProxy, ProxyBase, WorkerProxy


logger = logging.getLogger(__name__)


@contextmanager
def _set_cuda_visible_devices(rank: int):
    """Context manager to set CUDA_VISIBLE_DEVICES for subprocess spawning.

    If CUDA_VISIBLE_DEVICES is already set (e.g., "5,6,7,8"), picks device at rank % len(devices).
    If not set, uses torch.cuda.device_count() and picks rank % device_count.
    """
    import torch

    old_value = os.environ.get("CUDA_VISIBLE_DEVICES")

    if old_value:
        devices = old_value.split(",")
    else:
        device_count = torch.cuda.device_count()
        devices = [str(i) for i in range(device_count)] if device_count > 0 else ["0"]

    selected_device = devices[rank % len(devices)]
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_device

    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_value


class WorkerProcManager:
    """Manages lifecycle of worker subprocesses."""

    def __init__(self, config: VeXactConfig, proxy_cls: type[ProxyBase] | None = None):
        self.config = config
        self.world_size = config.parallel.world_size
        self._procs: list[BaseProcess] = []
        self._death_writers: list[Connection] = []
        self._finalizer = weakref.finalize(self, _shutdown_procs, self._procs, self._death_writers)

        ctx = multiprocessing.get_context("spawn")
        ready_readers: list[Connection] = []

        # Start all processes first (so they can all call init_process_group together)
        for rank in range(self.world_size):
            rank_proxy_cls = proxy_cls or (DriverWorkerProxy if rank == 0 else WorkerProxy)
            ready_reader, ready_writer = ctx.Pipe(duplex=False)
            death_reader, death_writer = ctx.Pipe(duplex=False)

            proc = ctx.Process(
                target=self._run_proxy_loop,
                args=(rank_proxy_cls, config, rank, ready_writer, death_reader),
                name=f"WorkerProxy-{rank}",
                daemon=True,
            )
            with _set_cuda_visible_devices(rank):
                proc.start()
            ready_writer.close()
            self._procs.append(proc)
            self._death_writers.append(death_writer)
            ready_readers.append(ready_reader)

        # Wait for all processes to be ready
        for rank, ready_reader in enumerate(ready_readers):
            status = ready_reader.recv()
            ready_reader.close()
            if status != "READY":
                self.close()
                raise RuntimeError(f"Worker rank={rank} failed to start")
            logger.info(f"Worker rank={rank} ready")

    def close(self):
        """Shutdown all worker processes."""
        self._finalizer()

    @property
    def procs(self) -> list[BaseProcess]:
        return self._procs

    @staticmethod
    def _run_proxy_loop(
        proxy_cls: type[ProxyBase],
        config: VeXactConfig,
        rank: int,
        ready_pipe: Connection,
        death_pipe: Connection,
    ):
        """Entry point for worker subprocess."""
        shutdown_event = threading.Event()
        shutdown_requested = False

        def signal_handler(_signum, _frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                shutdown_event.set()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        def monitor_parent():
            try:
                death_pipe.recv()
            except EOFError:
                logger.info(f"Parent exited, shutting down {proxy_cls.__name__} rank={rank}")
                shutdown_event.set()

        threading.Thread(target=monitor_parent, daemon=True, name="DeathMonitor").start()

        proxy = None
        try:
            proxy = proxy_cls(config, rank=rank)
            proxy.start()
            logger.info(f"{proxy_cls.__name__} rank={rank} started")
            ready_pipe.send("READY")
            ready_pipe.close()
            shutdown_event.wait()
        except Exception:
            logger.exception(f"{proxy_cls.__name__} rank={rank} failed")
            ready_pipe.send("FAILED")
            ready_pipe.close()
        finally:
            death_pipe.close()
            if proxy is not None:
                proxy.stop()


def _shutdown_procs(procs: list[BaseProcess], death_writers: list[Connection]):
    """Cleanup function for weak reference finalizer."""
    for writer in death_writers:
        writer.close()

    for proc in procs:
        proc.join(timeout=1)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.5)
        if proc.is_alive():
            proc.kill()
