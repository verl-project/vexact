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
from contextlib import contextmanager

import torch


try:
    import torch_memory_saver

    _memory_saver = torch_memory_saver.torch_memory_saver
    import_error = None
except ImportError as e:
    import_error = e
    pass


logger = logging.getLogger(__name__)


def _log_gpu_memory(message, gpu_id=0):
    """Print GPU memory usage with optional message"""
    mem = torch.cuda.device_memory_used(gpu_id)
    logger.info(f"GPU {gpu_id} memory: {mem / 1024**3:.2f} GB ({message})")


class TorchMemorySaverAdapter:
    _instance = None

    @staticmethod
    def create(enable: bool):
        if TorchMemorySaverAdapter._instance is not None:
            assert (
                isinstance(TorchMemorySaverAdapter._instance, _TorchMemorySaverAdapterReal)
                if enable
                else isinstance(TorchMemorySaverAdapter._instance, _TorchMemorySaverAdapterNoop)
            )
            return TorchMemorySaverAdapter._instance

        if enable and import_error is not None:
            logger.warning(
                "enable_memory_saver is enabled, but "
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )
            raise import_error

        TorchMemorySaverAdapter._instance = _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        return TorchMemorySaverAdapter._instance

    @staticmethod
    def get_instance():
        if TorchMemorySaverAdapter._instance is None:
            raise RuntimeError("TorchMemorySaverAdapter not created yet. Call create() first.")
        return TorchMemorySaverAdapter._instance

    def check_validity(self, caller_name):
        if not self.enabled:
            logger.warning(
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )

    def configure_subprocess(self):
        raise NotImplementedError

    def region(self, tag: str, enable_cpu_backup: bool = False):
        raise NotImplementedError

    def cuda_graph(self, **kwargs):
        raise NotImplementedError

    def disable(self):
        raise NotImplementedError

    def pause(self, tag: str):
        raise NotImplementedError

    def resume(self, tag: str):
        raise NotImplementedError

    @property
    def enabled(self):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    """Adapter for TorchMemorySaver with tag-based control"""

    def __init__(self):
        _memory_saver.hook_mode = "torch"
        super().__init__()

    def configure_subprocess(self):
        return torch_memory_saver.configure_subprocess()

    def region(self, tag: str, enable_cpu_backup: bool = False):
        r = _memory_saver.region(tag=tag, enable_cpu_backup=enable_cpu_backup)
        _log_gpu_memory(f"{tag}: allocated")
        return r

    def cuda_graph(self, **kwargs):
        return _memory_saver.cuda_graph(**kwargs)

    def disable(self):
        return _memory_saver.disable()

    def pause(self, tag: str):
        r = _memory_saver.pause(tag=tag)
        _log_gpu_memory(f"{tag}: relased")
        return r

    def resume(self, tag: str):
        r = _memory_saver.resume(tag=tag)
        _log_gpu_memory(f"{tag}: resumed")
        return r

    @property
    def enabled(self):
        return _memory_saver is not None and _memory_saver.enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        yield

    @contextmanager
    def cuda_graph(self, **kwargs):
        yield

    @contextmanager
    def disable(self):
        yield

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
