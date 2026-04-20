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
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional


logger = logging.getLogger(__name__)


@dataclass
class PPInfo:
    pp_size: int
    pp_rank: int
    is_first_rank: bool = field(init=False)
    is_last_rank: bool = field(init=False)

    def __post_init__(self):
        self.is_first_rank = self.pp_rank == 0
        self.is_last_rank = self.pp_rank == self.pp_size - 1


@dataclass(frozen=True)
class ProfilerConfig:
    """Configuration for profiling."""

    backend: Optional[Literal["torch", "proton"]] = field(
        default=None,
        metadata={"help": "Profiler backend to use (torch, proton, or None to disable profiling)."},
    )
    delay_iterations: int = field(
        default=0,
        metadata={"help": "Number of iterations to wait before starting profiler."},
    )
    max_iterations: int = field(
        default=0,
        metadata={"help": "Maximum number of iterations to profile (0 = until manually stopped)."},
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Output path for profiler trace (auto-generated if not specified)."},
    )
    profile_all_ranks: bool = field(
        default=False,
        metadata={"help": "Enable profiling on all pipeline-parallel ranks (default: rank 0 only)."},
    )

    def __post_init__(self):
        """Validate profiler configuration."""
        if self.delay_iterations < 0:
            raise ValueError(f"delay_iterations must be non-negative, got {self.delay_iterations}")
        if self.max_iterations < 0:
            raise ValueError(f"max_iterations must be non-negative, got {self.max_iterations}")


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for KV cache management."""

    page_size: int = field(
        default=256,
        metadata={"help": "Page size for KV cache organization (number of tokens per page)."},
    )
    max_cache_blocks: int = field(
        default=1024,
        metadata={"help": "Maximum number of KV cache blocks available for continuous batching."},
    )

    def __post_init__(self):
        """Validate cache configuration."""
        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}")
        if self.max_cache_blocks <= 0:
            raise ValueError(f"max_cache_blocks must be positive, got {self.max_cache_blocks}")
        logger.info(f"[CacheConfig] max_cache_blocks={self.max_cache_blocks}, page_size={self.page_size}")


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""

    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model or HuggingFace model name."},
    )
    attn_impl: str = field(
        default="fa-invariant",
        metadata={"help": "Attention implementation to use (fa-invariant, fa-invariant-cute, flex)."},
    )
    enable_batch_invariant: bool = field(
        default=True,
        metadata={"help": "Enable batch invariant operations for deterministic inference."},
    )
    enable_memory_saver: bool = field(
        default=False,
        metadata={"help": "Enable torch-memory-saver for GPU memory optimization via CPU offloading."},
    )
    use_fp32_logits: bool = field(
        default=False,
        metadata={
            "help": "Whether to use fp32 logits to compute log_probs. "
            "Useful when the training side using fused linear cross entropy for logprob computation.",
        },
    )
    hf_config: Optional[object] = field(
        default=None,
        metadata={"help": "HuggingFace model configuration object. Auto-loaded from model_path."},
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum model context length. Auto-detected if not provided."},
    )
    enforce_eager: bool = field(
        default=False,
        metadata={"help": "Disable cudagraph to enforce eager model forward. "},
    )

    def __post_init__(self):
        """Validate model configuration and initialize derived fields."""
        # Validate model_path
        if self.model_path is None:
            raise ValueError("model_path is required!\nUsage: python script.py model.model_path=/path/to/model")

        # Validate attention implementation
        valid_attn_impls = ["fa-invariant", "fa-invariant-cute", "flex"]
        if self.attn_impl not in valid_attn_impls:
            raise ValueError(f"attn_impl must be one of {valid_attn_impls}, got '{self.attn_impl}'")

        # Load HuggingFace config from model_path
        if self.hf_config is None:
            self._load_hf_config()

        # Initialize max_model_len
        self._initialize_max_model_len()

    def _load_hf_config(self):
        """Load HuggingFace configuration from model_path."""
        from transformers import AutoConfig

        model_kwargs = {
            "trust_remote_code": False,
            "_attn_implementation": self.attn_impl,
        }

        object.__setattr__(self, "hf_config", AutoConfig.from_pretrained(self.model_path, **model_kwargs))

    def _initialize_max_model_len(self):
        """Initialize or validate max_model_len from HuggingFace config."""
        # Try to get max_model_len from config
        possible_keys = [
            "max_position_embeddings",
            "max_sequence_length",
            "n_positions",
            "seq_length",
        ]

        config_max_len = None
        for key in possible_keys:
            if hasattr(self.hf_config, key):
                max_len = getattr(self.hf_config, key)
                if max_len is not None and max_len > 0:
                    config_max_len = int(max_len)
                    break

        # Fallback if not found in config
        if config_max_len is None:
            config_max_len = 2048  # default value
            logger.warning(
                f"Could not auto-detect max_model_len from config. "
                f"Using default value of {config_max_len}. Available config keys: "
                f"{list(vars(self.hf_config).keys())}",
                stacklevel=2,
            )

        # Handle user-provided max_model_len
        if self.max_model_len is None:
            # Use value from hf config
            object.__setattr__(self, "max_model_len", config_max_len)
            logger.info("Using max_model_len %d from model config.", config_max_len)
        elif self.max_model_len != config_max_len:
            # User provided a different value - warn but respect it
            logger.warning(
                f"Provided max_model_len ({self.max_model_len}) differs from "
                f"config value ({config_max_len}). Using provided value.",
                stacklevel=2,
            )


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for parallel execution."""

    pipeline_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of pipeline parallel stages for multi-GPU inference."},
    )
    torch_distributed_addr: str = field(
        default="0.0.0.0",
        metadata={"help": "Address for torch.distributed."},
    )
    torch_distributed_port: int = field(
        default=None,
        metadata={"help": "Port for torch.distributed."},
    )
    world_size: int = field(init=False)

    def __post_init__(self):
        """Validate parallel configuration."""
        if self.pipeline_parallel_size <= 0:
            raise ValueError(f"pipeline_parallel_size must be positive, got {self.pipeline_parallel_size}")
        object.__setattr__(self, "world_size", self.pipeline_parallel_size)
        if self.torch_distributed_port is None:
            from vexact.utils.sys import find_available_port

            object.__setattr__(self, "torch_distributed_port", str(find_available_port()))


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for request scheduling and batching."""

    max_num_batched_tokens: int = field(
        default=2048,
        metadata={"help": "Maximum number of query tokens per batch."},
    )
    max_num_seqs: int = field(default=2048, metadata={"help": "Maximum number of sequences per batch."})
    max_num_prefill_seqs: int = field(
        default=1,
        metadata={"help": "Maximum number of sequences for prefill."},
    )
    max_queue_size: int = field(
        default=0,
        metadata={"help": "Maximum number of requests in the waiting queue. 0 means infinite."},
    )
    enable_chunked_prefill: bool = field(
        default=True,
        metadata={"help": "Enable chunked prefill for long prompts (chunk size: 64 tokens)."},
    )
    enable_pp_fair_share: bool = field(
        default=False,
        metadata={"help": "Distribute new sequence admissions evenly across PP batch slots to fill pipeline bubbles."},
    )

    def __post_init__(self):
        """Validate scheduler configuration."""
        if self.max_num_batched_tokens <= 0:
            raise ValueError(f"max_num_batched_tokens must be positive, got {self.max_num_batched_tokens}")


@dataclass
class DriverConfig:
    """Configuration for driver client."""

    is_worker_proc_managed: bool = field(
        default=True,
        metadata={"help": "If True, VeXact manages worker subprocesses. If False, workers are launched externally."},
    )
    driver_id: Optional[str] = field(
        default=None,
        metadata={"help": "Unique identifier for this driver instance. Auto-generated if not provided."},
    )
    request_address: Optional[str] = field(
        default=None,
        metadata={"help": "Request channel address. Auto-generated from driver_id if not provided."},
    )
    control_addresses: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Control channel addresses. Auto-generated from driver_id if not provided."},
    )

    def __post_init__(self):
        """Validate driver configuration."""
        # Generate driver_id if not provided
        if self.driver_id is None:
            self.driver_id = uuid.uuid4().hex[:8]

    def generate_addresses(self, world_size: int) -> None:
        """Generate IPC addresses from driver_id if not already set.

        Args:
            world_size: Number of worker processes (used to generate control addresses).
        """
        if self.request_address is None:
            self.request_address = f"ipc:///tmp/vexact_req_{self.driver_id}.sock"
        if self.control_addresses is None:
            self.control_addresses = [f"ipc:///tmp/vexact_ctrl_{self.driver_id}_{i}.sock" for i in range(world_size)]


@dataclass(frozen=True)
class VeXactConfig:
    """Top-level configuration for VeXact engine."""

    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    driver: DriverConfig = field(default_factory=DriverConfig)

    def __post_init__(self):
        # Generate driver IPC addresses using world_size from parallel config
        # TODO: generate addresses with local_world_size with headless slave driver
        self.driver.generate_addresses(self.parallel.world_size)
