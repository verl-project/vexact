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

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import msgspec
import torch
from transformers import GenerationConfig


class RequestStatus(enum.IntEnum):
    """Status of inference requests."""

    PENDING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()
    FAILED = enum.auto()


# TODO: Rename to internal RequestState
@dataclass
class InferenceRequest:
    """Represents a single inference request.

    Either prompt, or input_ids_list must be provided:
    - prompt: Text prompt to be tokenized
    - input_ids_list: Pre-tokenized input as List[int] (converted to tensor in _prepare_request)
    """

    # Essential Inputs
    request_id: str
    generation_config: GenerationConfig
    input_ids_list: Optional[list[int]] = None  # Alternative to input_ids, avoids device-level tensor manipulation

    # States: Scheduling
    status: RequestStatus = RequestStatus.PENDING
    tokens_generated: int = 0
    num_computed_tokens: Optional[int] = 0  # all computed tokens, includes prefilling and decoding
    tokens_this_step: Optional[int] = 0

    block_ids: list[int] = field(default_factory=list)

    # State: Outputs
    generated_tokens: list[int] = field(default_factory=list)
    generated_logits: list[torch.Tensor] = field(default_factory=list)  # Store logits for each token
    generated_logprobs: list[float] = field(default_factory=list)

    # Tracks original prompt length for correct token folding on repeated preemptions
    _original_prompt_len: int = 0

    # Timer  # TODO: clean up
    start_generation_time: float = 0.0
    processing_time: float = 0.0

    def __post_init__(self):
        if self.input_ids_list is None:
            raise ValueError("'input_ids_list' must be provided for InferenceRequest")
        self._original_prompt_len = len(self.input_ids_list)

    @classmethod
    def from_driver_request(cls, driver_request: "DriverRequest") -> "InferenceRequest":
        """Create an InferenceRequest from a DriverRequest.

        Args:
            driver_request: The DriverRequest to convert

        Returns:
            A new InferenceRequest with fields initialized from the DriverRequest
        """
        return cls(
            request_id=driver_request.request_id,
            generation_config=driver_request.generation_config,
            input_ids_list=driver_request.input_ids_list,
        )

    @property
    def is_finished(self) -> bool:
        """Check if request is finished."""
        return self.status == RequestStatus.FINISHED

    def activate(self) -> None:
        """Activate this request for processing."""
        self.status = RequestStatus.RUNNING
        self.start_generation_time = time.time()

    def fail(self):
        self.status = RequestStatus.FAILED

    def should_finish(self, token_id: int) -> bool:
        """Check if this request should finish given the latest token.

        Args:
            token_id: The ID of the most recently generated token

        Returns:
            True if the request should finish, False otherwise
        """
        # Check for EOS token
        if self.generation_config.eos_token_id is not None:
            if token_id == self.generation_config.eos_token_id:
                return True

        # Check for maximum length
        if self.num_computed_tokens + 1 >= self.generation_config.max_length:
            return True

        # Check for max_new_tokens
        if len(self.generated_tokens) >= self.generation_config.max_new_tokens:
            return True

        return False

    def preempt(self) -> None:
        """Reset request state for re-prefill after preemption.

        Folds newly generated tokens into input_ids_list so the next prefill replays the
        full sequence, while preserving generated_tokens/logprobs for output tracking.
        """
        already_folded = len(self.input_ids_list) - self._original_prompt_len
        new_tokens = self.generated_tokens[already_folded:]
        self.input_ids_list.extend(new_tokens)
        self.block_ids = []
        self.num_computed_tokens = 0
        self.tokens_this_step = 0
        self.status = RequestStatus.PENDING

    def finish(self):
        """Mark request as finished and release block ownership."""
        self.status = RequestStatus.FINISHED
        self.processing_time = time.time() - self.start_generation_time
        self.tokens_generated = len(self.generated_tokens)
        self.block_ids = []

    def to_driver_request_output(self) -> "DriverRequestOutput":
        """Convert to DriverRequestOutput for IPC."""
        return DriverRequestOutput(
            request_id=self.request_id,
            new_token_ids=self.generated_tokens,
            new_logprobs=self.generated_logprobs if self.generated_logprobs else None,
            status=self.status,
            # TODO: reason
        )


def _generate_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:8]}"


class DriverRequest(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    generation_config: GenerationConfig
    input_ids_list: list[int]
    request_id: str = msgspec.field(default_factory=_generate_request_id)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, GenerationConfig):
            return obj.to_dict()
        raise NotImplementedError(f"Encoding objects of type {type(obj).__name__} is unsupported")

    @staticmethod
    def dec_hook(type_: type, obj: Any) -> Any:
        if type_ is GenerationConfig:
            return GenerationConfig.from_dict(obj)
        raise NotImplementedError(f"Decoding objects of type {type_.__name__} is unsupported")


class DriverRequestOutput(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    request_id: str
    new_token_ids: list[int]
    new_logprobs: list[float] | None = None
    status: RequestStatus = RequestStatus.RUNNING
    reason: str | None = None

    @property
    def is_finished(self) -> bool:
        """Check if request is finished (completed or failed)."""
        return self.status == RequestStatus.FINISHED

    @property
    def is_running(self) -> bool:
        return self.status == RequestStatus.RUNNING
