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

"""Tokenizer utilities for VExact."""

import logging

from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def load_tokenizer(model_path: str, padding_side: str = "left"):
    """
    Load HuggingFace tokenizer with standard configuration.

    Args:
        model_path: Path to the model or model name from HuggingFace Hub
        padding_side: Padding side for the tokenizer (default: "left")

    Returns:
        Configured tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side=padding_side)

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer loaded from {model_path}")
    return tokenizer
