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

from types import SimpleNamespace

import pytest

from vexact.config import ModelConfig


def _gpt_oss_config():
    return SimpleNamespace(model_type="gpt_oss", max_position_embeddings=128)


def test_gpt_oss_requires_fa4_attention():
    with pytest.raises(ValueError, match="learnable attention sinks require"):
        ModelConfig(model_path="unused", attn_impl="fa-invariant", hf_config=_gpt_oss_config())


def test_gpt_oss_accepts_fa4_attention():
    config = ModelConfig(model_path="unused", attn_impl="fa-invariant-cute", hf_config=_gpt_oss_config())

    assert config.attn_impl == "fa-invariant-cute"
