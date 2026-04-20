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

"""Simple test for the generate interface."""

import asyncio

import pytest

from vexact.core.request import DriverRequest, GenerationConfig


@pytest.mark.parametrize(
    "vexact_engine",
    [1, 2],  # pp_size
    indirect=True,
)
def test_generate_simple(vexact_engine):
    """Test simple generation with VeXact."""
    prompt_ids = vexact_engine.tokenizer.encode("Hello, world!")
    request = DriverRequest(
        request_id="test_req_1",
        generation_config=GenerationConfig(max_new_tokens=16),
        input_ids_list=prompt_ids,
    )

    result = asyncio.run(vexact_engine.generate(request, timeout=30.0))

    assert result is not None
    assert result.request_id == "test_req_1"
    assert result.new_token_ids is not None
    assert len(result.new_token_ids) > 0

    generated_text = vexact_engine.tokenizer.decode(result.new_token_ids, skip_special_tokens=True)
    print(f"Generated: {generated_text}")
