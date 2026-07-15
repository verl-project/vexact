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
from unittest.mock import AsyncMock

import pytest


pytest.importorskip("verl")

from vexact.integrations.verl.async_server import VeXactServer  # noqa: E402
from vexact.integrations.verl.rollout import ServerAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_generate_reports_updated_model_version():
    server = VeXactServer.__new__(VeXactServer)
    server.config = SimpleNamespace(
        max_model_len=16,
        response_length=8,
        prompt_length=4,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        do_sample=False,
    )
    server.tokenizer = SimpleNamespace(pad_token_id=0)
    server._eos_token_id = 1
    server.global_steps = None
    server.engine = SimpleNamespace(
        driver_client=SimpleNamespace(receive_weights=lambda: None),
        generate=AsyncMock(return_value=SimpleNamespace(new_token_ids=[2, 3], new_logprobs=None, reason="stop")),
    )

    await server.receive_weights(global_steps=7)
    output = await server.generate(prompt_ids=[4], sampling_params={}, request_id="request")

    assert output.extra_fields["global_steps"] == 7


@pytest.mark.asyncio
async def test_failed_weight_update_keeps_previous_model_version():
    def fail_receive_weights():
        raise RuntimeError("weight transfer failed")

    server = VeXactServer.__new__(VeXactServer)
    server.global_steps = 6
    server.engine = SimpleNamespace(driver_client=SimpleNamespace(receive_weights=fail_receive_weights))

    with pytest.raises(RuntimeError, match="weight transfer failed"):
        await server.receive_weights(global_steps=7)

    assert server.global_steps == 6


class _RemoteMethod:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def remote(self, *args, **kwargs):
        self.calls.append((self.name, args, kwargs))

        async def result():
            return None

        return result()


@pytest.mark.asyncio
async def test_weight_update_sets_model_version_after_transfer(monkeypatch):
    calls = []
    server_handle = SimpleNamespace(
        receive_weights=_RemoteMethod("receive_weights", calls),
        clear_kv_cache=_RemoteMethod("clear_kv_cache", calls),
    )
    adapter = ServerAdapter.__new__(ServerAdapter)
    adapter.rollout_rank = 0
    adapter.server_handle = server_handle
    adapter.zmq_handle = "inproc://unused"
    adapter.use_shm = False
    adapter._get_server_handle = lambda: server_handle

    class _Sender:
        def __init__(self, **kwargs):
            pass

        def send_weights(self, weights):
            list(weights)

    monkeypatch.setattr("vexact.integrations.verl.bucketed_weight_transfer.BucketedWeightSender", _Sender)
    monkeypatch.setattr("vexact.integrations.verl.rollout.get_device_name", lambda: "cpu")
    monkeypatch.setattr("vexact.integrations.verl.rollout.get_device_id", lambda: 0)

    await adapter.update_weights(iter(()), global_steps=7)

    assert calls == [
        ("receive_weights", (), {"global_steps": 7}),
        ("clear_kv_cache", (), {}),
    ]
