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

import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PretrainedConfig

from vexact.config import CacheConfig, ModelConfig, PPInfo, SchedulerConfig, VeXactConfig
from vexact.core.request import InferenceRequest
from vexact.core.runtime_data import InferencerOutput
from vexact.inferencer.cudagraph_utils import CudaGraphManager
from vexact.inferencer.inferencer import Inferencer
from vexact.inferencer.model_loader import ModelCreator, load_weights_from_weight_path


@pytest.fixture(scope="module")
def total_hidden_layers() -> int:
    return 3


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda")


@pytest.fixture(scope="module")
def model_path() -> str:
    return os.environ["VEXACT_TESTS_MODEL_PATH"]


@pytest.fixture(scope="module")
def model_config(model_path, device, total_hidden_layers) -> PretrainedConfig:
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": device},  # Place entire model on target device
        "trust_remote_code": True,
        "_attn_implementation": "fa-invariant",
    }
    config = AutoConfig.from_pretrained(model_path, **model_kwargs)
    config.num_hidden_layers = total_hidden_layers
    return config


@pytest.fixture
def config(model_path):
    return AutoConfig.from_pretrained(model_path)


@pytest.fixture
def tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)


@pytest.fixture
def cache_config() -> CacheConfig:
    return CacheConfig(page_size=256, max_cache_blocks=1024)


@pytest.fixture
def inference_request(tokenizer) -> InferenceRequest:
    generation_config = GenerationConfig(
        max_new_tokens=20,
        max_length=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt = "The old lighthouse keeper, who had spent nearly forty years watching over the rocky coastline and guiding ships safely through the treacherous waters during countless storms, finally decided on a misty autumn morning that it was time to retire and pass the responsibility to someone younger, someone with sharper eyes and steadier hands, though he knew he would deeply miss the solitary beauty of the crashing waves, the calls of the seabirds at dawn, the smell of salt air, and especially those quiet moments at sunset when the world seemed to pause and he felt truly at peace with his life's work.Retry"  # noqa: E501
    messages = [{"role": "user", "content": prompt}]
    prompt_with_chat_template = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(
        prompt_with_chat_template,
        padding=False,
        truncation=True,
        max_length=generation_config.max_length // 2,
    )

    input_len = len(inputs["input_ids"])
    req = InferenceRequest(
        request_id="sim_req_7f2829de",
        input_ids_list=inputs["input_ids"],
        generation_config=generation_config,
        block_ids=[82],
    )
    req.tokens_this_step = input_len
    req.num_computed_tokens = input_len
    return req


@pytest.fixture
def model(model_config, model_path, device) -> torch.nn.Module:
    causal_model = AutoModelForCausalLM.from_config(model_config).to(device)
    load_weights_from_weight_path(causal_model, model_config, model_path)
    return causal_model


@pytest.fixture
def baseline_inferenceroutput(repo_root) -> InferencerOutput:
    logits = torch.load(repo_root / "vexact/tests/ref_data/inferencer_baseline_logits.pt", map_location="cpu")
    token_ids = torch.load(repo_root / "vexact/tests/ref_data/inferencer_baseline_token_ids.pt", map_location="cpu")
    return InferencerOutput(token_ids=token_ids, logits=logits, logprobs=torch.empty(0))


def test_inferencer_generates_expected_token(
    cache_config, config, model, inference_request, baseline_inferenceroutput, device, model_path
):
    vexact_config = VeXactConfig(
        model=ModelConfig(model_path=model_path, hf_config=config),
        cache=cache_config,
        scheduler=SchedulerConfig(),
    )
    inferencer = Inferencer(
        model=model,
        config=vexact_config,
        pp_info=PPInfo(1, 0),
        pp_messager=None,
        device=device,
        enable_batch_invariant=True,
    )

    result = inferencer.infer([inference_request])

    assert torch.equal(result.token_ids.cpu(), baseline_inferenceroutput.token_ids)
    for i in range(len(result.logits)):
        assert torch.equal(result.logits[i].cpu(), baseline_inferenceroutput.logits[i])


def test_pp_first_rank(repo_root, model_config, model_path, cache_config, inference_request, device):
    pp_info = PPInfo(3, 0)
    model_creator = ModelCreator(model_config, model_path, device, pp_info)
    causal_model = model_creator.create_model()

    vexact_config = VeXactConfig(
        model=ModelConfig(model_path=model_path, hf_config=model_config),
        cache=cache_config,
        scheduler=SchedulerConfig(),
    )
    inferencer = Inferencer(
        model=causal_model,
        config=vexact_config,
        pp_info=pp_info,
        pp_messager=None,
        device=device,
        enable_batch_invariant=True,
    )

    gen_ctx = inferencer._prepare_gen_ctx([inference_request])
    intermediate_outputs = inferencer._forward(gen_ctx)

    expected_intermediate = torch.load(repo_root / "vexact/tests/ref_data/first_rank_intermediate.pt")
    assert torch.equal(intermediate_outputs.hidden_states, expected_intermediate)


def test_pp_mid_rank(repo_root, model_config, model_path, cache_config, inference_request, device):
    pp_info = PPInfo(3, 1)
    model_creator = ModelCreator(model_config, model_path, device, pp_info)
    causal_model = model_creator.create_model()

    vexact_config = VeXactConfig(
        model=ModelConfig(model_path=model_path, hf_config=model_config),
        cache=cache_config,
        scheduler=SchedulerConfig(),
    )
    inferencer = Inferencer(
        model=causal_model,
        config=vexact_config,
        pp_info=pp_info,
        pp_messager=None,
        device=device,
        enable_batch_invariant=True,
    )

    gen_ctx = inferencer._prepare_gen_ctx([inference_request])
    gen_ctx.batch_input_ids = None
    gen_ctx.intermediate_tensors = torch.load(repo_root / "vexact/tests/ref_data/first_rank_intermediate.pt")
    intermediate_outputs = inferencer._forward(gen_ctx)

    expected_intermediate = torch.load(repo_root / "vexact/tests/ref_data/mid_rank_intermediate.pt")
    assert torch.equal(intermediate_outputs.hidden_states, expected_intermediate)


def test_pp_last_rank(
    repo_root, model_config, model_path, cache_config, inference_request, baseline_inferenceroutput, device
):
    pp_info = PPInfo(3, 2)
    model_creator = ModelCreator(model_config, model_path, device, pp_info)
    causal_model = model_creator.create_model()

    vexact_config = VeXactConfig(
        model=ModelConfig(model_path=model_path, hf_config=model_config),
        cache=cache_config,
        scheduler=SchedulerConfig(),
    )
    inferencer = Inferencer(
        model=causal_model,
        config=vexact_config,
        pp_info=pp_info,
        pp_messager=None,
        device=device,
        enable_batch_invariant=True,
    )

    gen_ctx = inferencer._prepare_gen_ctx([inference_request])
    gen_ctx.batch_input_ids = None
    gen_ctx.intermediate_tensors = torch.load(repo_root / "vexact/tests/ref_data/mid_rank_intermediate.pt")

    outputs = inferencer._forward(gen_ctx)

    token_ids, logits, logprobs = inferencer._select_tokens(
        [inference_request.generation_config], [len(inference_request.generated_tokens)], outputs, gen_ctx
    )
    assert torch.equal(token_ids.cpu(), baseline_inferenceroutput.token_ids)
    for i in range(len(logits)):
        assert torch.equal(logits[i].cpu(), baseline_inferenceroutput.logits[i])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cudagraph replay")
def test_inferencer_cudagraph_decode_matches_eager(model, device, model_path, cache_config, monkeypatch):
    generation_config = GenerationConfig(
        max_new_tokens=2,
        max_length=16,
        do_sample=False,
    )
    generation_config.output_logits = True

    def make_decode_request(request_id: str, block_id: int) -> InferenceRequest:
        input_ids_list = [1, 2, 3, 4]
        req = InferenceRequest(
            request_id=request_id,
            input_ids_list=input_ids_list,
            generation_config=generation_config,
            block_ids=[block_id],
        )
        req.generated_tokens = [1]
        req.tokens_this_step = 1
        req.num_computed_tokens = len(input_ids_list) + len(req.generated_tokens)
        return req

    vexact_config = VeXactConfig(
        model=ModelConfig(model_path=model_path, hf_config=model.config, enforce_eager=False),
        cache=cache_config,
        scheduler=SchedulerConfig(max_num_batched_tokens=2),
    )

    orig_capture_graphs = CudaGraphManager.capture_graphs

    def _safe_capture_graphs(self: CudaGraphManager) -> None:
        if not self.capture_sizes:
            return
        max_size = max(self.capture_sizes)
        buffers = self.input_buffers
        buffers.input_ids.gpu.zero_()
        buffers.position_ids.gpu.zero_()
        buffers.block_tables.gpu.fill_(-1)
        if max_size > 0:
            zero_block_ids = torch.zeros(max_size, device=self.device, dtype=torch.int32)
            buffers.block_tables.gpu[:max_size, 0].copy_(zero_block_ids)
            buffers.slot_mapping.gpu[:max_size].zero_()
        orig_capture_graphs(self)

    monkeypatch.setattr(CudaGraphManager, "capture_graphs", _safe_capture_graphs)

    inferencer_cg = Inferencer(
        model=model,
        config=vexact_config,
        pp_info=PPInfo(1, 0),
        pp_messager=None,
        device=device,
        enable_batch_invariant=True,
    )
    for cache_tensor in inferencer_cg.cache_store.key_cache.values():
        cache_tensor.zero_()
    for cache_tensor in inferencer_cg.cache_store.value_cache.values():
        cache_tensor.zero_()
    requests_cg = [make_decode_request("cg_0", 0), make_decode_request("cg_1", 1)]
    out_cg = inferencer_cg.infer(requests_cg)

    inferencer_eager = Inferencer(
        model=model,
        config=VeXactConfig(
            model=ModelConfig(model_path=model_path, hf_config=model.config, enforce_eager=True),
            cache=cache_config,
            scheduler=SchedulerConfig(),
        ),
        pp_info=PPInfo(1, 0),
        pp_messager=None,
        device=device,
        enable_batch_invariant=True,
    )
    for cache_tensor in inferencer_eager.cache_store.key_cache.values():
        cache_tensor.zero_()
    for cache_tensor in inferencer_eager.cache_store.value_cache.values():
        cache_tensor.zero_()
    requests_eager = [make_decode_request("eg_0", 0), make_decode_request("eg_1", 1)]
    out_eager = inferencer_eager.infer(requests_eager)

    assert torch.equal(out_cg.token_ids, out_eager.token_ids)
    if out_cg.logits.numel() > 0 or out_eager.logits.numel() > 0:
        torch.testing.assert_close(out_cg.logits, out_eager.logits, atol=0, rtol=0)
