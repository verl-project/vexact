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

from tests.conftest import get_tests_attn_impl
from vexact.batch_invariant_ops import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
    is_batch_invariant_mode_enabled,
)
from vexact.config import CacheConfig, ModelConfig, PPInfo, SchedulerConfig, VeXactConfig
from vexact.core.request import InferenceRequest
from vexact.core.runtime_data import GenerationContext, InferencerOutput
from vexact.inferencer.cudagraph_utils import CudaGraphManager
from vexact.inferencer.inferencer import Inferencer
from vexact.inferencer.model_loader import ModelCreator, load_weights_from_weight_path


# Non-default attention backends compare PP hidden-state goldens within one bf16 step.
_BACKEND_HIDDEN_RTOL = 4e-3
_BACKEND_HIDDEN_ATOL = 4e-3


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
        "_attn_implementation": get_tests_attn_impl(),
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
        top_p=1.0,
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
    logits = torch.load(repo_root / "tests/ref_data/inferencer_baseline_logits.pt", map_location="cpu")
    token_ids = torch.load(repo_root / "tests/ref_data/inferencer_baseline_token_ids.pt", map_location="cpu")
    return InferencerOutput(token_ids=token_ids, logits=logits, logprobs=torch.empty(0))


def test_inferencer_generates_expected_token(
    cache_config, config, model, inference_request, baseline_inferenceroutput, device, model_path
):
    vexact_config = VeXactConfig(
        model=ModelConfig(model_path=model_path, attn_impl=get_tests_attn_impl(), hf_config=config),
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


def _real_token_hidden_states(hidden_states: torch.Tensor, gen_ctx: GenerationContext) -> torch.Tensor:
    num_tokens = gen_ctx.batch_position_ids.shape[1]
    return hidden_states[:, :num_tokens, :]


def _assert_tensor_on_device(tensor: torch.Tensor, device: torch.device) -> None:
    expected_device = torch.device(device)
    assert tensor.device.type == expected_device.type
    if expected_device.index is not None:
        assert tensor.device.index == expected_device.index


def _assert_backend_hidden_tensor(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if get_tests_attn_impl() == "fa-invariant":
        assert torch.equal(actual.cpu(), expected.cpu())
        return

    torch.testing.assert_close(
        actual.cpu(),
        expected.cpu(),
        rtol=_BACKEND_HIDDEN_RTOL,
        atol=_BACKEND_HIDDEN_ATOL,
    )


def test_pp_first_rank(repo_root, model_config, model_path, cache_config, inference_request, device):
    pp_info = PPInfo(3, 0)
    model_creator = ModelCreator(model_config, model_path, device, pp_info)
    causal_model = model_creator.create_model()

    vexact_config = VeXactConfig(
        model=ModelConfig(model_path=model_path, attn_impl=get_tests_attn_impl(), hf_config=model_config),
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

    expected_intermediate = torch.load(repo_root / "tests/ref_data/first_rank_intermediate.pt")
    actual_intermediate = _real_token_hidden_states(intermediate_outputs.hidden_states, gen_ctx)
    _assert_tensor_on_device(actual_intermediate, device)
    _assert_backend_hidden_tensor(actual_intermediate, expected_intermediate)


def test_pp_mid_rank(repo_root, model_config, model_path, cache_config, inference_request, device):
    pp_info = PPInfo(3, 1)
    model_creator = ModelCreator(model_config, model_path, device, pp_info)
    causal_model = model_creator.create_model()

    vexact_config = VeXactConfig(
        model=ModelConfig(model_path=model_path, attn_impl=get_tests_attn_impl(), hf_config=model_config),
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
    gen_ctx.intermediate_tensors = torch.load(repo_root / "tests/ref_data/first_rank_intermediate.pt")
    intermediate_outputs = inferencer._forward(gen_ctx)

    expected_intermediate = torch.load(repo_root / "tests/ref_data/mid_rank_intermediate.pt")
    actual_intermediate = _real_token_hidden_states(intermediate_outputs.hidden_states, gen_ctx)
    _assert_tensor_on_device(actual_intermediate, device)
    _assert_backend_hidden_tensor(actual_intermediate, expected_intermediate)


def test_pp_last_rank(
    repo_root, model_config, model_path, cache_config, inference_request, baseline_inferenceroutput, device
):
    pp_info = PPInfo(3, 2)
    model_creator = ModelCreator(model_config, model_path, device, pp_info)
    causal_model = model_creator.create_model()

    vexact_config = VeXactConfig(
        model=ModelConfig(model_path=model_path, attn_impl=get_tests_attn_impl(), hf_config=model_config),
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
    gen_ctx.intermediate_tensors = torch.load(repo_root / "tests/ref_data/mid_rank_intermediate.pt")

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
        top_p=1.0,
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
        model=ModelConfig(
            model_path=model_path,
            attn_impl=get_tests_attn_impl(),
            hf_config=model.config,
            enforce_eager=False,
        ),
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
            model=ModelConfig(
                model_path=model_path,
                attn_impl=get_tests_attn_impl(),
                hf_config=model.config,
                enforce_eager=True,
            ),
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


def _cudagraph_generation_config() -> GenerationConfig:
    generation_config = GenerationConfig(
        max_new_tokens=2,
        max_length=32,
        do_sample=False,
        top_p=1.0,
    )
    generation_config.output_logits = True
    return generation_config


def _zero_kv_cache(inferencer: Inferencer) -> None:
    for cache_tensor in inferencer.cache_store.key_cache.values():
        cache_tensor.zero_()
    for cache_tensor in inferencer.cache_store.value_cache.values():
        cache_tensor.zero_()


def _make_request(
    request_id: str,
    input_ids_list: list[int],
    tokens_this_step: int,
    num_computed_tokens: int,
    block_id: int,
    generation_config: GenerationConfig,
    generated_tokens: list[int] | None = None,
) -> InferenceRequest:
    req = InferenceRequest(
        request_id=request_id,
        input_ids_list=input_ids_list,
        generation_config=generation_config,
        block_ids=[block_id],
    )
    req.generated_tokens = generated_tokens or []
    req.tokens_this_step = tokens_this_step
    req.num_computed_tokens = num_computed_tokens
    return req


def _run_inferencer(
    model: torch.nn.Module,
    model_path: str,
    cache_config: CacheConfig,
    device: torch.device,
    requests: list[InferenceRequest],
    *,
    enforce_eager: bool,
    max_num_seqs: int = 2,
    max_num_batched_tokens: int = 8,
    return_inferencer: bool = False,
    enable_batch_invariant: bool = True,
) -> InferencerOutput | tuple[InferencerOutput, Inferencer]:
    inferencer = Inferencer(
        model=model,
        config=VeXactConfig(
            model=ModelConfig(
                model_path=model_path,
                attn_impl=get_tests_attn_impl(),
                hf_config=model.config,
                enforce_eager=enforce_eager,
            ),
            cache=cache_config,
            scheduler=SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens),
        ),
        pp_info=PPInfo(1, 0),
        pp_messager=None,
        device=device,
        enable_batch_invariant=enable_batch_invariant,
    )
    _zero_kv_cache(inferencer)
    out = inferencer.infer(requests)
    if return_inferencer:
        return out, inferencer
    return out


def _assert_inferencer_outputs_close(out_cg: InferencerOutput, out_eager: InferencerOutput) -> None:
    assert torch.equal(out_cg.token_ids, out_eager.token_ids)
    torch.testing.assert_close(out_cg.logits, out_eager.logits, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cudagraph replay")
def test_inferencer_cudagraph_prefill_matches_eager(model, device, model_path, cache_config):
    generation_config = _cudagraph_generation_config()

    def make_requests(prefix: str) -> list[InferenceRequest]:
        input_ids_list = [1, 2, 3, 4]
        return [
            _make_request(
                f"{prefix}_0",
                input_ids_list=input_ids_list,
                tokens_this_step=len(input_ids_list),
                num_computed_tokens=len(input_ids_list),
                block_id=0,
                generation_config=generation_config,
            )
        ]

    out_cg = _run_inferencer(model, model_path, cache_config, device, make_requests("cg"), enforce_eager=False)
    out_eager = _run_inferencer(model, model_path, cache_config, device, make_requests("eg"), enforce_eager=True)

    _assert_inferencer_outputs_close(out_cg, out_eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cudagraph replay")
def test_inferencer_cudagraph_does_not_require_batch_invariant_mode(model, device, model_path, cache_config):
    generation_config = _cudagraph_generation_config()
    input_ids_list = [1, 2, 3, 4]
    requests = [
        _make_request(
            "cg_no_batch_invariant",
            input_ids_list=input_ids_list,
            tokens_this_step=len(input_ids_list),
            num_computed_tokens=len(input_ids_list),
            block_id=0,
            generation_config=generation_config,
        )
    ]

    batch_invariant_was_enabled = is_batch_invariant_mode_enabled()
    disable_batch_invariant_mode()
    try:
        out_cg, inferencer_cg = _run_inferencer(
            model,
            model_path,
            cache_config,
            device,
            requests,
            enforce_eager=False,
            return_inferencer=True,
            enable_batch_invariant=False,
        )
    finally:
        if batch_invariant_was_enabled:
            enable_batch_invariant_mode()

    assert inferencer_cg._cudagraph_mgr is not None
    assert out_cg.token_ids.numel() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cudagraph replay")
def test_inferencer_cudagraph_mixed_prefill_decode_matches_eager(model, device, model_path, cache_config):
    generation_config = _cudagraph_generation_config()

    def make_requests(prefix: str) -> list[InferenceRequest]:
        prefill_ids = [1, 2, 3]
        decode_ids = [4, 5, 6, 7]
        return [
            _make_request(
                f"{prefix}_prefill",
                input_ids_list=prefill_ids,
                tokens_this_step=len(prefill_ids),
                num_computed_tokens=len(prefill_ids),
                block_id=0,
                generation_config=generation_config,
            ),
            _make_request(
                f"{prefix}_decode",
                input_ids_list=decode_ids,
                tokens_this_step=1,
                num_computed_tokens=len(decode_ids) + 1,
                block_id=1,
                generation_config=generation_config,
                generated_tokens=[8],
            ),
        ]

    out_cg, inferencer_cg = _run_inferencer(
        model, model_path, cache_config, device, make_requests("cg"), enforce_eager=False, return_inferencer=True
    )
    out_eager = _run_inferencer(model, model_path, cache_config, device, make_requests("eg"), enforce_eager=True)

    _assert_inferencer_outputs_close(out_cg, out_eager)
    assert any(desc.num_tokens == 4 and desc.num_seqs >= 2 for desc in inferencer_cg._cudagraph_mgr.graphs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cudagraph replay")
def test_inferencer_cudagraph_chunked_prefill_padded_bucket_matches_eager(model, device, model_path, cache_config):
    generation_config = _cudagraph_generation_config()

    def make_requests(prefix: str) -> list[InferenceRequest]:
        input_ids_list = [1, 2, 3, 4, 5]
        return [
            _make_request(
                f"{prefix}_0",
                input_ids_list=input_ids_list,
                tokens_this_step=len(input_ids_list),
                num_computed_tokens=len(input_ids_list),
                block_id=0,
                generation_config=generation_config,
            )
        ]

    out_cg, inferencer_cg = _run_inferencer(
        model, model_path, cache_config, device, make_requests("cg"), enforce_eager=False, return_inferencer=True
    )
    out_eager = _run_inferencer(model, model_path, cache_config, device, make_requests("eg"), enforce_eager=True)

    _assert_inferencer_outputs_close(out_cg, out_eager)
    assert any(desc.num_tokens == 8 and desc.num_seqs >= 1 for desc in inferencer_cg._cudagraph_mgr.graphs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cudagraph replay")
def test_inferencer_cudagraph_raises_when_no_capture_bucket(model, device, model_path, cache_config):
    generation_config = _cudagraph_generation_config()
    input_ids_list = [1, 2, 3, 4, 5]
    requests = [
        _make_request(
            "too_large",
            input_ids_list=input_ids_list,
            tokens_this_step=len(input_ids_list),
            num_computed_tokens=len(input_ids_list),
            block_id=0,
            generation_config=generation_config,
        )
    ]

    with pytest.raises(RuntimeError, match="no captured descriptor can cover"):
        _run_inferencer(
            model,
            model_path,
            cache_config,
            device,
            requests,
            enforce_eager=False,
            max_num_seqs=1,
            max_num_batched_tokens=4,
        )
