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

import pytest
import torch
from transformers import GenerationConfig

import vexact.inferencer.sampler as sampler_module
from vexact.inferencer.sampler import Sampler


def _batch_sample_token_ref(
    logits: torch.Tensor,
    gen_configs: list[GenerationConfig],
) -> torch.Tensor:
    """
    Sample tokens from logits for a batch of sequences.
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        gen_configs: List of generation configs, one per batch item
    Returns:
        Sampled token indices of shape (batch_size,)
    """
    sampler = Sampler()
    gen_config = gen_configs[0]

    top_k = gen_config.top_k
    top_p = gen_config.top_p
    assert top_p is not None, "top_p must be set (use 1.0 to disable)"
    if top_k is not None and int(top_k) <= 0:
        top_k = None
    temperature = gen_config.temperature
    do_sample = gen_config.do_sample and temperature > 0 and float(top_p) > 0
    if not do_sample:
        return logits.argmax(dim=-1).view(-1)

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None:
        top_k = torch.full((logits.shape[0],), int(top_k), device=logits.device)
    if top_p is not None:
        top_p = torch.full((logits.shape[0],), float(top_p), device=logits.device)
    logits = sampler.apply_top_k_top_p(logits, top_k, top_p)

    # Convert to probabilities and sample
    probs = torch.softmax(logits, dim=-1)
    # Gumbel-max sampling
    noise = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / noise, dim=-1)


@pytest.mark.parametrize("batch_size", [1, 2048, 4096])
@pytest.mark.parametrize("vocab_size", [151936])
@pytest.mark.parametrize("top_k", [0, 50, None])
@pytest.mark.parametrize("top_p", [0, 0.9])
def test_batch_sample_token_matches_sample_token_for_loop(batch_size, vocab_size, top_k, top_p):
    sampler = Sampler()
    device = torch.device("cuda:0")

    logits = torch.empty((batch_size, vocab_size), device=device).uniform_(-1.0, 1.0)

    gen_config = GenerationConfig(do_sample=True, temperature=1.0, top_k=top_k, top_p=top_p)
    gen_configs = [gen_config for _ in range(batch_size)]

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    ref_tokens = _batch_sample_token_ref(logits.clone(), gen_configs)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    batch_tokens = sampler.batch_sample_token(logits.clone(), gen_configs)

    torch.testing.assert_close(batch_tokens, ref_tokens, rtol=0, atol=0)


def test_seeded_batch_sample_token_rejects_mixed_seed_batch():
    sampler = Sampler()
    logits = torch.randn(2, 16)
    gen_configs = [
        GenerationConfig(do_sample=True, temperature=1.0, top_k=4, top_p=1.0),
        GenerationConfig(do_sample=True, temperature=1.0, top_k=4, top_p=1.0),
    ]
    gen_configs[0].seed = 1234

    with pytest.raises(ValueError, match="every request in the batch"):
        sampler.batch_sample_token(logits, gen_configs, torch.arange(2))


def test_seeded_batch_sample_token_requires_vllm_gumbel_kernel(monkeypatch):
    sampler = Sampler()
    logits = torch.randn(2, 16)
    gen_configs = [
        GenerationConfig(do_sample=True, temperature=1.0, top_k=4, top_p=1.0),
        GenerationConfig(do_sample=True, temperature=1.0, top_k=4, top_p=1.0),
    ]
    for gen_config in gen_configs:
        gen_config.seed = 1234

    monkeypatch.setattr(sampler_module, "vllm_gumbel_sample", None)
    with pytest.raises(ImportError, match="vLLM is required for seeded sampling"):
        sampler.batch_sample_token(logits, gen_configs, torch.arange(2))


def test_seeded_batch_sample_token_requires_sampling_positions():
    sampler = Sampler()
    logits = torch.randn(2, 16)
    gen_configs = [
        GenerationConfig(do_sample=True, temperature=1.0, top_k=4, top_p=1.0),
        GenerationConfig(do_sample=True, temperature=1.0, top_k=4, top_p=1.0),
    ]
    for gen_config in gen_configs:
        gen_config.seed = 1234

    with pytest.raises(ValueError, match="sampling_positions must be set"):
        sampler.batch_sample_token(logits, gen_configs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Seeded sampler requires CUDA.")
@pytest.mark.skipif(sampler_module.vllm_gumbel_sample is None, reason="Seeded sampler requires vLLM.")
@pytest.mark.parametrize("top_k, top_p", [(50, 1.0), (50, 0.9)])
def test_seeded_batch_sample_token_is_repeatable(top_k, top_p):
    sampler = Sampler()
    device = torch.device("cuda:0")
    batch_size = 8
    vocab_size = 4096

    logits = torch.empty((batch_size, vocab_size), device=device).uniform_(-1.0, 1.0)
    positions = torch.arange(17, 17 + batch_size, device=device, dtype=torch.long)
    gen_configs = []
    for i in range(batch_size):
        gen_config = GenerationConfig(do_sample=True, temperature=1.0, top_k=top_k, top_p=top_p)
        gen_config.seed = 1234 + i
        gen_configs.append(gen_config)

    first_tokens = sampler.batch_sample_token(logits.clone(), gen_configs, positions)
    torch.empty((1024,), device=device).exponential_()
    second_tokens = sampler.batch_sample_token(logits.clone(), gen_configs, positions)

    torch.testing.assert_close(second_tokens, first_tokens, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Seeded sampler requires CUDA.")
@pytest.mark.skipif(sampler_module.vllm_gumbel_sample is None, reason="Seeded sampler requires vLLM.")
def test_seeded_batch_sample_token_is_request_order_invariant():
    sampler = Sampler()
    device = torch.device("cuda:0")
    batch_size = 8
    vocab_size = 4096

    logits = torch.empty((batch_size, vocab_size), device=device).uniform_(-1.0, 1.0)
    positions = torch.arange(23, 23 + batch_size, device=device, dtype=torch.long)
    gen_configs = []
    for i in range(batch_size):
        gen_config = GenerationConfig(do_sample=True, temperature=1.0, top_k=50, top_p=0.9)
        gen_config.seed = 4321 + i
        gen_configs.append(gen_config)

    tokens = sampler.batch_sample_token(logits.clone(), gen_configs, positions)
    permutation = torch.tensor([3, 0, 7, 1, 4, 2, 6, 5], device=device)
    permuted_tokens = sampler.batch_sample_token(
        logits.index_select(0, permutation).clone(),
        [gen_configs[i] for i in permutation.cpu().tolist()],
        positions.index_select(0, permutation),
    )

    torch.testing.assert_close(permuted_tokens, tokens.index_select(0, permutation), rtol=0, atol=0)
