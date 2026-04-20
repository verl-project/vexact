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
    # Extract sampling parameters from configs
    temperatures = [config.temperature for config in gen_configs]
    top_ks = [0 if config.top_k is None else config.top_k for config in gen_configs]
    top_ps = [1.0 if config.top_p is None else config.top_p for config in gen_configs]
    # Apply temperature
    if any(t != 1.0 for t in temperatures):
        temp_tensor = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)
        logits = logits / temp_tensor.unsqueeze(1)
    # Apply top-k filtering
    if any(k > 0 for k in top_ks):
        for i, k in enumerate(top_ks):
            if k > 0:
                values, _ = torch.topk(logits[i], k)
                min_value = values[-1]
                logits[i][logits[i] < min_value] = float("-inf")
    # Apply top-p filtering
    if any(p < 1.0 for p in top_ps):
        for i, p in enumerate(top_ps):
            if p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(probs, dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumsum_probs > p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[i][indices_to_remove] = float("-inf")
    # Convert to probabilities and sample
    probs = torch.softmax(logits, dim=-1)
    # Use greedy sampling if all temperatures are 0 or top_k is 1
    if all(t == 0.0 or k == 1 for t, k in zip(temperatures, top_ks)):
        sampled_tokens = torch.argmax(probs, dim=-1)
    else:
        # Gumbel-max sampling
        noise = torch.empty_like(probs).exponential_(1)
        sampled_tokens = torch.argmax(probs / noise, dim=-1)
    return sampled_tokens


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
