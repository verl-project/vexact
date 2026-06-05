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

import torch
import triton.profiler as proton
from torch import Tensor
from transformers import GenerationConfig


try:
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p as vllm_apply_top_k_top_p
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample as vllm_gumbel_sample
except ImportError:
    vllm_apply_top_k_top_p = None
    vllm_gumbel_sample = None


class Sampler:
    @proton.cpu_timed_scope("_batch_sample_token")
    @torch.inference_mode()
    def batch_sample_token(
        self,
        batched_logits: Tensor,
        gen_configs: list[GenerationConfig],
        sampling_positions: Tensor | None = None,
    ) -> Tensor:
        """Batch sample next tokens for a batch of logits and generation configs.

        Args:
            batched_logits: Tensor of shape (batch_size, vocab_size)
            gen_configs: Sequence of GenerationConfig objects for each item in the batch
            sampling_positions: Absolute token positions for each sampled row.
        """

        batched_logits = batched_logits.to(torch.float32)
        # we assume all k and p are the same in the batch
        k = gen_configs[0].top_k
        p = gen_configs[0].top_p

        assert p is not None, "top_p must be set (use 1.0 to disable)"
        if k is not None and int(k) <= 0:
            k = None
        temp = gen_configs[0].temperature
        do_sample = gen_configs[0].do_sample and gen_configs[0].temperature > 0 and float(p) > 0
        if do_sample:
            seeds = [getattr(gen_config, "seed", None) for gen_config in gen_configs]
            has_seed = [seed is not None for seed in seeds]
            if any(has_seed):
                if not all(has_seed):
                    raise ValueError("Seeded sampling requires every request in the batch to set seed.")
                if sampling_positions is None:
                    raise ValueError("sampling_positions must be set for seeded sampling.")
                if vllm_gumbel_sample is None:
                    raise ImportError("vLLM is required for seeded sampling but is not importable.")
            if k is not None:
                k = torch.full((batched_logits.shape[0],), int(k), device=batched_logits.device)
            if p is not None:
                p = torch.full((batched_logits.shape[0],), float(p), device=batched_logits.device)
            # apply temperature
            if temp != 1.0:
                batched_logits.div_(temp)
            # apply top-k and top-p
            logits = self.apply_top_k_top_p(batched_logits, k, p)
            if any(has_seed):
                next_tokens = self._seeded_sample(
                    logits, [int(seed) for seed in seeds], sampling_positions, float(temp)
                )
            else:
                # sample tokens
                probs = torch.softmax(logits, dim=-1)
                next_tokens = self._random_sample(probs, generators={})
        else:
            # greedy
            next_tokens = batched_logits.argmax(dim=-1).view(-1)

        return next_tokens.to(device=batched_logits.device, dtype=torch.long)

    @proton.cpu_timed_scope("_seeded_sample")
    @torch.inference_mode()
    def _seeded_sample(
        self,
        logits: Tensor,
        seeds: list[int],
        sampling_positions: Tensor,
        temperature: float,
    ) -> Tensor:
        """Sample with vLLM's per-request seeded Triton Gumbel kernel."""
        if vllm_gumbel_sample is None:
            raise ImportError("vLLM is required for seeded sampling but is not importable.")

        device = logits.device
        num_reqs = logits.shape[0]
        idx_mapping = torch.arange(num_reqs, device=device, dtype=torch.long)
        seed_tensor = torch.tensor(seeds, device=device, dtype=torch.int64)
        temperature_tensor = torch.full((num_reqs,), temperature, device=device, dtype=torch.float32)
        pos_tensor = sampling_positions.to(device=device, dtype=torch.long)

        return vllm_gumbel_sample(
            logits,
            idx_mapping,
            temperature_tensor,
            seed_tensor,
            pos_tensor,
            apply_temperature=False,
        )

    @proton.cpu_timed_scope("_random_sample")
    @torch.inference_mode()
    def _random_sample(
        self,
        probs: torch.Tensor,
        generators: dict[int, torch.Generator],
    ) -> torch.Tensor:
        """Randomly sample from the probabilities.

        We use this function instead of torch.multinomial because torch.multinomial
        causes CPU-GPU synchronization.
        """
        q = torch.empty_like(probs)
        # NOTE(woosuk): To batch-process the requests without their own seeds,
        # which is the common case, we first assume that every request does
        # not have its own seed. Then, we overwrite the values for the requests
        # that have their own seeds.
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            # TODO(woosuk): This can be slow because we handle each request
            # one by one. Optimize this.
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
        return probs.div_(q).argmax(dim=-1).view(-1)

    @proton.cpu_timed_scope("apply_top_k_top_p")
    @torch.inference_mode()
    def apply_top_k_top_p(
        self,
        batched_logits: Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> Tensor:
        """
        Reference: https://github.com/vllm-project/vllm/blob/8e2a469b3b2f67bc900ed72724fe3f05e3564994/vllm/v1/sample/ops/topk_topp_sampler.py#L243
        """
        if vllm_apply_top_k_top_p is not None:
            return vllm_apply_top_k_top_p(batched_logits, k, p)

        if p is None:
            if k is None:
                return batched_logits
            else:
                # Avoid sorting vocab for top-k only case.
                return self.apply_top_k_only(batched_logits, k)
        logits_sort, logits_idx = batched_logits.sort(dim=-1, descending=False)
        if k is not None:
            # Apply top-k.
            top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
            # Get all the top_k values.
            top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
            top_k_mask = logits_sort < top_k_mask
            logits_sort.masked_fill_(top_k_mask, -float("inf"))

        if p is not None:
            # Apply top-p.
            probs_sort = logits_sort.softmax(dim=-1)
            probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
            top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
            # at least one
            top_p_mask[:, -1] = False
            logits_sort.masked_fill_(top_p_mask, -float("inf"))

        # Re-sort the probabilities.
        logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)

        return logits

    @proton.cpu_timed_scope("apply_top_k_only")
    @torch.inference_mode()
    def apply_top_k_only(
        self,
        logits: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply top-k mask to the logits.

        This implementation doesn't involve sorting the entire vocab.

        The logits tensor may be updated in-place.
        """
        no_top_k_mask = k == logits.shape[1]
        # Set non-top-k rows to 1 so that we can gather.
        k = k.masked_fill(no_top_k_mask, 1)
        max_top_k = k.max()
        # topk.values tensor has shape [batch_size, max_top_k].
        # Convert top k to 0-based index in range [0, max_top_k).
        k_index = k.sub_(1).unsqueeze(1)
        top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
        # Handle non-topk rows.
        top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
        logits.masked_fill_(logits < top_k_mask, -float("inf"))
        return logits
