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

import logging
from typing import Any, Optional, Sequence

import numpy as np
import torch
import triton.profiler as proton
from torch import Tensor
from transformers import GenerationConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import ModelOutput

from vexact.batch_invariant_ops import (
    enable_batch_invariant_mode,
    flex_attention_forward,
    is_batch_invariant_mode_enabled,
)
from vexact.batch_invariant_ops import flash_attention_forward as flash_attention_forward_impl
from vexact.batch_invariant_ops import flash_attention_forward_cute as flash_attention_forward_cute_impl
from vexact.batch_invariant_ops.kv_cache_context import (
    KVCacheStore,
    set_kv_cache_context,
)
from vexact.batch_invariant_ops.standalone_logprobs import logprobs_from_logits_flash_attn
from vexact.config import PPInfo, VeXactConfig
from vexact.core.request import InferenceRequest
from vexact.core.runtime_data import GenerationContext, InferencerOutput, InputBuffers
from vexact.distributed.pp_messager import PPMessager
from vexact.inferencer.cudagraph_utils import CudaGraphManager
from vexact.inferencer.sampler import Sampler
from vexact.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


# Module-level logger
logger = logging.getLogger(__name__)

# Register two invariant attention implementations
ALL_ATTENTION_FUNCTIONS["flex"] = flex_attention_forward
ALL_ATTENTION_FUNCTIONS["fa-invariant"] = flash_attention_forward_impl
ALL_ATTENTION_FUNCTIONS["fa-invariant-cute"] = flash_attention_forward_cute_impl


class Inferencer:
    """Run a single forward pass for a batch of requests and sample next tokens."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: VeXactConfig,
        pp_info: PPInfo,
        pp_messager: Optional[PPMessager],
        device: torch.device,
        enable_batch_invariant: bool,
    ) -> None:
        self.model = model
        self.device = device
        self._pp_info = pp_info
        self._pp_messager = pp_messager
        self.cache_config = config.cache
        self.forward_kwargs = dict(
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
            use_fused_lce=False,  # we need logits in model inference outputs
        )
        self.sampler = Sampler()
        # When training side uses fused lce, logits are upcasted to fp32 for computing logprobs
        # Eager path computes log_probs from bf16 logits to save GPU memory
        self.use_fp32_logits = bool(config.model.use_fp32_logits)
        logger.info(f"[VEXACT] Inferencer: {self.use_fp32_logits=}")

        with TorchMemorySaverAdapter.get_instance().region("kv_cache", enable_cpu_backup=False):
            self.cache_store = KVCacheStore(config.model.hf_config, self.cache_config, device)

        # TODO: remove them here, create block table helper to get slot mappings
        self.page_size = self.cache_config.page_size

        if not is_batch_invariant_mode_enabled():
            enable_batch_invariant_mode()

        self.input_buffers = InputBuffers(
            device=self.device,
            max_num_seqs=config.scheduler.max_num_seqs,
            max_num_batched_tokens=config.scheduler.max_num_batched_tokens,
            max_blocks_per_req=(config.model.max_model_len + self.page_size - 1) // self.page_size,
            # TODO: probably we should get real hidden_size for intemediate tensors buffer from model warmup forward
            hidden_size=config.model.hf_config.hidden_size,
        )

        self._cudagraph_mgr: Optional[CudaGraphManager] = None
        should_enforce_eager = config.model.enforce_eager
        if not should_enforce_eager:
            logger.info("Enabling cudagraph...")
            # CUDA graphs only replay for decode-only batches where each sequence
            # contributes exactly 1 token, so the max batch size is bounded by
            # both max_num_seqs and max_num_batched_tokens.
            cudagraph_max_size = min(config.scheduler.max_num_seqs, config.scheduler.max_num_batched_tokens)
            self._cudagraph_mgr = CudaGraphManager(
                model=self.model,
                device=self.device,
                cache_store=self.cache_store,
                max_size=cudagraph_max_size,
                cache_config=self.cache_config,
                forward_kwargs=self.forward_kwargs,
                input_buffers=self.input_buffers,
            )
            self._cudagraph_mgr.capture_graphs()
        else:
            logger.info("Cudagraphs disabled because enforce_eager is set or PP size > 1.")

    def infer(self, requests: Sequence[InferenceRequest], recv_infer_out: bool = False) -> Optional[InferencerOutput]:
        """
        The function defines the sequence of GPU kernels in one cycle.

        |pp rank| gen_ctx | forward | infer_out | gen_ctx |
        |-------|---------|---------|-----------|---------|
        | first | prep    | yes     | recv      | send    |
        | mid   | recv    | yes     | no        | send    |
        | last  | recv    | yes     | send      | no      |

        Args:
            requests: the sequence of request that's in the batch for this cycle
            recv_infer_out: determine whether inference output from the last pp rank will be received this cycle
        """
        gen_ctx = None
        if self._pp_info.is_first_rank:
            if len(requests) > 0:
                gen_ctx = self._prepare_gen_ctx(requests)
        else:
            gen_ctx = self._pp_messager.recv_gen_ctx()

            set_kv_cache_context(
                is_paged_attn=True,
                key_cache=self.cache_store.key_cache,
                value_cache=self.cache_store.value_cache,
                block_tables=gen_ctx.block_tables,
                context_lens=gen_ctx.context_lens,
                slot_mapping=gen_ctx.slot_mapping,
                query_start_loc=gen_ctx.query_start_loc,
                max_seqlen_q=gen_ctx.max_seqlen_q,
            )

        if gen_ctx is not None:
            outputs = self._forward(gen_ctx)

        if self._pp_info.is_last_rank and gen_ctx is not None:
            token_ids, logits, logprobs = self._select_tokens(
                gen_ctx.generation_configs, gen_ctx.tokens_generated, outputs, gen_ctx
            )
            final_output = InferencerOutput(token_ids=token_ids, logits=logits, logprobs=logprobs)
            if self._pp_info.is_first_rank:
                return final_output
            else:
                self._pp_messager.send_infer_out(final_output)

        final_output = None

        if self._pp_info.is_first_rank and recv_infer_out:
            final_output = self._pp_messager.recv_infer_out()

        if not self._pp_info.is_last_rank and gen_ctx is not None:
            num_tokens = gen_ctx.batch_position_ids.shape[1]
            intermediate_output = GenerationContext(
                batch_input_ids=None,
                # We need to slice here to actual number of tokens since fake tokens are filled for cudagraph
                intermediate_tensors=outputs.hidden_states[:, :num_tokens, :],
                query_start_loc=gen_ctx.query_start_loc,
                batch_position_ids=gen_ctx.batch_position_ids,
                block_tables=gen_ctx.block_tables,
                context_lens=gen_ctx.context_lens,
                slot_mapping=gen_ctx.slot_mapping,
                max_seqlen_q=gen_ctx.max_seqlen_q,
                generation_configs=gen_ctx.generation_configs,
                tokens_generated=gen_ctx.tokens_generated,
                is_decode_only=gen_ctx.is_decode_only,
            )
            self._pp_messager.send_gen_ctx(intermediate_output)

        return final_output

    def _prepare_gen_ctx(self, requests: Sequence[InferenceRequest]) -> GenerationContext:
        num_seqs = len(requests)
        max_blocks_per_req = self.input_buffers.max_blocks_per_req

        input_lengths = []
        current_tokens = []
        total_tokens = 0

        # First pass: collect metadata and compute total tokens
        for req in requests:
            input_lengths.append(len(req.input_ids_list))
            tokens_this_step = req.tokens_this_step
            current_tokens.append(tokens_this_step)
            total_tokens += tokens_this_step

        # Build slot_mapping on CPU
        # Init slot to -1 to skip any write to unused KV block
        slot_mapping = np.full((total_tokens,), -1, dtype=np.int32)
        slot_idx = 0
        for req in requests:
            start_pos = req.num_computed_tokens - req.tokens_this_step
            assert start_pos >= 0

            for pos in range(start_pos, start_pos + req.tokens_this_step):
                block_idx = pos // self.page_size
                block_offset = pos % self.page_size
                block_id = req.block_ids[block_idx]
                slot_mapping[slot_idx] = block_id * self.page_size + block_offset
                slot_idx += 1

        # Build context_lens on CPU
        context_lens = np.empty((num_seqs,), dtype=np.int32)
        for i, req in enumerate(requests):
            context_lens[i] = req.num_computed_tokens

        # Build block_tables on CPU
        block_tables = np.full((num_seqs, max_blocks_per_req), -1, dtype=np.int32)
        for i, req in enumerate(requests):
            if req.block_ids:
                for j, block_id in enumerate(req.block_ids):
                    block_tables[i, j] = block_id

        # Build query_start_loc on CPU
        query_start_loc = np.empty((num_seqs + 1,), dtype=np.int32)
        query_start_loc[0] = 0
        for i in range(num_seqs):
            query_start_loc[i + 1] = query_start_loc[i] + current_tokens[i]

        max_seqlen_q = max(current_tokens) if current_tokens else 0

        # Transfer metadata to GPU (non-blocking)
        block_tables = torch.from_numpy(block_tables).to(self.device, non_blocking=True)
        slot_mapping = torch.from_numpy(slot_mapping).to(self.device, non_blocking=True)
        context_lens = torch.from_numpy(context_lens).to(self.device, non_blocking=True)
        query_start_loc = torch.from_numpy(query_start_loc).to(self.device, non_blocking=True)

        set_kv_cache_context(
            is_paged_attn=True,
            key_cache=self.cache_store.key_cache,
            value_cache=self.cache_store.value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            max_seqlen_q=max_seqlen_q,
        )

        # Build input sequences # TODO: clean up position ids generation
        current_sequences: list[int] = []
        position_ids_sequences: list[np.ndarray] = []
        for idx, req in enumerate(requests):
            if req.num_computed_tokens - current_tokens[idx] < len(req.input_ids_list):
                # Prefill (or re-prefill after preemption): feed the input tokens for this chunk
                start_pos = req.num_computed_tokens - current_tokens[idx]
                assert start_pos >= 0
                end_pos = req.num_computed_tokens
                current_sequences.extend(req.input_ids_list[start_pos:end_pos])
                position_ids_sequences.append(np.arange(start_pos, end_pos, dtype=np.int64))
            else:
                # Decode: feed the last generated token; position = num_computed_tokens - 1
                last_token = req.generated_tokens[-1]
                current_sequences.append(last_token)
                position_ids_sequences.append(np.array([req.num_computed_tokens - 1], dtype=np.int64))

        batch_input_ids = torch.from_numpy(np.array(current_sequences, dtype=np.int64)).to(self.device).unsqueeze(0)
        batch_position_ids = torch.from_numpy(np.concatenate(position_ids_sequences)).to(self.device).unsqueeze(0)

        is_decode_only = all(
            req.num_computed_tokens - req.tokens_this_step >= len(req.input_ids_list) for req in requests
        )

        return GenerationContext(
            batch_input_ids=batch_input_ids,
            intermediate_tensors=None,
            query_start_loc=query_start_loc,
            batch_position_ids=batch_position_ids,
            block_tables=block_tables,
            context_lens=context_lens,
            slot_mapping=slot_mapping,
            max_seqlen_q=max_seqlen_q,
            generation_configs=[req.generation_config for req in requests],
            tokens_generated=[len(req.generated_tokens) for req in requests],
            is_decode_only=is_decode_only,
        )

    def _prepare_input_buffers_from_gen_ctx(self, gen_ctx: GenerationContext) -> None:
        buffers = self.input_buffers
        num_seqs = gen_ctx.context_lens.size(0)
        total_tokens = gen_ctx.batch_position_ids.size(1)

        buffers.slot_mapping.gpu.fill_(-1)
        buffers.query_start_loc.gpu.fill_(total_tokens)

        buffers.block_tables.gpu[:num_seqs, :].copy_(gen_ctx.block_tables)
        buffers.context_lens.gpu[:num_seqs].copy_(gen_ctx.context_lens)
        buffers.slot_mapping.gpu[:total_tokens].copy_(gen_ctx.slot_mapping)
        buffers.query_start_loc.gpu[: num_seqs + 1].copy_(gen_ctx.query_start_loc)

        num_tokens = gen_ctx.batch_position_ids.shape[1]
        if gen_ctx.batch_input_ids is not None:
            buffers.input_ids.gpu[:, :num_tokens].copy_(gen_ctx.batch_input_ids)
        buffers.position_ids.gpu[:, :num_tokens].copy_(gen_ctx.batch_position_ids)
        if gen_ctx.intermediate_tensors is not None:
            buffers.intermediate_tensors.gpu[:, :num_tokens, :].copy_(gen_ctx.intermediate_tensors)

        block_tables = buffers.block_tables.gpu[:num_seqs, :]
        slot_mapping = buffers.slot_mapping.gpu[:total_tokens]
        context_lens = buffers.context_lens.gpu[:num_seqs]
        query_start_loc = buffers.query_start_loc.gpu[: num_seqs + 1]

        set_kv_cache_context(
            is_paged_attn=True,
            key_cache=self.cache_store.key_cache,
            value_cache=self.cache_store.value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            max_seqlen_q=gen_ctx.max_seqlen_q,
        )

    @torch.inference_mode()
    def _forward(self, gen_ctx: GenerationContext) -> Any:
        if self._cudagraph_mgr and gen_ctx.is_decode_only:
            if gen_ctx.batch_input_ids is not None:
                input_size = gen_ctx.batch_input_ids.size(1)
            else:
                input_size = gen_ctx.intermediate_tensors.size(1)

            capture_size = self._cudagraph_mgr.select_capture_size(input_size)
            if capture_size is not None:
                # cudagraph forward does not read inputs from gen_ctx but from preallocated input buffers
                self._prepare_input_buffers_from_gen_ctx(gen_ctx)
                return self._forward_cudagraph(capture_size)

        return self._forward_eager(gen_ctx)

    @torch.inference_mode()
    def _forward_eager(self, gen_ctx: GenerationContext) -> Any:
        # TODO: we can extend scope name with better context such as new_req_num and cached_req_num
        with torch.cuda.nvtx.range("model_forward_eager"):
            outputs = self.model.forward(
                input_ids=gen_ctx.batch_input_ids,
                intermediate_tensors=gen_ctx.intermediate_tensors,
                position_ids=gen_ctx.batch_position_ids,
                **self.forward_kwargs,
            )

        return outputs

    def _forward_cudagraph(self, capture_size: int) -> Any:
        with torch.cuda.nvtx.range("model_forward_cudagraph"):
            outputs = self._cudagraph_mgr.replay(capture_size)

        return outputs

    @proton.cpu_timed_scope("_select_tokens")
    @torch.inference_mode()
    def _select_tokens(
        self,
        gen_configs: Sequence[GenerationConfig],
        tokens_generated: Sequence[int],
        outputs: ModelOutput,
        gen_ctx: GenerationContext,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        # Collect indices for all requests and slice logits once to avoid per-token LM head work
        # The selected token is always the last token in each request slice. For decode
        # requests the slice length is 1, so this is also the first token.
        last_token_positions_tensor = (gen_ctx.query_start_loc[1:] - 1).to(dtype=torch.long)
        batched_logits = outputs.logits[0, last_token_positions_tensor, :]

        # Use batch sampler when sampling params are uniform across the batch.
        sample_ref = gen_configs[0]
        uniform_sampling = all(
            (
                gen_config.do_sample == sample_ref.do_sample
                and gen_config.temperature == sample_ref.temperature
                and gen_config.top_k == sample_ref.top_k
                and gen_config.top_p == sample_ref.top_p
            )
            for gen_config in gen_configs
        )
        assert uniform_sampling
        batch_tokens = self.sampler.batch_sample_token(batched_logits, gen_configs)

        if sample_ref.output_logits:
            logits = batched_logits.detach()
        else:
            logits = torch.empty(0, device=self.device)

        # Compute logprobs in batch for the subset of requests that asked for scores
        score_indices = [idx for idx, gen_config in enumerate(gen_configs) if gen_config.output_scores]
        if score_indices:
            # we have cpu sync here
            score_index_tensor = torch.tensor(score_indices, device="cpu", dtype=torch.long)
            score_index_tensor = score_index_tensor.to(self.device, non_blocking=True)
            tokens_for_scores = batch_tokens.index_select(0, score_index_tensor)
            logits_for_scores = batched_logits.index_select(0, score_index_tensor)
            if self.use_fp32_logits:
                logits_for_scores = logits_for_scores.to(torch.float32)

            logprobs_batch = logprobs_from_logits_flash_attn(
                logits_for_scores, tokens_for_scores, inplace_backward=False
            )
            # log_probs here are fp32 if use_fp32_logits is used
            logprobs = logprobs_batch.reshape(-1).to(self.device)
            if self.use_fp32_logits:
                assert logprobs.dtype == torch.float32
        else:
            logprobs = torch.empty(0, device=self.device)

        token_ids = batch_tokens.to(self.device)
        return token_ids, logits, logprobs
