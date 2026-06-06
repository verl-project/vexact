#!/usr/bin/env python3
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

"""
Logits Verification Script

This script loads saved inference data and verifies the logits by running
a standard HuggingFace batch inference with all requests in one batch.

It compares the logits from the saved data against fresh inference to verify
correctness and determinism.

Usage:
    python verify_logits.py --model_path /path/to/model --data_dir inference_outputs

    # With custom tolerance
    python verify_logits.py --model_path gpt2 --data_dir inference_outputs --rtol 1e-4 --atol 1e-5
"""

import argparse
import json
import logging
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward as _transformers_flash_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vexact.batch_invariant_ops import flash_attention_forward as flash_attention_forward_impl
from vexact.batch_invariant_ops import flash_attention_forward_cute as flash_attention_forward_cute_impl
from vexact.batch_invariant_ops import flex_attention_forward
from vexact.batch_invariant_ops import triton_flash_attention_forward as triton_flash_attention_forward_impl
from vexact.config import PPInfo
from vexact.inferencer.model_loader import ModelCreator
from vexact.models.register import register_models as _register_models
from vexact.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

_register_models()
# Register two invariant attention implementations
ALL_ATTENTION_FUNCTIONS["flex"] = flex_attention_forward
ALL_ATTENTION_FUNCTIONS["fa-invariant"] = flash_attention_forward_impl
ALL_ATTENTION_FUNCTIONS["fa-invariant-cute"] = flash_attention_forward_cute_impl
ALL_ATTENTION_FUNCTIONS["triton-invariant"] = triton_flash_attention_forward_impl


def _patch_lazy_imports_for_fa4():
    """
    Transformers now cannot properly support flash_attention_4.
    For v4.57.3, we have to patch like this.
    Monkey-patch transformers' _lazy_imports to handle "flash_attention_4".

    On transformers v4.57.3, _lazy_imports has no built-in branch for FA4 — the string
    "flash_attention_4" falls through to the getattr() kernels-fallback which fails on
    a plain string. We intercept it and load flash_attn.cute locally, following the same
    approach as VeOmni's flash_attn/__init__.py.
    """
    import transformers.modeling_flash_attention_utils as fa_utils

    _original_lazy_imports = fa_utils._lazy_imports

    def _patched_lazy_imports(implementation):
        if implementation == "flash_attention_4":
            from types import SimpleNamespace

            from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

            # Pass the kernel as a SimpleNamespace so the getattr() fallback resolves it
            return _original_lazy_imports(
                SimpleNamespace(
                    flash_attn_func=flash_attn_func,
                    flash_attn_varlen_func=flash_attn_varlen_func,
                )
            )
        return _original_lazy_imports(implementation)

    fa_utils._lazy_imports = _patched_lazy_imports


_patch_lazy_imports_for_fa4()


def _fa4_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    dropout: float = 0.0,
    scaling=None,
    sliding_window=None,
    softcap=None,
    **kwargs,
):
    """
    Flash Attention 4 (flash_attn.cute) forward registered in ALL_ATTENTION_FUNCTIONS.

    On transformers v4.57.3, we pass the FA4 kernel as a SimpleNamespace object via the
    implementation= kwarg so _lazy_imports resolves it via the getattr() fallback.
    See VeOmni's flash_attn/__init__.py for reference.
    """
    from types import SimpleNamespace

    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

    seq_len = query.shape[2]

    # FA uses non-transposed inputs: (B, H, S, D) -> (B, S, H, D)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = module.is_causal

    fa4_kernel = SimpleNamespace(
        flash_attn_func=flash_attn_func,
        flash_attn_varlen_func=flash_attn_varlen_func,
    )

    attn_output = _transformers_flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=False,
        target_dtype=target_dtype,
        implementation=fa4_kernel,
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        **kwargs,
    )

    return attn_output, None


ALL_ATTENTION_FUNCTIONS["flash_attention_4"] = _fa4_attention_forward


def load_saved_data(data_dir: str, logger) -> tuple[dict, dict, dict, dict]:
    """
    Load saved inference data.

    Args:
        data_dir: Directory containing saved data
        logger: Logger instance

    Returns:
        Tuple of (all_logits, all_logprobs, all_token_ids, metadata)
    """
    logger.info(f"Loading saved data from {data_dir}")

    logits_path = os.path.join(data_dir, "all_logits.pt")
    logprobs_path = os.path.join(data_dir, "all_logprobs.pt")
    token_ids_path = os.path.join(data_dir, "all_token_ids_list.json")
    metadata_path = os.path.join(data_dir, "metadata.json")

    # Check if all files exist
    for path in [logits_path, logprobs_path, token_ids_path, metadata_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    all_logits = torch.load(logits_path)
    logger.info(f"loaded all_logits shapes: { {k: len(v) for k, v in all_logits.items()} }")
    all_logprobs = torch.load(logprobs_path)
    logger.info(f"loaded all_logprobs shapes: { {k: len(v) for k, v in all_logprobs.items()} }")
    with open(token_ids_path) as f:
        all_token_ids = json.load(f)

    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.info(f"Loaded data for {len(all_token_ids)} requests")

    return all_logits, all_logprobs, all_token_ids, metadata


def _reshape_logprobs(log_probs: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Ensure log_probs matches [batch, seq] shape."""
    if log_probs.dim() == 2 and list(log_probs.shape) == list(target_shape):
        return log_probs
    numel = log_probs.numel()
    expected = 1
    for dim in target_shape:
        expected *= dim
    if numel != expected:
        raise RuntimeError(f"Cannot reshape log_probs of shape {tuple(log_probs.shape)} into {tuple(target_shape)}")
    return log_probs.reshape(*target_shape)


def load_model_and_tokenizer(
    model_path: str,
    device: torch.device,
    logger,
    attn_impl: str = "eager",
    use_remove_padding: bool = True,
    use_fused_lce: bool = True,
) -> tuple:
    """
    Load HuggingFace model and tokenizer.

    Args:
        model_path: Path to the model or model name
        device: Device to load the model on
        logger: Logger instance
        attn_impl: Attention implementation to use
        use_fused_lce: enable VeRL monkey patch with torch backend to use fused LCE kernel.

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer from: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto" if torch.cuda.is_available() else "cpu",
        "trust_remote_code": True,
        "_attn_implementation": attn_impl,
    }
    config = AutoConfig.from_pretrained(model_path, **model_kwargs)
    if config.model_type in ("qwen3_moe", "deepseek_v3"):
        # Use ModelCreator to load with vexact patches (fused experts, patched attention).
        # This avoids issues with custom auto_map model classes rejecting non-standard attn_implementation.
        logger.info("Loading model using ModelCreator (from_config + custom weights)")
        TorchMemorySaverAdapter.create(enable=False)
        pp_info = PPInfo(pp_rank=0, pp_size=1)
        model = ModelCreator(config, model_path=model_path, device=device, pp_info=pp_info).create_model()
    else:
        logger.info("Loading model using AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if use_fused_lce:
        from verl.models.transformers.monkey_patch import apply_monkey_patch

        logger.info(f"Applying Verl Triton forward patch with use_remove_padding={use_remove_padding}")
        apply_monkey_patch(
            model, use_remove_padding=use_remove_padding, use_fused_kernels=True, fused_kernels_backend="torch"
        )
    else:
        logger.info("Skipping Verl Triton forward patch")

    logger.info("Model loaded successfully")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


def prepare_batch_inputs(all_token_ids: dict, metadata: dict, tokenizer, device: torch.device, logger) -> tuple:
    """
    Prepare batched inputs for inference.

    Args:
        all_token_ids: Dictionary mapping request_id to token tensors
        metadata: Metadata dictionary
        tokenizer: Tokenizer instance
        device: Device for tensors
        logger: Logger instance

    Returns:
        Tuple of (input_ids, attention_mask, request_ids, prompt_lens)
    """
    logger.info("Preparing batch inputs...")

    request_ids = list(all_token_ids.keys())
    token_sequences = [all_token_ids[req_id] for req_id in request_ids]
    prompt_lens = [metadata[req_id]["prompt_len"] for req_id in request_ids]

    # Find max length
    max_length = max(len(seq) for seq in token_sequences)

    logger.info(f"Batch size: {len(request_ids)}")
    logger.info(f"Max sequence length: {max_length}")

    # Right pad all sequences
    padded_sequences = []
    attention_masks = []

    for seq in token_sequences:
        seq_len = len(seq)
        padding_len = max_length - seq_len

        # Right pad with pad_token_id
        padded_seq = torch.cat(
            [
                torch.tensor(seq, dtype=torch.long),
                torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long),
            ]
        )

        # Attention mask: 1 for real tokens, 0 for padding
        attn_mask = torch.cat(
            [
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long),
            ]
        )

        padded_sequences.append(padded_seq)
        attention_masks.append(attn_mask)

    # Stack into batch tensors
    input_ids = torch.stack(padded_sequences).to(device)
    attention_mask = torch.stack(attention_masks).to(device)
    # position_ids = attention_mask.cumsum(dim=1) * attention_mask - 1
    print(f"{attention_mask=}, all one? {torch.all(attention_mask == 1)}")
    print(input_ids[0])
    print(tokenizer.decode(input_ids[0], skip_special_tokens=False))

    logger.info(f"Input IDs shape: {input_ids.shape}")
    logger.info(f"Attention mask shape: {attention_mask.shape}")

    return input_ids, attention_mask, request_ids, prompt_lens, None  # position ids


def prepare_batch_inputs_remove_padding(
    all_token_ids: dict, metadata: dict, tokenizer, device: torch.device, logger
) -> tuple:
    """
    Prepare batched inputs for inference using remove-padding approach.
    All sequences are concatenated without padding for efficiency.

    Args:
        all_token_ids: Dictionary mapping request_id to token tensors
        metadata: Metadata dictionary
        tokenizer: Tokenizer instance
        device: Device for tensors
        logger: Logger instance

    Returns:
        Tuple of (input_ids, attention_mask, position_ids, request_ids,
                  prompt_lens, cu_seqlens, max_seqlen)
    """
    logger.info("Preparing batch inputs (remove padding)...")

    request_ids = list(all_token_ids.keys())
    token_sequences = [torch.tensor(all_token_ids[req_id]) for req_id in request_ids]
    prompt_lens = [metadata[req_id]["prompt_len"] for req_id in request_ids]

    batch_size = len(request_ids)
    seq_lens = [len(seq) for seq in token_sequences]
    max_seqlen = max(seq_lens)
    total_tokens = sum(seq_lens)

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Total tokens (no padding): {total_tokens}")
    logger.info(f"Max sequence length: {max_seqlen}")
    logger.info(f"Sequence lengths: {seq_lens}")

    # Concatenate all sequences
    input_ids = torch.cat(token_sequences, dim=0).unsqueeze(0).to(device)  # [1, total_tokens]

    # Create attention mask (all 1s since no padding)
    attention_mask = torch.ones(1, total_tokens, dtype=torch.long, device=device)

    # Create position_ids for each sequence
    position_ids_list = []
    for seq_len in seq_lens:
        position_ids_list.append(torch.arange(seq_len, dtype=torch.long))
    position_ids = torch.cat(position_ids_list, dim=0).unsqueeze(0).to(device)  # [1, total_tokens]

    # Create cumulative sequence lengths for indexing
    # cu_seqlens[i] is the start position of sequence i in the concatenated tensor
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).numpy()),
        dtype=torch.int32,
        device=device,
    )

    logger.info(f"Input IDs shape: {input_ids.shape}")
    logger.info(f"Attention mask shape: {attention_mask.shape}")
    logger.info(f"Position IDs shape: {position_ids.shape}")
    logger.info(f"Cumulative sequence lengths: {cu_seqlens.tolist()}")

    # Return additional metadata needed for extraction
    return (
        input_ids,
        attention_mask,
        position_ids,
        request_ids,
        prompt_lens,
        cu_seqlens,
        max_seqlen,
    )


def extract_generated_logits_remove_padding(
    batch_logits: torch.Tensor,
    request_ids: list[str],
    metadata: dict,
    prompt_lens: list[int],
    cu_seqlens: torch.Tensor,
    logger,
) -> dict[str, list[torch.Tensor]]:
    """
    Extract logits for generated tokens from remove-padding batch output.

    In remove-padding format, batch_logits has shape [1, total_tokens, vocab_size]
    where all sequences are concatenated. We use cu_seqlens to identify boundaries.

    Args:
        batch_logits: Concatenated batch logits [1, total_tokens, vocab_size]
        request_ids: List of request IDs
        metadata: Metadata dictionary
        prompt_lens: List of prompt lengths for each request
        cu_seqlens: Cumulative sequence lengths [batch_size + 1]
        logger: Logger instance

    Returns:
        Dictionary mapping request_id to list of logit tensors for generated tokens
    """
    logger.info("Extracting generated token logits (remove padding)...")

    extracted_logits = {}

    # Squeeze batch dimension since we know it's 1
    batch_logits = batch_logits.squeeze(0)  # [total_tokens, vocab_size]

    for i, req_id in enumerate(request_ids):
        prompt_len = prompt_lens[i]
        response_len = metadata[req_id]["response_len"]

        # Get start and end positions in concatenated sequence
        seq_start = cu_seqlens[i].item()
        seq_end = cu_seqlens[i + 1].item()

        # Extract this sequence's logits
        seq_logits = batch_logits[seq_start:seq_end, :]  # [seq_len, vocab_size]

        # Extract logits for generated tokens
        # Logits at position j predict token at position j+1
        # So to predict generated tokens (positions prompt_len to prompt_len+response_len-1),
        # we need logits from positions prompt_len-1 to prompt_len+response_len-2
        logits_list = []

        for token_pos in range(response_len):
            # Position in this sequence that predicts this generated token
            logit_pos = prompt_len + token_pos - 1

            # Extract logits: [1, 1, vocab_size] to match original format
            token_logits = seq_logits[logit_pos : logit_pos + 1, :].unsqueeze(0)
            logits_list.append(token_logits)

        extracted_logits[req_id] = logits_list
        logger.debug(
            f"Extracted {len(logits_list)} logit tensors for request {req_id} (seq range: {seq_start}-{seq_end})"
        )

    return extracted_logits


def compute_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute the total gradient norm of model parameters.

    Args:
        parameters: Model parameters (iterable)
        norm_type: Type of norm to use (default: 2.0 for L2 norm)

    Returns:
        Total gradient norm as a tensor
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return torch.tensor(0.0)

    device = grads[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
        norm_type,
    )
    return total_norm


def run_batch_inference(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    enable_batch_invariant: bool,
    logger,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Run batch inference to get logits, then run backward pass to check gradients.

    Args:
        model: HuggingFace model
        input_ids: Batched input token IDs
        attention_mask: Attention mask
        enable_batch_invariant: Whether to enable batch invariant mode
        logger: Logger instance

    Returns:
        Tuple - (logits [batch_size, seq_len, vocab_size], logprobs [batch_size, seq_len] or None)
    """
    logger.info(f"Running batch inference (batch_invariant={'enabled' if enable_batch_invariant else 'disabled'})...")

    model.train()
    if enable_batch_invariant:
        from vexact.batch_invariant_ops import set_batch_invariant_mode

        logger.info("Using batch invariant mode for inference")
        logger.info(
            "Model forward in use: %s.%s",
            model.__class__.forward.__module__,
            getattr(model.__class__.forward, "__name__", "<no_name>"),
        )
        # Run in train mode to enable gradient computation
        with set_batch_invariant_mode(True):
            if cu_seqlens is None or max_seqlen is None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                    num_splits=1,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=max_seqlen,
                    max_length_k=max_seqlen,
                    use_cache=False,
                    return_dict=True,
                    num_splits=1,
                )
    else:
        raise NotImplementedError

    logger.info("Model output type: %s, has log_probs: %s", type(outputs), hasattr(outputs, "log_probs"))
    logits = outputs.logits
    log_probs = getattr(outputs, "log_probs", None)
    if log_probs is not None:
        log_probs = _reshape_logprobs(log_probs, input_ids.shape)
    else:
        logger.info("Model outputs did not include log_probs; skipping logprob comparison.")

    logger.info(f"Generated logits shape: {logits.shape if logits is not None else None}")
    logger.info(f"Generated logprobs shape: {log_probs.shape if log_probs is not None else None}")

    # Run backward pass and check grad_norm
    if log_probs is not None:
        # Simple loss: negative mean of log_probs
        # Mask out padding positions using attention_mask
        masked_log_probs = log_probs * attention_mask
        loss = -masked_log_probs.sum() / attention_mask.sum()

        logger.info(f"Computed loss from log_probs: {loss.item():.6f}")

        # Zero gradients before backward
        model.zero_grad()

        # Backward pass
        loss.backward()

        # Compute total gradient norm
        grad_norm = compute_grad_norm(model.parameters())
        logger.info(f"Total gradient norm: {grad_norm.item():.6e}")

        assert grad_norm.item() != 0.0, " grad_norm should not be zero. Please debug the fused lce computation."

    else:
        logger.warning("Cannot run backward: log_probs is None")

    return logits, log_probs


def extract_generated_logits(
    batch_logits: torch.Tensor,
    request_ids: list[str],
    metadata: dict,
    prompt_lens: list[int],
    logger,
) -> dict[str, list[torch.Tensor]]:
    """
    Extract logits for generated tokens only (not prompt tokens).

    Args:
        batch_logits: Full batch logits [batch_size, seq_len, vocab_size]
        request_ids: List of request IDs
        metadata: Metadata dictionary
        prompt_lens: List of prompt lengths for each request
        logger: Logger instance

    Returns:
        Dictionary mapping request_id to list of logit tensors for generated tokens
    """
    logger.info("Extracting generated token logits...")

    extracted_logits = {}

    for i, req_id in enumerate(request_ids):
        prompt_len = prompt_lens[i]
        response_len = metadata[req_id]["response_len"]

        # Logits for generated tokens start at position prompt_len-1 (predict first generated token)
        # and go until prompt_len + response_len - 1
        logits_list = []

        for token_pos in range(response_len):
            # Position in sequence that predicts this generated token
            logit_pos = prompt_len + token_pos - 1

            # Extract logits: [1, 1, vocab_size]
            token_logits = batch_logits[i : i + 1, logit_pos : logit_pos + 1, :]
            logits_list.append(token_logits)

        extracted_logits[req_id] = logits_list
        logger.debug(f"Extracted {len(logits_list)} logit tensors for request {req_id}")

    return extracted_logits


def extract_generated_logprobs(
    batch_logprobs: torch.Tensor,
    request_ids: list[str],
    metadata: dict,
    prompt_lens: list[int],
    logger,
) -> dict[str, list[torch.Tensor]]:
    """
    Extract logprobs for generated tokens only (not prompt tokens).

    Args:
        batch_logprobs: Full batch logprobs [batch_size, seq_len]
        request_ids: List of request IDs
        metadata: Metadata dictionary
        prompt_lens: List of prompt lengths for each request
        logger: Logger instance

    Returns:
        Dictionary mapping request_id to list of logprob scalars for generated tokens
    """
    logger.info("Extracting generated token logprobs...")

    extracted_logprobs = {}

    for i, req_id in enumerate(request_ids):
        prompt_len = prompt_lens[i]
        response_len = metadata[req_id]["response_len"]

        # Logprobs for generated tokens start at position prompt_len-1 (predict first generated token)
        # and go until prompt_len + response_len - 1
        logprobs_list = []

        for token_pos in range(response_len):
            # Position in sequence that predicts this generated token
            logprob_pos = prompt_len + token_pos - 1

            # Extract logprob: scalar value
            token_logprob = batch_logprobs[i, logprob_pos : logprob_pos + 1]
            logprobs_list.append(token_logprob)

        extracted_logprobs[req_id] = logprobs_list
        logger.debug(f"Extracted {len(logprobs_list)} logprob values for request {req_id}")

    return extracted_logprobs


def extract_generated_logprobs_remove_padding(
    batch_logprobs: torch.Tensor,
    request_ids: list[str],
    metadata: dict,
    prompt_lens: list[int],
    cu_seqlens: torch.Tensor,
    logger,
) -> dict[str, list[torch.Tensor]]:
    """
    Extract logprobs for generated tokens from remove-padding batch output.

    Args:
        batch_logprobs: Concatenated batch logprobs [1, total_tokens]
        request_ids: List of request IDs
        metadata: Metadata dictionary
        prompt_lens: List of prompt lengths for each request
        cu_seqlens: Cumulative sequence lengths [batch_size + 1]
        logger: Logger instance

    Returns:
        Dictionary mapping request_id to list of logprob scalars for generated tokens
    """
    logger.info("Extracting generated token logprobs (remove padding)...")

    extracted_logprobs = {}

    # Squeeze batch dimension since we know it's 1
    batch_logprobs = batch_logprobs.squeeze(0)  # [total_tokens]

    for i, req_id in enumerate(request_ids):
        prompt_len = prompt_lens[i]
        response_len = metadata[req_id]["response_len"]

        # Get start and end positions in concatenated sequence
        seq_start = cu_seqlens[i].item()
        seq_end = cu_seqlens[i + 1].item()

        # Extract this sequence's logprobs
        seq_logprobs = batch_logprobs[seq_start:seq_end]  # [seq_len]

        # Extract logprobs for generated tokens
        logprobs_list = []

        for token_pos in range(response_len):
            # Position in this sequence that predicts this generated token
            logprob_pos = prompt_len + token_pos - 1

            # Extract logprob: scalar value
            token_logprob = seq_logprobs[logprob_pos : logprob_pos + 1]
            logprobs_list.append(token_logprob)

        extracted_logprobs[req_id] = logprobs_list
        logger.debug(
            f"Extracted {len(logprobs_list)} logprob values for request {req_id} (seq range: {seq_start}-{seq_end})"
        )

    return extracted_logprobs


def compare_logits(
    saved_logits: dict[str, list[torch.Tensor]],
    fresh_logits: dict[str, list[torch.Tensor]],
    request_ids: list[str],
    prompt_lens: list[int],
    rtol: float,
    atol: float,
    logger,
) -> dict:
    """
    Compare saved logits with freshly generated logits.

    Args:
        saved_logits: Saved logits from continuous batching
        fresh_logits: Freshly generated logits from batch inference
        request_ids: List of request IDs to compare
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        logger: Logger instance

    Returns:
        Dictionary with comparison results
    """
    logger.info("=" * 80)
    logger.info("COMPARING LOGITS")
    logger.info("=" * 80)

    results = {
        "total_requests": len(request_ids),
        "matching_requests": 0,
        "mismatching_requests": 0,
        "per_request_results": {},
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "total_tokens": 0,
        "matching_tokens": 0,
    }

    for i, req_id in enumerate(request_ids):
        saved = saved_logits[req_id]
        fresh = fresh_logits[req_id]
        # print(f"{saved=}")
        # print(f"{fresh=}")

        if len(saved) != len(fresh):
            logger.error(f"Request {req_id}: Length mismatch! Saved: {len(saved)}, Fresh: {len(fresh)}")
            results["mismatching_requests"] += 1
            results["per_request_results"][req_id] = {
                "match": False,
                "error": "Length mismatch",
            }
            continue

        # Compare each token's logits
        request_matches = True
        max_abs_diff_req = 0.0
        max_rel_diff_req = 0.0
        token_matches = 0

        for token_idx, (saved_logit, fresh_logit) in enumerate(zip(saved, fresh, strict=True)):
            # Convert to same device for comparison
            saved_logit = saved_logit.cpu()
            fresh_logit = fresh_logit.cpu()

            # Compute differences
            abs_diff = torch.abs(saved_logit - fresh_logit)
            max_abs_diff_token = abs_diff.max().item()

            # Relative difference
            rel_diff = abs_diff / (torch.abs(saved_logit) + 1e-8)
            max_rel_diff_token = rel_diff.max().item()

            # Update global maxes
            max_abs_diff_req = max(max_abs_diff_req, max_abs_diff_token)
            max_rel_diff_req = max(max_rel_diff_req, max_rel_diff_token)
            results["max_abs_diff"] = max(results["max_abs_diff"], max_abs_diff_token)
            results["max_rel_diff"] = max(results["max_rel_diff"], max_rel_diff_token)

            # Check if within tolerance
            matches = torch.allclose(saved_logit, fresh_logit, rtol=rtol, atol=atol)

            if matches:
                token_matches += 1
            else:
                request_matches = False
                logger.warning(
                    f"Request {req_id}, Token {token_idx}: Mismatch! "
                    f"Max abs diff: {max_abs_diff_token:.6e}, Max rel diff: {max_rel_diff_token:.6e}"
                )

        results["total_tokens"] += len(saved)
        results["matching_tokens"] += token_matches

        if request_matches:
            results["matching_requests"] += 1
            logger.info(
                f"✓ Request {req_id}: MATCH ({len(saved)} tokens) - "
                f"Max abs diff: {max_abs_diff_req:.6e}, Max rel diff: {max_rel_diff_req:.6e}"
            )
        else:
            results["mismatching_requests"] += 1
            logger.error(
                f"✗ Request {req_id} with prompt len {prompt_lens[i]}: "
                f"MISMATCH ({token_matches}/{len(saved)} tokens matched) - "
                f"Max abs diff: {max_abs_diff_req:.6e}, Max rel diff: {max_rel_diff_req:.6e}"
            )

        results["per_request_results"][req_id] = {
            "match": request_matches,
            "num_tokens": len(saved),
            "matching_tokens": token_matches,
            "max_abs_diff": max_abs_diff_req,
            "max_rel_diff": max_rel_diff_req,
        }

    return results


def compare_logprobs(
    saved_logprobs: dict[str, list[torch.Tensor]],
    fresh_logprobs: dict[str, list[torch.Tensor]],
    request_ids: list[str],
    rtol: float,
    atol: float,
    logger,
) -> dict:
    """
    Compare saved logprobs with freshly generated logprobs.

    Args:
        saved_logprobs: Saved logprobs from continuous batching
        fresh_logprobs: Freshly generated logprobs from batch inference
        request_ids: List of request IDs to compare
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        logger: Logger instance

    Returns:
        Dictionary with comparison results
    """
    logger.info("=" * 80)
    logger.info("COMPARING LOGPROBS")
    logger.info("=" * 80)

    results = {
        "total_requests": len(request_ids),
        "matching_requests": 0,
        "mismatching_requests": 0,
        "per_request_results": {},
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "total_tokens": 0,
        "matching_tokens": 0,
    }

    for req_id in request_ids:
        saved = saved_logprobs[req_id]
        fresh = fresh_logprobs[req_id]

        if len(saved) != len(fresh):
            logger.error(f"Request {req_id}: Length mismatch! Saved: {len(saved)}, Fresh: {len(fresh)}")
            results["mismatching_requests"] += 1
            results["per_request_results"][req_id] = {
                "match": False,
                "error": "Length mismatch",
            }
            continue

        # Compare each token's logprob
        request_matches = True
        max_abs_diff_req = 0.0
        max_rel_diff_req = 0.0
        token_matches = 0

        for token_idx, (saved_logprob, fresh_logprob) in enumerate(zip(saved, fresh, strict=True)):
            # Convert to tensors on CPU for comparison
            if isinstance(saved_logprob, torch.Tensor):
                saved_tensor = saved_logprob.detach().to(torch.float32).cpu()
            else:
                saved_tensor = torch.tensor(saved_logprob, dtype=torch.float32)

            if isinstance(fresh_logprob, torch.Tensor):
                fresh_tensor = fresh_logprob.detach().to(torch.float32).cpu()
            else:
                fresh_tensor = torch.tensor(fresh_logprob, dtype=torch.float32)

            # Compute differences
            abs_diff = torch.abs(saved_tensor - fresh_tensor)
            max_abs_diff_token = abs_diff.max().item()

            # Relative difference
            rel_diff = abs_diff / (torch.abs(saved_tensor) + 1e-8)
            max_rel_diff_token = rel_diff.max().item()

            # Update global maxes
            max_abs_diff_req = max(max_abs_diff_req, max_abs_diff_token)
            max_rel_diff_req = max(max_rel_diff_req, max_rel_diff_token)
            results["max_abs_diff"] = max(results["max_abs_diff"], max_abs_diff_token)
            results["max_rel_diff"] = max(results["max_rel_diff"], max_rel_diff_token)

            # Check if within tolerance
            matches = torch.allclose(saved_tensor, fresh_tensor, rtol=rtol, atol=atol)

            if matches:
                token_matches += 1
            else:
                request_matches = False
                logger.warning(
                    f"Request {req_id}, Token {token_idx}: Mismatch! "
                    f"Saved: {saved_tensor.item():.6f}, Fresh: {fresh_tensor.item():.6f}, "
                    f"Abs diff: {max_abs_diff_token:.6e}, Rel diff: {max_rel_diff_token:.6e}"
                )

        results["total_tokens"] += len(saved)
        results["matching_tokens"] += token_matches

        if request_matches:
            results["matching_requests"] += 1
            logger.info(
                f"✓ Request {req_id}: MATCH ({len(saved)} tokens) - "
                f"Max abs diff: {max_abs_diff_req:.6e}, Max rel diff: {max_rel_diff_req:.6e}"
            )
        else:
            results["mismatching_requests"] += 1
            logger.error(
                f"✗ Request {req_id}: MISMATCH ({token_matches}/{len(saved)} tokens matched) - "
                f"Max abs diff: {max_abs_diff_req:.6e}, Max rel diff: {max_rel_diff_req:.6e}"
            )

        results["per_request_results"][req_id] = {
            "match": request_matches,
            "num_tokens": len(saved),
            "matching_tokens": token_matches,
            "max_abs_diff": max_abs_diff_req,
            "max_rel_diff": max_rel_diff_req,
        }

    return results


def print_summary(logits_results: dict, logger, logprobs_results: dict = None) -> tuple[bool | None, bool | None]:
    """Print summary of comparison results, independently for logits and logprobs."""
    logger.info("=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)

    assert logits_results is not None or logprobs_results is not None, "at least need to have something to compare!"

    logits_match: bool | None = None
    logprobs_match: bool | None = None

    # Logits results
    if logits_results is not None:
        logger.info("LOGITS COMPARISON:")
        total = logits_results["total_requests"]
        matching = logits_results["matching_requests"]
        mismatching = logits_results["mismatching_requests"]

        logger.info(f"  Total requests: {total}")
        logger.info(f"  Matching requests: {matching} ({100 * matching / total:.1f}%)")
        logger.info(f"  Mismatching requests: {mismatching} ({100 * mismatching / total:.1f}%)")
        logger.info("")
        logger.info(f"  Total tokens: {logits_results['total_tokens']}")
        logger.info(
            f"  Matching tokens: {logits_results['matching_tokens']} "
            f"({100 * logits_results['matching_tokens'] / logits_results['total_tokens']:.1f}%)"
        )
        logger.info("")
        logger.info(f"  Maximum absolute difference: {logits_results['max_abs_diff']:.6e}")
        logger.info(f"  Maximum relative difference: {logits_results['max_rel_diff']:.6e}")
        logits_match = matching == total
    else:
        logger.info("Logits comparison skipped (logits_results is None).")

    # Logprobs results if available
    if logprobs_results is not None:
        logger.info("")
        logger.info("LOGPROBS COMPARISON:")
        lp_total = logprobs_results["total_requests"]
        lp_matching = logprobs_results["matching_requests"]
        lp_mismatching = logprobs_results["mismatching_requests"]

        logger.info(f"  Total requests: {lp_total}")
        logger.info(f"  Matching requests: {lp_matching} ({100 * lp_matching / lp_total:.1f}%)")
        logger.info(f"  Mismatching requests: {lp_mismatching} ({100 * lp_mismatching / lp_total:.1f}%)")
        logger.info("")
        logger.info(f"  Total tokens: {logprobs_results['total_tokens']}")
        logger.info(
            f"  Matching tokens: {logprobs_results['matching_tokens']} "
            f"({100 * logprobs_results['matching_tokens'] / logprobs_results['total_tokens']:.1f}%)"
        )
        logger.info("")
        logger.info(f"  Maximum absolute difference: {logprobs_results['max_abs_diff']:.6e}")
        logger.info(f"  Maximum relative difference: {logprobs_results['max_rel_diff']:.6e}")
        logprobs_match = lp_matching == lp_total
    else:
        logger.info("Logprobs comparison skipped (logprobs_results is None).")

    logger.info("=" * 80)
    return logits_match, logprobs_match

    # Overall verification status
    all_match = False
    if logits_results is not None:
        all_match = matching == total
    if logprobs_results is not None:
        all_match = all_match and (logprobs_results["matching_requests"] == logprobs_results["total_requests"])

    if all_match:
        logger.info("✓✓✓ ALL COMPARISONS MATCH! Verification successful! ✓✓✓")
    else:
        if logits_results is not None and matching != total:
            logger.error("✗✗✗ LOGITS DO NOT MATCH! Verification failed! ✗✗✗")
        if logprobs_results is not None and logprobs_results["matching_requests"] != logprobs_results["total_requests"]:
            logger.error("✗✗✗ LOGPROBS DO NOT MATCH! Verification failed! ✗✗✗")
        exit(1)

    logger.info("=" * 80)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify saved logits by comparing with fresh batch inference")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model or HuggingFace model name",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="inference_outputs",
        help="Directory containing saved inference data (default: inference_outputs)",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for logits comparison (default: 1e-5)",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for logits comparison (default: 1e-6)",
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default="verify_logits.log",
        help="Log file name (default: verify_logits.log)",
    )

    parser.add_argument(
        "--attn_impl",
        type=str,
        default="eager",
        help="Attention implementation to use (default: eager)",
    )

    parser.add_argument(
        "--enable_batch_invariant",
        action="store_true",
        help="Enable batch invariant mode for deterministic inference",
    )

    parser.add_argument(
        "--use_remove_padding",
        action="store_true",
        help="Use remove-padding batch format (no padding between sequences)",
    )

    parser.add_argument(
        "--use_fused_lce",
        action="store_true",
        help="Enable VeRL FusedLienarForPPO forward patch",
    )

    return parser.parse_args()


def main():
    """Main verification function."""
    args = parse_arguments()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        logger.info("=" * 80)
        logger.info("LOGITS VERIFICATION SCRIPT")
        logger.info("=" * 80)
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Relative tolerance: {args.rtol}")
        logger.info(f"Absolute tolerance: {args.atol}")
        logger.info(f"Attention implementation: {args.attn_impl}")
        logger.info(f"Batch invariant mode: {'enabled' if args.enable_batch_invariant else 'disabled'}")
        logger.info("Qwen3-MoE Triton patch: %s", "enabled" if args.use_fused_lce else "disabled")
        logger.info("=" * 80)

        if args.attn_impl == "triton-invariant" and not args.use_remove_padding:
            raise ValueError("triton-invariant verification requires --use_remove_padding")

        # Load saved data
        saved_logits, saved_logprobs, all_token_ids, metadata = load_saved_data(args.data_dir, logger)

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            args.model_path,
            device,
            logger,
            args.attn_impl,
            use_remove_padding=args.use_remove_padding,
            use_fused_lce=args.use_fused_lce,
        )

        if args.use_remove_padding:
            (
                input_ids,
                attention_mask,
                position_ids,
                request_ids,
                prompt_lens,
                cu_seqlens,
                max_seqlen,
            ) = prepare_batch_inputs_remove_padding(all_token_ids, metadata, tokenizer, device, logger)

            batch_logits, batch_logprobs = run_batch_inference(
                model,
                input_ids,
                attention_mask,
                position_ids,
                args.enable_batch_invariant,
                logger,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            # when using triton fused kernels, logits would not be materialized
            if batch_logits is not None:
                fresh_logits = extract_generated_logits_remove_padding(
                    batch_logits, request_ids, metadata, prompt_lens, cu_seqlens, logger
                )
            else:
                fresh_logits = None

            # Extract logprobs if available
            if batch_logprobs is not None:
                fresh_logprobs = extract_generated_logprobs_remove_padding(
                    batch_logprobs, request_ids, metadata, prompt_lens, cu_seqlens, logger
                )
            else:
                fresh_logprobs = None
        else:
            # Prepare batch inputs
            (
                input_ids,
                attention_mask,
                request_ids,
                prompt_lens,
                position_ids,
            ) = prepare_batch_inputs(all_token_ids, metadata, tokenizer, device, logger)

            # Run batch inference
            batch_logits, batch_logprobs = run_batch_inference(
                model,
                input_ids,
                attention_mask,
                position_ids,
                args.enable_batch_invariant,
                logger,
            )

            # Extract generated token logits
            # when using triton fused kernels, logits would not be materialized
            if batch_logits is not None:
                fresh_logits = extract_generated_logits(batch_logits, request_ids, metadata, prompt_lens, logger)
            else:
                fresh_logits = None

            # Extract logprobs if available
            if batch_logprobs is not None:
                fresh_logprobs = extract_generated_logprobs(batch_logprobs, request_ids, metadata, prompt_lens, logger)
            else:
                fresh_logprobs = None

        # Compare logits
        if batch_logits is not None and fresh_logits is not None:
            logits_results = compare_logits(
                saved_logits,
                fresh_logits,
                request_ids,
                prompt_lens,
                args.rtol,
                args.atol,
                logger,
            )
        else:
            logits_results = None

        # Compare logprobs if available
        logprobs_results = None
        if batch_logprobs is not None and fresh_logprobs is not None:
            logprobs_results = compare_logprobs(
                saved_logprobs,
                fresh_logprobs,
                request_ids,
                args.rtol,
                args.atol,
                logger,
            )

        logger.info(
            "Comparison inputs - batch_logits: %s, fresh_logits: %s, batch_logprobs: %s, fresh_logprobs: %s",
            "present" if batch_logits is not None else "None",
            "present" if (batch_logits is not None and fresh_logits is not None) else "None",
            "present" if batch_logprobs is not None else "None",
            "present" if (batch_logprobs is not None and fresh_logprobs is not None) else "None",
        )

        # Merge logprobs results with logits results when both are present
        # Only pass if both logits and logprobs match
        if (
            logits_results is not None
            and logprobs_results is not None
            and logprobs_results["matching_requests"] != logprobs_results["total_requests"]
        ):
            logits_results["matching_requests"] = 0
            logger.warning("Logprobs comparison failed - marking overall verification as failed")

        # Print summary
        logits_match, logprobs_match = print_summary(logits_results, logger, logprobs_results)
        logger.info(
            "Results availability - logits_results: %s, logprobs_results: %s",
            "present" if logits_results is not None else "None",
            "present" if logprobs_results is not None else "None",
        )

        # Return exit code based on results
        success = True
        if logits_match is False:
            success = False
        if logprobs_match is False:
            success = False
        logger.info(
            "Verification outcome - logits_match: %s, logprobs_match: %s, success: %s",
            logits_match,
            logprobs_match,
            success,
        )

        if success:
            logger.info("Verification completed successfully!")
            return 0
        else:
            logger.error("Verification failed!")
            return 1

    except Exception as e:
        logger.error(f"Verification failed with error: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
