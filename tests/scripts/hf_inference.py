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

import argparse
import gc
import json
import logging
import os
import random
import subprocess
import time

import numpy as np
import torch
from transformers import GenerationConfig
from transformers.utils import is_flash_attn_3_available

from vexact.config import CacheConfig, ModelConfig, ParallelConfig, ProfilerConfig, SchedulerConfig, VeXactConfig
from vexact.core.request import InferenceRequest

# from vexact.models.register import register_models as _register_models
from vexact.utils.subprocess_utils import get_sys_executable
from vexact.utils.tokenizer import load_tokenizer
from vexact.worker.driver_worker import DriverWorker


assert is_flash_attn_3_available()
# _register_models()


# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def compute_token_level_kl_divergence(logits1: list[torch.Tensor], logits2: list[torch.Tensor]) -> tuple:
    """
    Compute token-level KL divergence to identify where variance starts.

    Args:
        logits1: List of logit tensors from first request
        logits2: List of logit tensors from second request

    Returns:
        Tuple of (per_token_kl_list, average_kl, first_divergent_token)
    """
    if len(logits1) != len(logits2):
        # If sequences have different lengths, compute KL for the common prefix
        min_len = min(len(logits1), len(logits2))
        logits1 = logits1[:min_len]
        logits2 = logits2[:min_len]

    if len(logits1) == 0:
        return [], 0.0, -1

    kl_divergences = []
    first_divergent_token = -1

    for token_idx, (l1, l2) in enumerate(zip(logits1, logits2)):
        # Convert logits to probabilities
        p1 = torch.softmax(l1.flatten(), dim=-1)
        p2 = torch.softmax(l2.flatten(), dim=-1)

        # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        p1_safe = p1 + eps
        p2_safe = p2 + eps

        kl = torch.sum(p1_safe * torch.log(p1_safe / p2_safe))
        kl_value = kl.item()
        kl_divergences.append(kl_value)

        # Mark first token where divergence starts (threshold > 1e-6)
        if first_divergent_token == -1 and kl_value > 1e-6:
            first_divergent_token = token_idx

    avg_kl = np.mean(kl_divergences) if kl_divergences else 0.0
    return kl_divergences, avg_kl, first_divergent_token


def setup_logging(level: int = logging.INFO):
    """
    Set up basic logging configuration if not already configured.

    Args:
        level: Logging level (default: logging.INFO)
    """
    # Only configure if no handlers exist (i.e., basicConfig hasn't been called)
    if not logging.root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s")


def save_completed_requests_data(completed_requests: list[InferenceRequest], output_dir: str = "inference_outputs"):
    """
    Save logits, log probabilities, and token IDs for all completed requests.

    Args:
        completed_requests: List of completed InferenceRequest objects
        output_dir: Directory to save the output files

    Saves:
        - {output_dir}/all_logits.pt: Dictionary mapping request_id to list of logit tensors
        - {output_dir}/all_logprobs.pt: Dictionary mapping request_id to list of logprob tensors
        - {output_dir}/all_token_ids.pt: Dictionary mapping request_id to full token sequence (prompt + generated)
        - {output_dir}/metadata.json: Metadata about requests (prompts, generated text, lengths, etc.)
    """
    if not completed_requests:
        logger.warning("No completed requests to save")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data structures
    all_logits = {}
    all_logprobs = {}
    all_token_ids = {}
    metadata = {}

    for req in completed_requests:
        req_id = req.request_id

        # Store logits (already stored in req.generated_logits)
        if req.generated_logits:
            all_logits[req_id] = req.generated_logits

        # Store log probabilities (already computed during generation if output_scores=True)
        if req.generated_logprobs:
            all_logprobs[req_id] = req.generated_logprobs

        # Combine prompt token ids and generated token ids into one tensor
        all_token_ids[req_id] = req.input_ids_list + req.generated_tokens

        # Store metadata with lengths instead of concrete token IDs
        prompt_len = len(req.input_ids_list)
        response_len = len(req.generated_tokens)

        metadata[req_id] = {
            "generated_text": req.result,
            "prompt_len": prompt_len,
            "response_len": response_len,
            "total_tokens": prompt_len + response_len,
            "processing_time": req.processing_time,
            "status": req.status.value,
        }

    # Save logits
    logits_path = os.path.join(output_dir, "all_logits.pt")
    torch.save(all_logits, logits_path)
    logger.info(f"Saved logits for {len(all_logits)} requests to {logits_path}")

    # Save log probabilities
    logprobs_path = os.path.join(output_dir, "all_logprobs.pt")
    torch.save(all_logprobs, logprobs_path)
    logger.info(f"Saved log probabilities for {len(all_logprobs)} requests to {logprobs_path}")

    # Save token IDs
    token_ids_list_path = os.path.join(output_dir, "all_token_ids_list.json")
    with open(token_ids_list_path, "w") as f:
        json.dump(all_token_ids, f)
    logger.info(f"Saved token IDs for {len(all_token_ids)} requests to {token_ids_list_path}")

    # Save metadata as JSON
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    # Print summary statistics
    total_tokens = sum(len(logits) for logits in all_logits.values())
    avg_tokens = total_tokens / len(all_logits) if all_logits else 0
    total_prompt_tokens = sum(meta["prompt_len"] for meta in metadata.values())
    total_generated_tokens = sum(meta["response_len"] for meta in metadata.values())
    logger.info(
        f"Summary: {len(completed_requests)} requests, {total_tokens} generated tokens, "
        f"{avg_tokens:.1f} avg tokens per request"
    )
    logger.info(f"  Total prompt tokens: {total_prompt_tokens}, Total generated tokens: {total_generated_tokens}")


def parse_arguments():
    """
    Parse command line arguments for the inference script.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="HuggingFace Model Inference Script")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model or HuggingFace model name",
    )

    parser.add_argument("--pipeline_parallel_size", type=int, default=1)

    parser.add_argument("--prompt", type=str, help="Single prompt for text generation")

    parser.add_argument("--prompts", nargs="+", help="Multiple prompts for batch text generation")

    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length of generated text (default: 100)",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate (default: 50)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter (default: 0.9)",
    )

    parser.add_argument("--top_k", type=int, default=-1, help="Top-k sampling parameter (default: 50)")

    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_outputs",
        help="Directory to save logits and logprobs (default: inference_outputs)",
    )

    # Continuous batching arguments

    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=2048,
        help="Maximum batch size for continuous batching (default: 2048)",
    )

    parser.add_argument(
        "--simulate_requests",
        type=int,
        default=0,
        help="Simulate N concurrent requests for testing continuous batching (default: 0)",
    )

    parser.add_argument(
        "--simulate_requests_random_contents",
        action="store_true",
        help="Enable simulation of requests with identical prompts",
    )

    parser.add_argument(
        "--request_interval",
        type=float,
        default=0.05,
        help="Interval between simulated requests in seconds (default: 0.05)",
    )

    # Profiler arguments
    parser.add_argument(
        "--profile_backend",
        type=str,
        default=None,
        choices=["torch", "proton"],
        help="Profiler backend to use (None = disabled, default: None)",
    )

    parser.add_argument(
        "--profile_output",
        type=str,
        default=None,
        help="Output file for profiler trace (default: auto-generated)",
    )

    parser.add_argument(
        "--profile_delay_iterations",
        type=int,
        default=0,
        help="Number of steps to wait before starting profiler (default: 0)",
    )

    parser.add_argument(
        "--profile_max_iterations",
        type=int,
        default=0,
        help="Number of steps to profile (0 = until manually stopped, default: 0)",
    )
    parser.add_argument(
        "--profile_all_ranks",
        action="store_true",
        help="Enable profiling on all pipeline-parallel ranks (default: rank 0 only)",
    )

    parser.add_argument(
        "--enable_batch_invariant",
        action="store_true",
        help="Enable batch invariant operations for deterministic inference",
    )

    parser.add_argument(
        "--enable_memory_saver",
        action="store_true",
        help="Enable memory saver for GPU memory offload",
    )

    parser.add_argument(
        "--attn_impl",
        default="fa-invariant",
        help="Attention implementation to use (fa-variant or flex)",
    )

    parser.add_argument(
        "--enable_chunked_prefill",
        action="store_true",
        default=False,
        help="Enable chunked prefill for prefill stage (default: disabled)",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable cudagraph capture/replay and force eager model forward",
    )
    parser.add_argument(
        "--use_fp32_logits",
        action="store_true",
        help="Use fused linear cross entropy for logprob computation",
    )
    parser.add_argument(
        "--max_cache_blocks",
        type=int,
        default=None,
        help="Maximum number of KV cache blocks (default: auto from CacheConfig)",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        default=None,
        help="Page size for KV cache (number of tokens per page, default: auto from CacheConfig)",
    )

    args = parser.parse_args()

    # Validate prompts - only require prompts if not simulating requests
    if args.simulate_requests <= 0:
        parser.error("Either --simulate_requests has to be greater than 0.")

    return args


def run_continuous_batching(args, tokenizer, generation_config):
    """
    Run continuous batching inference mode.

    Args:
        args: Parsed command line arguments
        tokenizer: Loaded tokenizer
        generation_config: Generation configuration
    """
    logger.info("=" * 60)
    logger.info("STARTING CONTINUOUS BATCHING MODE")
    logger.info("=" * 60)
    logger.info(f"Max bached tokens: {args.max_num_batched_tokens}")
    logger.info(f"Batch invariant mode: {'enabled' if args.enable_batch_invariant else 'disabled'}")
    logger.info(f"Chunked prefill mode: {'enabled' if args.enable_chunked_prefill else 'disabled'}")

    # Log profiler configuration
    if args.profile_backend:
        logger.info(
            "Profiler enabled: backend=%s, output=%s, delay=%d, max_iters=%d, all_ranks=%s",
            args.profile_backend,
            args.profile_output,
            args.profile_delay_iterations,
            args.profile_max_iterations,
            args.profile_all_ranks,
        )
    else:
        logger.info("Profiler disabled")

    logger.info("=" * 60)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "8576")

    processes: list[subprocess.Popen] = []
    # spawn other pp rank as subprocesses
    for i in range(1, args.pipeline_parallel_size):
        worker_cmd = get_sys_executable() + [
            "-m",
            "vexact.worker.worker",
            "--pp-rank",
            str(i),
            "--pp-size",
            str(args.pipeline_parallel_size),
            "--model-path",
            args.model_path,
            "--attn-impl",
            args.attn_impl,
        ]
        if args.use_fp32_logits:
            worker_cmd.append("--use-fp32-logits")
        if args.enforce_eager:
            worker_cmd.append("--enforce-eager")
        if args.profile_backend:
            worker_cmd.extend(["--profile-backend", args.profile_backend])
            if args.profile_output:
                worker_cmd.extend(["--profile-output", args.profile_output])
            if args.profile_delay_iterations:
                worker_cmd.extend(["--profile-delay-iterations", str(args.profile_delay_iterations)])
            if args.profile_max_iterations:
                worker_cmd.extend(["--profile-max-iterations", str(args.profile_max_iterations)])
        if args.profile_all_ranks:
            worker_cmd.append("--profile-all-ranks")
        logger.info(f"Launching worker subprocess for PP rank {i}: {' '.join(worker_cmd)}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        proc = subprocess.Popen(worker_cmd, env=env)
        processes.append(proc)

    # Create VeXactConfig from args
    cache_kwargs = {}
    if args.max_cache_blocks is not None:
        cache_kwargs["max_cache_blocks"] = args.max_cache_blocks
    if args.page_size is not None:
        cache_kwargs["page_size"] = args.page_size

    vexact_config = VeXactConfig(
        model=ModelConfig(
            model_path=args.model_path,
            attn_impl=args.attn_impl,
            enable_batch_invariant=args.enable_batch_invariant,
            enable_memory_saver=args.enable_memory_saver,
            enforce_eager=args.enforce_eager,
            use_fp32_logits=args.use_fp32_logits,
        ),
        cache=CacheConfig(**cache_kwargs),
        parallel=ParallelConfig(
            pipeline_parallel_size=args.pipeline_parallel_size,
        ),
        scheduler=SchedulerConfig(
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_chunked_prefill=args.enable_chunked_prefill,
        ),
        profiler=ProfilerConfig(
            backend=args.profile_backend,
            delay_iterations=args.profile_delay_iterations,
            max_iterations=args.profile_max_iterations,
            output_path=args.profile_output,
            profile_all_ranks=args.profile_all_ranks,
        ),
    )
    logger.info(
        f"Resolved cudagraph config: enabled={not vexact_config.model.enforce_eager}, "
        f"max_size={vexact_config.scheduler.max_num_batched_tokens}"
    )

    # Initialize continuous batching engine
    engine = DriverWorker(config=vexact_config)

    # Start the engine
    engine.start()

    try:
        # We release the weight then recover with the load state dict interface.
        if args.enable_memory_saver:
            _old_params = {k: v.detach().clone() for k, v in engine.model.named_parameters()}
            engine.sleep()
            engine.wake_up()
            engine.load_state_dict(_old_params)

        assert args.simulate_requests > 0
        # Simulate concurrent requests for testing
        simulate_concurrent_requests(args, engine, tokenizer, generation_config)

    finally:
        # Stop the subprocess worker loops
        import signal

        for proc in processes:
            os.kill(proc.pid, signal.SIGKILL)
        # Stop the engine
        engine.stop()


EXAMPLE_PROMPTS = [
    "What is AI?",
    "Explain photosynthesis in one sentence.",
    "Who invented the telephone?",
    "Why is the sky blue?",
    "List three uses of machine learning in daily life.",
    "Describe the process of making bread from flour.",
    "How does gravity affect time according to Einstein?",
    "Summarize the causes of World War II in two lines.",
    "What are the key differences between RAM and ROM?",
    "Translate 'good morning' into French, Spanish, and Japanese.",
    "How does a blockchain ensure data integrity and security?",
    "What are the pros and cons of remote work for tech companies?",
    "Explain how vaccines help the immune system develop protection.",
    "Compare the political systems of the United States and China briefly.",
    "Describe the lifecycle of a butterfly in chronological order.",
    "What are common challenges when scaling a web application to millions of users?",
    "How do neural networks approximate complex nonlinear functions?",
    "Why do some metals conduct electricity better than others?",
    "Summarize the main argument of the book 'The Lean Startup'.",
    "If global temperatures rise by 2°C, what are the likely environmental consequences?",
]


def simulate_concurrent_requests(args, engine: DriverWorker, tokenizer, generation_config):
    """
    Simulate concurrent requests for testing continuous batching.

    Args:
        args: Parsed command line arguments
        engine: Continuous batching engine
        tokenizer: Tokenizer for encoding prompts
        generation_config: Generation configuration
    """
    import uuid

    logger.info(f"Simulating {args.simulate_requests} concurrent requests...")

    # Submit requests
    request_ids = []
    start_time = time.time()

    for i in range(args.simulate_requests):
        request_id = f"sim_req_{uuid.uuid4().hex[:8]}"
        # 'prompt = '"1+1=?"
        # prompt = "Tell me about artificial intelligence"
        prompt = "The old lighthouse keeper, who had spent nearly forty years watching over the rocky coastline and guiding ships safely through the treacherous waters during countless storms, finally decided on a misty autumn morning that it was time to retire and pass the responsibility to someone younger, someone with sharper eyes and steadier hands, though he knew he would deeply miss the solitary beauty of the crashing waves, the calls of the seabirds at dawn, the smell of salt air, and especially those quiet moments at sunset when the world seemed to pause and he felt truly at peace with his life's work.Retry"  # noqa: E501

        if args.simulate_requests_random_contents:
            prompt = random.choice(EXAMPLE_PROMPTS)

        # Tokenize the prompt
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

        request = InferenceRequest(
            request_id=request_id,
            input_ids_list=inputs["input_ids"],
            generation_config=generation_config,
        )

        engine.submit_request(request)
        request_ids.append(request_id)
        logger.info(f"Submitted request {i + 1}/{args.simulate_requests}: {request_id}")

        # Wait between requests
        if i < args.simulate_requests - 1:
            time.sleep(args.request_interval)

    submission_time = time.time() - start_time
    logger.info(f"All requests submitted in {submission_time:.3f}s")

    # Wait for all results
    logger.info("Waiting for all requests to complete...")
    completed = {}
    pending = set(request_ids)
    timeout_per_poll = 1.0
    total_timeout = 50.0 * len(request_ids)
    start_wait = time.time()

    while pending and (time.time() - start_wait) < total_timeout:
        for result in engine.poll_results(timeout=timeout_per_poll):
            if result.request_id in pending:
                if result.generated_tokens:
                    result.result = tokenizer.decode(result.generated_tokens, skip_special_tokens=True)
                completed[result.request_id] = result
                pending.discard(result.request_id)
                logger.info(f"Completed request {result.request_id} in {result.processing_time:.3f}s")

    for request_id in pending:
        logger.warning(f"Request {request_id} timed out")

    completed_requests = [completed[rid] for rid in request_ids if rid in completed]

    total_time = time.time() - start_time
    logger.info(f"All requests processed in {total_time:.3f}s")
    logger.info(f"Completed {len(completed_requests)}/{len(request_ids)} requests successfully")

    # Print sample results
    logger.info("\nSample Results:")
    for i, result in enumerate(completed_requests[:3]):  # Show first 3 results
        logger.info(f"Prompt: {result.input_ids_list}")
        logger.info(f"Generated: {result.result}")
        logger.info("-" * 40)

    logger.info("\nSaving logits and log probabilities to disk...")
    save_completed_requests_data(completed_requests=completed_requests, output_dir=args.output_dir)

    if args.simulate_requests_random_contents:
        logger.info(
            "\nSkipping KL divergence computation (simulate_requests_random_contents=True, "
            "results are non-deterministic between different requests.)"
        )
        return

    # Only compute KL divergence for deterministic generation (do_sample=False)
    # Sampling is non-deterministic, so KL divergence is not meaningful
    if generation_config.do_sample:
        logger.info(
            "\nSkipping KL divergence computation (do_sample=True, "
            "results are non-deterministic between consecutive requests.)"
        )
        return

    # Compute KL divergence between consecutive request pairs
    logger.info("\nComputing KL divergence between consecutive request pairs...")

    if len(completed_requests) > 1:
        kl_divergences = []

        for i in range(len(completed_requests) - 1):
            req1 = completed_requests[i]
            req2 = completed_requests[i + 1]

            if len(req1.generated_logits) > 0 and len(req2.generated_logits) > 0:
                # Use token-level KL divergence to identify where variance starts
                (
                    per_token_kl,
                    avg_kl,
                    first_divergent_token,
                ) = compute_token_level_kl_divergence(req1.generated_logits, req2.generated_logits)
                kl_divergences.append(avg_kl)

                # Debug-level detailed divergence information
                logger.debug(f"KL divergence between {req1.request_id} and {req2.request_id}: {avg_kl:.6f}")
                if first_divergent_token is not None:
                    logger.debug(f"  First divergent token at position: {first_divergent_token}")
                    kl_preview = [f"{kl:.6f}" for kl in per_token_kl[: min(5, len(per_token_kl))]]
                    preview_text = str(kl_preview) + ("..." if len(per_token_kl) > 5 else "")
                    logger.debug(f"  Token-level KL divergences: {preview_text}")
                else:
                    logger.debug(f"  All tokens identical (max KL: {max(per_token_kl) if per_token_kl else 0:.6f})")

        if kl_divergences:
            mean_kl = np.mean(kl_divergences)
            std_kl = np.std(kl_divergences)
            min_kl = np.min(kl_divergences)
            max_kl = np.max(kl_divergences)

            logger.info(f"\nKL Divergence Statistics (across {len(kl_divergences)} consecutive pairs):")
            logger.info(f"Mean KL divergence: {mean_kl:.6f}")
            logger.info(f"Std KL divergence: {std_kl:.6f}")
            logger.info(f"Min KL divergence: {min_kl:.6f}")
            logger.info(f"Max KL divergence: {max_kl:.6f}")
            if mean_kl > 0 or max_kl > 0:
                raise ValueError("KL not matched.")
        else:
            logger.info("No valid logits found for KL divergence computation")
            raise RuntimeError
    else:
        logger.info("Need at least 2 completed requests for KL divergence computation")
        raise RuntimeError


def main():
    """
    Main inference function.
    """
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging()

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    try:
        logger.info("=" * 60)
        logger.info("HUGGINGFACE MODEL INFERENCE")
        logger.info("=" * 60)
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Max length: {args.max_length}")
        logger.info(f"Max new tokens: {args.max_new_tokens}")
        logger.info(f"Temperature: {args.temperature}")
        logger.info(f"Top-p: {args.top_p}")
        logger.info(f"Top-k: {args.top_k}")
        logger.info(f"Do sample: {args.do_sample}")
        logger.info(f"attn_impl: {args.attn_impl}")
        logger.info("=" * 60)

        # Load tokenizer only (engine will load model)
        tokenizer = load_tokenizer(args.model_path)

        # Resolve eos_token_id from model config, which may differ from tokenizer
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        eos_token_id = getattr(model_config, "eos_token_id", tokenizer.eos_token_id)
        if eos_token_id != tokenizer.eos_token_id:
            logger.warning(
                f"Model config eos_token_id ({eos_token_id}) differs from "
                f"tokenizer eos_token_id ({tokenizer.eos_token_id}). Using model config value."
            )

        # Setup generation config
        generation_config = GenerationConfig(
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=True,
            output_logits=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Run continuous batching mode
        run_continuous_batching(args, tokenizer, generation_config)

        logger.info("=" * 60)
        logger.info("INFERENCE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Inference failed with error: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise

    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
