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

"""Test throughput of concurrent generate calls with asyncio tasks."""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

# os.environ["VEXACT_LOGGING_LEVEL"] = "DEBUG"
from vexact.config import DriverConfig, ModelConfig, ParallelConfig, ProfilerConfig, SchedulerConfig, VeXactConfig
from vexact.core.request import DriverRequest, GenerationConfig
from vexact.engine import VExact


BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from dataset import ShareGPTDataset  # noqa: E402


def build_vexact_engine(
    model_path: str,
    pp_size: int = 1,
    max_num_batched_tokens: int = 8,
    max_num_seqs: int = 512,
    enable_chunked_prefill: bool = True,
    profiler_backend: str | None = None,
    profiler_output: str | None = None,
    profiler_delay_iterations: int = 0,
    profiler_max_iterations: int = 0,
):
    config = VeXactConfig(
        model=ModelConfig(
            model_path=model_path,
            attn_impl="fa-invariant",
            enable_batch_invariant=True,
            enable_memory_saver=False,
            enforce_eager=False,
            use_fp32_logits=True,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=pp_size,
        ),
        driver=DriverConfig(
            is_worker_proc_managed=True,
        ),
        profiler=ProfilerConfig(
            backend=profiler_backend,
            output_path=profiler_output,
            delay_iterations=profiler_delay_iterations,
            max_iterations=profiler_max_iterations,
            profile_all_ranks=True,
        ),
        scheduler=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            max_queue_size=0,  # unlimited
        ),
    )
    return VExact(config)


def _load_sharegpt_samples(
    tokenizer,
    num_requests: int | None,
    output_len: int | None,
    dataset_path: str,
    include_multimodal: bool = False,
):
    dataset = ShareGPTDataset(dataset_path=dataset_path)
    wants_all = num_requests is None
    if num_requests is None:
        num_requests = len(dataset.data)
    samples = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        output_len=output_len,
        enable_multimodal_chat=False,
        no_oversample=True,
    )
    if not include_multimodal:
        samples = [sample for sample in samples if sample.multi_modal_data is None]
    if not wants_all and len(samples) < num_requests:
        raise ValueError(
            f"Only sampled {len(samples)} text-only requests; need {num_requests}. "
            "Provide a larger dataset or adjust output length."
        )
    return samples if wants_all else samples[:num_requests]


async def _run_test(vexact_engine, samples, timeout_s: float | None):
    total_prompt_tokens = 0
    total_output_tokens = 0
    latencies = []
    errors = []
    completed = 0

    async def submit_one(sample):
        nonlocal total_prompt_tokens, total_output_tokens, completed
        prompt = sample.prompt
        try:
            input_ids = vexact_engine.tokenizer.encode(prompt, add_special_tokens=True)
            gen_config = GenerationConfig(
                max_new_tokens=sample.expected_output_len,
                max_length=vexact_engine.config.model.max_model_len,
                do_sample=True,
                temperature=1,
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            request = DriverRequest(
                generation_config=gen_config,
                input_ids_list=input_ids,
            )
            t0 = time.time()
            result = await vexact_engine.generate(request, timeout=timeout_s)
            latency = time.time() - t0
            total_prompt_tokens += len(input_ids)
            total_output_tokens += len(result.new_token_ids)
            latencies.append(latency)
            completed += 1
        except Exception as e:
            errors.append(e)

    t0 = time.time()
    await tqdm_asyncio.gather(*[submit_one(sample) for sample in samples], desc="Generating")
    total_time = time.time() - t0

    return total_prompt_tokens, total_output_tokens, latencies, errors, completed, total_time


def run_throughput(
    vexact_engine,
    total_requests: int | None,
    output_len: int | None,
    dataset_path: str,
    timeout_s: float | None,
    include_multimodal: bool = False,
):
    tokenizer = vexact_engine.tokenizer
    samples = _load_sharegpt_samples(
        tokenizer,
        total_requests,
        output_len,
        dataset_path,
        include_multimodal=include_multimodal,
    )

    total_prompt_tokens, total_output_tokens, latencies, errors, completed, total_time = asyncio.run(
        _run_test(vexact_engine, samples, timeout_s)
    )
    return total_prompt_tokens, total_output_tokens, latencies, errors, completed, total_time


def _parse_args():
    parser = argparse.ArgumentParser(description="VExact ShareGPT throughput benchmark.")
    parser.add_argument("--model-path", required=True, help="Path to the model.")
    parser.add_argument("--dataset-path", required=True, help="Path to the ShareGPT JSON dataset.")
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Override output length. If unset, uses dataset completion length.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Total number of requests to submit. If unset, uses the entire dataset.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Per-request timeout in seconds. If unset, waits indefinitely.",
    )
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size.")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=2048,
        help="Scheduler max_num_batched_tokens.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=512,
        help="Scheduler max_num_seqs.",
    )
    parser.add_argument(
        "--disable-chunked-prefill",
        action="store_true",
        help="Disable chunked prefill in scheduler.",
    )
    parser.add_argument(
        "--include-multimodal",
        action="store_true",
        help="Include multimodal samples if present in the dataset.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (e.g., DEBUG, INFO, WARNING).",
    )
    parser.add_argument(
        "--profile-backend",
        choices=["torch", "proton"],
        default=None,
        help="Enable GPU profiling with the selected backend.",
    )
    parser.add_argument(
        "--profile-output",
        default=None,
        help="Output file for profiler trace (default: auto-generated).",
    )
    parser.add_argument(
        "--profile-delay-iterations",
        type=int,
        default=0,
        help="Number of steps to wait before starting profiler (default: 0).",
    )
    parser.add_argument(
        "--profile-max-iterations",
        type=int,
        default=0,
        help="Number of steps to profile (0 = until manually stopped, default: 0).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        force=True,
    )
    engine = build_vexact_engine(
        model_path=args.model_path,
        pp_size=args.pp_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        profiler_backend=args.profile_backend,
        profiler_output=args.profile_output,
        profiler_delay_iterations=args.profile_delay_iterations,
        profiler_max_iterations=args.profile_max_iterations,
    )
    try:
        total_prompt_tokens, total_output_tokens, latencies, errors, completed, total_time = run_throughput(
            engine,
            total_requests=args.num_requests,
            output_len=args.output_len,
            dataset_path=args.dataset_path,
            timeout_s=args.timeout_s,
            include_multimodal=args.include_multimodal,
        )
    finally:
        engine.close()

    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for i, err in enumerate(errors[:5]):
            print(f"Error {i + 1}: {type(err).__name__}: {err}")

    latencies_sorted = sorted(latencies)
    avg_latency = (sum(latencies_sorted) / len(latencies_sorted)) if latencies_sorted else 0.0
    p50 = latencies_sorted[int(0.50 * len(latencies_sorted)) - 1] if latencies_sorted else 0.0
    p95 = latencies_sorted[int(0.95 * len(latencies_sorted)) - 1] if latencies_sorted else 0.0

    total_tokens = total_prompt_tokens + total_output_tokens
    rps = (completed / total_time) if total_time > 0 else 0.0
    total_tps = (total_tokens / total_time) if total_time > 0 else 0.0
    output_tps = (total_output_tokens / total_time) if total_time > 0 else 0.0

    print(f"\n{'=' * 60}")
    print("THROUGHPUT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Throughput: {rps:.2f} requests/s, {total_tps:.2f} total tokens/s, {output_tps:.2f} output tokens/s")
    print(f"Total num prompt tokens:  {total_prompt_tokens}")
    print(f"Total num output tokens:  {total_output_tokens}")
    print(f"Total wall time: {total_time:.3f}s")
    print(f"Avg latency: {avg_latency:.3f}s")
    print(f"P50 latency: {p50:.3f}s")
    print(f"P95 latency: {p95:.3f}s")


if __name__ == "__main__":
    main()
