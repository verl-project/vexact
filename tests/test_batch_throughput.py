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

import asyncio
import time

from vexact.core.request import DriverRequest, GenerationConfig


def test_generate_throughput(vexact_engine):
    """Measure throughput (requests/sec and tokens/sec) under concurrency."""
    num_workers = 5
    requests_per_worker = 5

    tokenizer = vexact_engine.tokenizer

    prompts = [
        "What is 1+1?",
        "The capital of France is",
        "Machine learning is",
        "Hello, how are",
        "Python is a",
    ]

    async def run_test():
        total_tokens = 0
        latencies = []
        errors = []
        completed = 0

        async def worker(worker_id):
            nonlocal total_tokens, completed
            for j in range(requests_per_worker):
                prompt = prompts[(worker_id + j) % len(prompts)]
                try:
                    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
                    gen_config = GenerationConfig(
                        max_new_tokens=15,
                        do_sample=True,
                        temperature=1,
                    )
                    request = DriverRequest(
                        generation_config=gen_config,
                        input_ids_list=input_ids,
                    )
                    t0 = time.time()
                    result = await vexact_engine.generate(request, timeout=60.0)
                    latency = time.time() - t0
                    total_tokens += len(result.new_token_ids)
                    latencies.append(latency)
                    completed += 1
                except Exception as e:
                    errors.append(e)

        t0 = time.time()
        await asyncio.gather(*[worker(i) for i in range(num_workers)])
        total_time = time.time() - t0

        return total_tokens, latencies, errors, completed, total_time

    total_tokens, latencies, errors, completed, total_time = asyncio.run(run_test())
    total_requests = num_workers * requests_per_worker

    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for i, err in enumerate(errors[:5]):
            print(f"Error {i + 1}: {type(err).__name__}: {err}")

    assert completed == total_requests, f"{completed} of {total_requests} completed. {len(errors)} errors occurred."
    assert len(errors) == 0, f"{len(errors)} errors occurred: {errors[0] if errors else None}"

    latencies_sorted = sorted(latencies)
    avg_latency = (sum(latencies_sorted) / len(latencies_sorted)) if latencies_sorted else 0.0
    p50 = latencies_sorted[int(0.50 * len(latencies_sorted)) - 1] if latencies_sorted else 0.0
    p95 = latencies_sorted[int(0.95 * len(latencies_sorted)) - 1] if latencies_sorted else 0.0

    rps = (total_requests / total_time) if total_time > 0 else 0.0
    tps = (total_tokens / total_time) if total_time > 0 else 0.0

    print(f"\n{'=' * 60}")
    print("THROUGHPUT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Workers: {num_workers}")
    print(f"Requests per worker: {requests_per_worker}")
    print(f"Total requests: {total_requests}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total wall time: {total_time:.3f}s")
    print(f"Requests/sec: {rps:.2f}")
    print(f"Tokens/sec: {tps:.2f}")
    print(f"Avg latency: {avg_latency:.3f}s")
    print(f"P50 latency: {p50:.3f}s")
    print(f"P95 latency: {p95:.3f}s")
