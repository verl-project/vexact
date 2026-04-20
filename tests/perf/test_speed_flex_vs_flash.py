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
import sys

import torch
import torch.nn as nn


# Add parent directory to path to import kv_cache_context
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vexact.batch_invariant_ops import flex_attention
from vexact.batch_invariant_ops.flash_attention import flash_attention_forward
from vexact.batch_invariant_ops.flex_attention import flex_attention_forward
from vexact.batch_invariant_ops.kv_cache_context import (
    set_kv_cache_context,
)


torch.set_default_device("cuda")


class MockModule(nn.Module):
    def __init__(self, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        # Required by flex_attention_forward
        self.num_key_value_groups = 2  # H (8) / KV_Heads (4)
        # Required by vLLM flex attention - must be tensors, not floats
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)


def run_benchmark(name, func, args, warmup=5, active=20):
    print(f"Benchmarking {name}...")

    # Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    # Measure
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(active)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(active)]

    for i in range(active):
        start_events[i].record()
        func(*args)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = sum(times) / active
    print(f"{name}: {avg_time:.3f} ms")
    return avg_time


def benchmark_attention():
    # Parameters
    B = 4
    H = 8
    D = 64
    S = 1024
    KV_Heads = 4
    Page_Size = 16

    print(f"Config: B={B}, H={H}, D={D}, S={S}, KV_Heads={KV_Heads}, Page_Size={Page_Size}")

    # Data Generation
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, KV_Heads, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, KV_Heads, S, D, dtype=torch.bfloat16)

    mock_module = MockModule()
    scaling = float(D) ** -0.5

    args_common = (mock_module, q, k, v, None, scaling)

    results = {}

    # Paged Data Setup
    # Create Paged Cache
    total_tokens = B * S
    num_blocks = (S + Page_Size - 1) // Page_Size
    total_blocks = B * num_blocks

    # CRITICAL FIX: k_cache/v_cache must be dict[layer_idx -> tensor]
    # Tensor shape: (num_blocks, page_size, num_kv_heads, head_dim)
    k_cache = {0: torch.randn(total_blocks, Page_Size, KV_Heads, D, dtype=torch.bfloat16)}
    v_cache = {0: torch.randn(total_blocks, Page_Size, KV_Heads, D, dtype=torch.bfloat16)}

    block_tables = torch.arange(total_blocks, dtype=torch.int32).reshape(B, num_blocks)
    context_lens = torch.full((B,), S, dtype=torch.int32)
    query_start_loc = torch.arange(0, (B + 1) * S, S, dtype=torch.int32)

    slot_mapping = torch.arange(total_tokens, dtype=torch.int64)  # vLLM requires int64

    # 1. Flex Unpaged Compiled
    flex_attention.FLEX_ATTENTION_USE_COMPILE = True
    set_kv_cache_context(
        is_paged_attn=False,
        query_start_loc=torch.arange(0, (B + 1) * S, S, dtype=torch.int32),
        context_lens=torch.full((B,), S, dtype=torch.int32),
    )

    results["Flex Unpaged"] = run_benchmark("Flex Unpaged", flex_attention_forward, args_common)

    # 2. Flex Paged
    flex_attention.FLEX_ATTENTION_USE_COMPILE = False
    set_kv_cache_context(
        is_paged_attn=True,
        key_cache=k_cache,
        value_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        query_start_loc=query_start_loc,
        slot_mapping=slot_mapping,
    )

    results["Flex Paged"] = run_benchmark("Flex Paged", flex_attention_forward, args_common)

    # 3. Flex Paged Compiled
    flex_attention.FLEX_ATTENTION_USE_COMPILE = True
    set_kv_cache_context(
        is_paged_attn=True,
        key_cache=k_cache,
        value_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        query_start_loc=query_start_loc,
        slot_mapping=slot_mapping,
    )

    print("Warmup for Compile...")
    flex_attention_forward(*args_common)

    results["Flex Paged Compiled"] = run_benchmark("Flex Paged Compiled", flex_attention_forward, args_common)
    flex_attention.FLEX_ATTENTION_USE_COMPILE = False

    # 4. Flash Paged
    set_kv_cache_context(
        is_paged_attn=True,
        key_cache=k_cache,
        value_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        query_start_loc=query_start_loc,
        slot_mapping=slot_mapping,
        max_seqlen_q=S,
    )

    results["Flash Paged"] = run_benchmark("Flash Paged", flash_attention_forward, args_common)

    print("\n" + "=" * 50)
    print(f"{'Method':<25} | {'Latency (ms)':<15}")
    print("-" * 50)
    for name, latency in results.items():
        print(f"{name:<25} | {latency:.3f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    benchmark_attention()
