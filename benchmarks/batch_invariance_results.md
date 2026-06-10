# Batch-Invariant vs Variant Throughput

Throughput cost of batch-invariance, measured on a single H100. Covers two
engines:

- **VeXact** — its `variant` mode (commit `support variant mode`) disables the
    deterministic ATen patches and unlocks FA Split-KV.
- **vllm 0.14.1** — its native batch-invariant mode, toggled by the
    `VLLM_BATCH_INVARIANT` env var.

Both trade bit-level reproducibility for throughput. The VeXact and vllm numbers
are **not** directly comparable (different engines / graph settings); each
engine's own on/off delta is the meaningful result.

## Summary (256 requests, 1× H100, Qwen3-1.7B)

| Engine                             | batch-invariant | Throughput (req/s) | Total tokens/s | Wall time | Wall-time slowdown (ON vs OFF) |
| ---------------------------------- | --------------- | ------------------ | -------------- | --------- | ------------------------------ |
| VeXact                             | ON (invariant)  | 9.25               | 4015.19        | 27.66s    | **+82%** (1.82×)               |
| VeXact                             | OFF (variant)   | 16.86              | 7314.25        | 15.19s    | — (baseline)                   |
| vexact-prefill-cudagraph           | ON (invariant)  | 14.72              | 6385.00        | 17.40s    | **+56%** (1.56×)               |
| vexact-prefill-cudagraph           | OFF (variant)   | 23.00              | 9976.88        | 11.13s    | — (baseline)                   |
| vexact-prefill-cudagraph-scheduler | ON (invariant)  | 16.12              | 6993.82        | 15.88s    | **+72%** (1.72×)               |
| vexact-prefill-cudagraph-scheduler | OFF (variant)   | 27.77              | 12045.83       | 9.22s     | — (baseline)                   |
| vllm                               | ON (invariant)  | 10.83              | 4698.53        | 23.64s    | **+127%** (2.27×)              |
| vllm                               | OFF (variant)   | 24.59              | 10666.40       | 10.41s    | — (baseline)                   |

Turning batch-invariance ON costs **+82% wall time on VeXact** (1.82×) and
**+127% on vllm** (2.27×). Cross-engine absolute numbers are not comparable
(VeXact runs with CUDA graph; vllm runs with `enforce_eager`). Per-engine
details and metric definitions follow below.

### Summary — 512 requests

| Engine                             | batch-invariant | Throughput (req/s) | Total tokens/s | Wall time | Wall-time slowdown (ON vs OFF) |
| ---------------------------------- | --------------- | ------------------ | -------------- | --------- | ------------------------------ |
| VeXact                             | ON (invariant)  | 11.82              | 5030.05        | 43.30s    | **+77%** (1.77×)               |
| VeXact                             | OFF (variant)   | 20.88              | 8881.09        | 24.52s    | — (baseline)                   |
| vexact-prefill-cudagraph           | ON (invariant)  | 19.37              | 8237.76        | 26.44s    | **+53%** (1.53×)               |
| vexact-prefill-cudagraph           | OFF (variant)   | 29.63              | 12604.04       | 17.28s    | — (baseline)                   |
| vexact-prefill-cudagraph-scheduler | ON (invariant)  | 22.96              | 9768.09        | 22.30s    | **+52%** (1.52×)               |
| vexact-prefill-cudagraph-scheduler | OFF (variant)   | 34.89              | 14842.32       | 14.67s    | — (baseline)                   |
| vllm                               | ON (invariant)  | 18.95              | 8060.05        | 27.02s    | **+111%** (2.11×)              |
| vllm                               | OFF (variant)   | 40.03              | 17028.39       | 12.79s    | — (baseline)                   |

## Setup

| Item      | Value                                                                 |
| --------- | --------------------------------------------------------------------- |
| Model     | `Qwen3-1.7B`                                                          |
| Dataset   | `ShareGPT_V3_unfiltered_cleaned_split.json`                           |
| Requests  | 256                                                                   |
| GPU       | 1× H100-SXM-80GB (`CUDA_VISIBLE_DEVICES=0`)                           |
| Scheduler | `max_num_batched_tokens=2048`, `max_num_seqs=512`, chunked prefill on |
| Branch    | `feat/switch_mode` (rebased on `main`)                                |

Reproduce with `benchmarks/run_invariance_compare.sh` (knobs: `NUM_REQUESTS`, `GPU`).

# VeXact

## Results

### 256 requests

| Metric             | invariant (default) | variant | Δ        |
| ------------------ | ------------------- | ------- | -------- |
| Throughput (req/s) | 9.25                | 16.86   | **+82%** |
| Total tokens/s     | 4015.19             | 7314.25 | **+82%** |
| Output tokens/s    | 2017.25             | 3674.71 | **+82%** |
| Wall time          | 27.66s              | 15.19s  | **−45%** |
| Avg latency        | 15.31s              | 8.44s   | −45%     |
| P50 latency        | 16.26s              | 9.01s   | −45%     |
| P95 latency        | 24.28s              | 13.39s  | −45%     |

### 512 requests

| Metric             | invariant (default) | variant | Δ        |
| ------------------ | ------------------- | ------- | -------- |
| Throughput (req/s) | 11.82               | 20.88   | **+77%** |
| Total tokens/s     | 5030.05             | 8881.09 | **+77%** |
| Output tokens/s    | 2597.63             | 4586.38 | **+77%** |
| Wall time          | 43.30s              | 24.52s  | **−43%** |
| Avg latency        | 24.72s              | 13.83s  | −44%     |
| P50 latency        | 26.31s              | 14.72s  | −44%     |
| P95 latency        | 39.21s              | 22.21s  | −43%     |

The speedup holds across scales (+82% at 256 req, +77% at 512 req) — it is not
a small-sample artifact. Larger batches lift both modes (fuller batches), so the
relative gap stays roughly constant.

> Token counts differ slightly between runs because sampling draws different
> ShareGPT samples; throughput/latency are the comparable metrics.

## Metric definitions

All requests are submitted concurrently via `asyncio.gather`; **wall time** is
the clock time wrapping that gather (`benchmarks/throughput.py:147-149`):

```
wall_time   = t_end − t_start          # around gather() of all concurrent requests
throughput  = completed / wall_time    # requests/s
total_tps   = (prompt_tokens + output_tokens) / wall_time
output_tps  = output_tokens / wall_time
latency_i   = t_done_i − t_submit_i    # per-request, end-to-end
avg/P50/P95 = mean / 50th / 95th percentile over latency_i
```

Because requests run concurrently, `wall_time` is the makespan of the whole
batch — close to `max(latency_i)`, not their sum.

## What changes between modes

| Layer                               | invariant                                                               | variant                                                                    |
| ----------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| ATen ops (`enable_batch_invariant`) | deterministic Triton kernels patched in (matmul, rms_norm, …)           | patches off — native CUDA kernels                                          |
| Attention (`attn_impl`)             | `fa-invariant` — FA forced to `num_splits=1`, fixed LSE reduction order | `fa-variant` — `num_splits` unlocked, kernel picks Split-KV by batch shape |

Both effects are confirmed in the run logs (`batch invariant mode DISABLED (variant)`,
`attn_impl: fa-variant`).

## Takeaway

Disabling batch-invariance is **~1.8× faster** end-to-end and nearly halves
wall time. The invariant mode spends roughly **45% extra latency** to buy
bit-level reproducibility / batch invariance — useful when training and
inference must produce identical logits regardless of how requests are batched.

## How to switch

```bash
# invariant (default)
python benchmarks/throughput.py --model-path <model> --dataset-path <data> --mode invariant

# variant
python benchmarks/throughput.py --model-path <model> --dataset-path <data> --mode variant
```

In code (`ModelConfig`): `enable_batch_invariant=True/False` paired with
`attn_impl="fa-invariant"` / `"fa-variant"`.

# vllm (native batch-invariant)

vllm ships its own batch-invariant mode (`vllm.model_executor.layers.batch_invariant`),
independent of VeXact's. It is toggled by the `VLLM_BATCH_INVARIANT` env var and
**requires** an explicit attention backend (`FLASH_ATTN` / `FLASHINFER` /
`*_MLA`) — enabling it with the default `None` backend raises at engine init.

## Setup

| Item            | Value                                                                                     |
| --------------- | ----------------------------------------------------------------------------------------- |
| vllm            | 0.14.1                                                                                    |
| Model / Dataset | `Qwen3-1.7B` / same ShareGPT file as above                                                |
| Requests        | 256 (identical samples — same loader + fixed seed)                                        |
| Output length   | fixed via `ignore_eos`, so on/off do identical token work                                 |
| Engine          | `enforce_eager=True` for **both** runs (cudagraph off, isolates the batch-invariant cost) |
| Backend         | `VLLM_ATTENTION_BACKEND=FLASH_ATTN` for both                                              |
| GPU             | 1× H100-SXM-80GB                                                                          |

Reproduce with `benchmarks/run_vllm_invariance_compare.sh`
(benchmark: `benchmarks/vllm_throughput.py`).

## Results (256 requests)

| Metric               | invariant (`=1`) | variant (`=0`) | Δ         |
| -------------------- | ---------------- | -------------- | --------- |
| Throughput (req/s)   | 10.83            | 24.59          | **+127%** |
| Total tokens/s       | 4698.53          | 10666.40       | **+127%** |
| Output tokens/s      | 2360.56          | 5358.84        | **+127%** |
| Wall time (makespan) | 23.64s           | 10.41s         | **−56%**  |

Prompt/output token counts were identical across runs (55265 / 55799).
Per-request latency percentiles are omitted: offline `LLM.generate` in vllm
0.14.1 does not populate `RequestOutput.metrics`; measuring them needs the
`AsyncLLMEngine` submit-and-await path.

## Takeaway

vllm's batch-invariant mode is **~2.3× slower** (variant +127%) under eager
mode — a steeper penalty than VeXact's ~1.8×, since vllm replaces more ATen ops
(addmm/bmm/mm/rms_norm/softmax/log_softmax/mean) with deterministic kernels.

## How to switch

```bash
# variant (default) — batch-invariant off
VLLM_BATCH_INVARIANT=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN python ...

# invariant — batch-invariant on (attention backend is mandatory)
VLLM_BATCH_INVARIANT=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN python ...
```
