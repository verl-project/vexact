# Benchmarks

Throughput benchmarks for the VeXact inference engine on the ShareGPT dataset.

> **Note:** VeXact currently supports text-only input.

## Layout

```
benchmarks/
├── dataset.py      # ShareGPT loader
├── throughput.py   # Concurrent asyncio throughput benchmark
└── README.md
```

## Dataset

`ShareGPTDataset` loads a JSON file in the
[Aeala/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered)
format — a list of conversations, each with at least two turns. Entries with
fewer than two turns are dropped; the first turn is used as the prompt and the
second as the reference completion.

`sample(tokenizer, num_requests, output_len=None, ...)` returns a list of
`SampleRequest`:

- `prompt`, `prompt_len` — the user turn and its token count.
- `expected_output_len` — either the reference completion length or the value
    passed via `output_len`.
- `multi_modal_data` — optional image/video payload (`data:` URL for raw bytes
    or `file://` / `http(s)://` URL for paths).

Default filtering (`is_valid_sequence`): `4 ≤ prompt_len ≤ 1024`,
`output_len ≥ 4`, `prompt_len + output_len ≤ 2048`.

## Throughput benchmark

`throughput.py` spins up a `VeXact` engine and fires all sampled requests
concurrently via `asyncio.gather`. It reports RPS, total tokens/s, output
tokens/s, and avg / P50 / P95 latency.

### Quick start

```bash
python benchmarks/throughput.py \
    --model-path  /path/to/model \
    --dataset-path /path/to/ShareGPT_V3_unfiltered.json \
    --num-requests 512
```

### Switching attention backend

Defaults to flash attention (`fa-invariant`). To run the same benchmark with
flex attention:

```bash
python benchmarks/throughput.py \
    --model-path  /path/to/model \
    --dataset-path /path/to/ShareGPT_V3_unfiltered.json \
    --num-requests 512 \
    --attn-impl flex
```

Valid values: `fa-invariant` (default), `fa-invariant-cute`, `flex`.

> **Note:** flex attention is currently incompatible with CUDA graph. When
> running with `--attn-impl flex`, edit `build_vexact_engine` in
> `throughput.py` to set `enforce_eager=True`

### Arguments

| Flag                         | Default        | Purpose                                                                |
| ---------------------------- | -------------- | ---------------------------------------------------------------------- |
| `--model-path`               | *(required)*   | HF-format model directory.                                             |
| `--dataset-path`             | *(required)*   | ShareGPT JSON file.                                                    |
| `--num-requests`             | all            | Number of requests to submit.                                          |
| `--output-len`               | dataset        | Override completion length per request.                                |
| `--timeout-s`                | none           | Per-request timeout, `None` = wait forever.                            |
| `--pp-size`                  | 1              | Pipeline parallel size.                                                |
| `--max-num-batched-tokens`   | 2048           | Scheduler batched-token cap.                                           |
| `--max-num-seqs`             | 512            | Scheduler concurrency cap.                                             |
| `--disable-chunked-prefill`  | off            | Disable chunked prefill.                                               |
| `--include-multimodal`       | off            | Keep image/video samples (prompt text only; media not sent to engine). |
| `--attn-impl`                | `fa-invariant` | `fa-invariant` / `fa-invariant-cute` / `flex`.                         |
| `--log-level`                | `INFO`         | Python logging level.                                                  |
| `--profile-backend`          | off            | `torch` or `proton` GPU profiler.                                      |
| `--profile-output`           | auto           | Profiler trace path.                                                   |
| `--profile-delay-iterations` | 0              | Warmup steps before profiling.                                         |
| `--profile-max-iterations`   | 0              | Profiled step count (0 = until stop).                                  |

### Fixed engine settings

`build_vexact_engine` hard-codes a few knobs geared toward deterministic
throughput measurement:

- `enable_batch_invariant=True`, `use_fp32_logits=True`, `enforce_eager=False`
- `is_worker_proc_managed=True` (driver spawns workers)
- `max_queue_size=0` (unlimited scheduler queue)

Tweak them in `throughput.py` if a run needs different behavior.

## Example output

Qwen3-1.7B, 100 ShareGPT requests, 2× H100:

```
============================================================
THROUGHPUT SUMMARY
============================================================
Throughput: 5.84 requests/s, 2537.86 total tokens/s, 1219.29 output tokens/s
Total num prompt tokens:  22594
Total num output tokens:  20893
Total wall time: 17.135s
Avg latency: 7.853s
P50 latency: 7.226s
P95 latency: 15.478s
```

## Batch-invariant kernel performance

Notes behind the batch-invariance performance work, measured on one H100 SXM. Every change listed here
leaves rollout tokens and logprobs bitwise identical, so training/rollout alignment is unaffected.

### What invariance constrains

A reduction's result depends on its evaluation tree, because floating-point addition is not
associative. Batch invariance needs each output's tree to depend only on static quantities (K,
head_dim, a static chunk size) and the element's own extent (its KV length), never on the batch.
That leaves a lot free:

| free per call                                           | fixed (it is the tree)                                    |
| ------------------------------------------------------- | --------------------------------------------------------- |
| tile shape (BM, BN, BK), stages, warps, persistence     | where K / KV is split and the order partials are combined |
| which CTA computes which tile, grid size, rasterization | FMA contraction and rounding points in epilogues          |
| on H100 bf16: `wgmma` vs `mma.sync`, any instruction N  |                                                           |

On H100 every tile and MMA instruction gives identical bits for a fixed K order. Two compiler
hazards change bits silently and need guarding: Triton folds `dot` + `add` across a K chunk boundary
when a chunk has a single k-iteration, and it drops a `.to(bf16).to(fp32)` round trip in front of an
add (round with integer bit operations instead).

### Is invariance free? A first-order model

Per tile, time = max(tensor-core time, L2→SM ingest time) + fixed overhead; tiles run in waves over
the SMs; nothing beats streaming the operands through HBM. Compared with an oracle that may pick any
tile and any split-K per call:

- **Dense GEMM.** A fixed 128×128 tile without K splits loses up to 1.5–2.5× at decode batch sizes
    on narrow layers (o_proj, down: too few tiles to pull HBM bandwidth). Choosing the tile per shape
    recovers most of it at no numerical risk. The rest needs a static K split chosen from (N, K) only.
- **Grouped (MoE) GEMM.** Within 3–12% of the oracle: the expert dimension supplies the parallelism
    a small batch can't.
- **Decode attention.** `num_splits=1` loses 3–6.5× at small batch × long context (RL's long-tail
    straggler phase). A fixed split *size* with an in-kernel ordered combine removes it, but training
    must then use the same split.

### Changes and measured effect

- **Dense GEMM tiles** (`matmul_persistent`, used by training and rollout). Rows beyond M were
    clamped to row 0, which made M < BLOCK_M 3–4.5× slower; they are now masked. The bf16 tile is
    picked per shape from 5 candidates by the model above, with constants fitted to measured times
    (0.13% geomean regret). Decode sizes (M ≤ 64) are 2.2–5× faster, prefill sizes 1.1–1.4×.
- **LM head on sampled rows only.** Rollout ran `lm_head` on every packed token, prefill chunks and
    CUDA-graph padding included, then kept one row per request. It now passes each request's last row
    as `logits_to_keep`. About 2.5 TFLOP saved on a 4096-token mixed step of Qwen3-1.7B.

## Acknowledgements

Adapted from [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).
