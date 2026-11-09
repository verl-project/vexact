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
- **MoE expert GEMM launches.** VeOmni ships no tuned configs, so every expert GEMM ran a fallback
    128×128×32 tile. Launching the same kernel with a tile chosen by routed-row count reaches the
    weight-streaming bound at decode: the MoE layer is 1.16–1.45× faster for T ≤ 512 tokens
    (Qwen3-30B-A3B shapes).
- **`matmul_ogs` for prefill-size expert GEMMs.** Above 2048 routed rows, triton_kernels' ragged
    `matmul_ogs` (split-K pinned to 1) gives VeOmni's bits at 400–530 TFLOP/s where VeOmni's kernel
    stays near 200: the MoE layer is 1.30–1.71× faster at T = 2048–8192.
- **Inference-only fused MoE forward.** With autograd off, the token gather is fused into fc1's
    loads and `silu(gate) * up * gate_weight` runs as one Triton pass. That pass must replay VeOmni's
    three bf16 roundings with IEEE `expf` and division: a single fp32 chain with one rounding (as
    DeepGEMM's Mega MoE does) mismatches 33% of outputs. MoE layer 1.33–2.10× over VeOmni as shipped.
- **No NaN fill of `torch.empty`.** `torch.use_deterministic_algorithms(True)` also fills every
    uninitialized allocation with NaN: about 120 fill kernels per decode step, 9% of GPU time on the
    MoE rollout. Every kernel writes its outputs before they are read, so batch-invariant mode turns
    the fill off (1.07–1.08× end to end).
- **Fused RMSNorm and RoPE** (no-grad calls on HF Qwen3 / Qwen3-MoE). RMSNorm is written in Gluon so
    each row is summed in exactly the layout of vexact's `mean_kernel`; a plain Triton kernel loads bf16
    8-wide and changes 30% of the variances. Intermediate bf16 roundings use integer bit operations,
    since Triton folds `.to(bf16).to(fp32)` in front of an add (6% of RoPE outputs changed). Row
    lengths are limited to multiples of 1024 or a single power-of-two block. +17–28% end to end on
    top of the earlier changes.
- **Async scheduling** (`SchedulerConfig.async_scheduling`, off by default). The generation thread
    blocked on each step's `.cpu()` of sampled tokens, then built the next step while the GPU idled
    (GPU busy 84% of wall time). Step N+1 now launches before step N's tokens reach the host, with
    decode inputs copied on the GPU. Batch composition changes but, by batch invariance, no token does.
    1.21× on a 4096-request MoE rollout.
- **Norm and RoPE at the bandwidth bound.** A 128-wide row in `mean_kernel`'s layout sits entirely in
    warp 0, so its sum is "4 values per lane in order, then a 32-lane butterfly"; a kernel that keeps
    exactly that while packing 16 rows per program takes the q/k norms from 235 to 46.5 µs (262k rows).
    RoPE runs one program per token over all heads: 159 → 48.9 µs (8192 tokens × 32 heads).
- **MoE routing as a counting sort.** VeOmni orders routed rows with a histogram, a general stable
    `argsort` (multi-pass radix sort), an inverse scatter and a `cumsum`: ~40 µs per call at decode.
    Expert ids are small integers, so a stable counting sort gives the same permutation in three small
    kernels (per-block histograms, a column-parallel prefix, in-block ranks from sorting unique keys
    `expert * BLOCK + i`). 3.3–14× faster from 1 to 65,536 tokens. A first version scanned blocks in
    one program and lost to the radix sort above ~32k routed rows; parallelizing the prefix over
    expert columns and growing the block to 1024 rows fixed that without a fallback.
- **Sampler.** With `top_p=1.0` (verl's default) the sampler still sorted the full vocabulary,
    then un-sorted it with `scatter`, which deterministic mode turns into a sorted `index_put`: 35 ms
    of a 66 ms decode step at 512 sequences. `top_p >= 1` now skips top-p, and real top-p un-sorts
    with a one-writer-per-element Triton kernel (14 ms, the vocabulary sort itself).
- **4-stage GEMM tile.** The 128×128×64 tile with 4 pipeline stages instead of 3 is 5% faster per
    Qwen3-8B layer at M = 512 and gives the same bits; 3 stages remain where 4 do not fit shared
    memory. +2.5% end to end on a 512-request Qwen3-8B wave.
- **Fused projections** (no-grad forwards). Each model declares which projections may be fused next to
    its other patches (q/k/v and gate/up for Qwen3 and Qwen3-MoE); their weights are concatenated once
    at model creation and the separate parameters become views of them, so loading and weight syncs fill
    the fused weight in place. Rollout then runs 4 GEMMs per dense layer instead of 7, reading the
    activations once per group. The batch-invariant matmul reduces each output column over K in the same
    order whatever N is, so the bits are unchanged and training, which keeps separate projections, stays
    aligned. +2.4% end to end on top of the tile change (+4.9% over the previous stack).
- **Dense SwiGLU in one pass.** A dense MLP read the fused gate/up GEMM output back twice, as two
    strided halves, to run `silu` and a multiply: 4 of the 5 elementwise kernels per layer, against
    vLLM's single fused `triton_poi_fused_mul_silu_slice`. The inference fused-MoE SwiGLU kernel is
    the same arithmetic plus a routing-weight multiply, so it takes the whole GEMM output with that
    multiply switched off at compile time; it already replays aten's two bf16 roundings, so the bits
    are eager's. Only a bf16 MLP whose activation is SiLU uses it. +3.8% (GSM8K) and +3.0% (512
    requests × 512 tokens) end to end on Qwen3-8B. Note that `hidden_act: "silu"` builds
    transformers' own `SiLUActivation`, which is not an `nn.SiLU` subclass: a guard that checks only
    `nn.SiLU` leaves the kernel unreachable, and no bitwise test can see the difference, so a test
    asserts the kernel is reached once per dense layer.

### End to end vs vLLM

vLLM 0.19.1, one H100, same token workloads (EOS ignored, temperature 1, KV cache capped equally;
vexact with async scheduling). Output tokens/s:

| model / workload       | vexact | vLLM batch-invariant | vLLM default |
| ---------------------- | -----: | -------------------: | -----------: |
| Qwen3-8B GSM8K         | 14,483 |               13,892 |       16,690 |
| Qwen3-8B ShareGPT      |  8,738 |                6,910 |       11,122 |
| Qwen3-30B-A3B GSM8K    |  5,807 |                6,080 |        7,795 |
| Qwen3-30B-A3B ShareGPT |  4,497 |                4,425 |        5,806 |

vLLM's batch-invariant mode makes vLLM consistent with itself, not with a training engine. vexact is
at 0.96–1.27× of it and 0.74–0.87× of vLLM's default mode. The 4-stage tile, the fused projections and
the fused dense SwiGLU are worth +8.7% on Qwen3-8B GSM8K but only +0.4% on Qwen3-30B-A3B GSM8K, where
every layer is MoE: its experts already hold merged gate/up weights and run the SwiGLU kernel already,
so only q/k/v is fused there.

Per decode step on Qwen3-8B at 512 sequences, what is left against vLLM's default mode is attention
(vLLM's FA3 fork picks a 64-row query tile with one consumer warpgroup when the query rows per KV head
are ≤ 64, which is 14–27% faster per decode call and returns the same bits) and GEMM: cuBLAS still wins
by a margin that tile choice alone does not close, since at decode it splits K per batch size. A static
K split chosen from (N, K) only is the invariant substitute and reaches cuBLAS at small M on narrow
layers, but its tree costs 5–12% at prefill sizes, so it is not a net win as a single kernel.

## Acknowledgements

Adapted from [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).
