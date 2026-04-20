# Benchmarks

Throughput benchmarks for the VeXact inference engine on the ShareGPT dataset.

> **Note:** VeXact currently supports text-only input. `DriverRequest` only
> carries `input_ids_list`, so even when `--include-multimodal` is set, the
> image/video payloads attached to samples are not forwarded to the engine —
> only the prompt text is tokenized and sent.

## Layout

```
benchmarks/
├── dataset.py      # ShareGPT loader (text + image/video), adapted from vLLM
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
source /mlx_devbox/users/neiwen.ling/playground/modelchef/.venv/bin/activate

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
> `throughput.py` to set `enforce_eager=True` on `ModelConfig`, otherwise the
> run will fail.

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

## Acknowledgements

Adapted from [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).
