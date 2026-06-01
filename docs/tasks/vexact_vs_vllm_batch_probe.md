# VeXact vs vllm: where is it slow? — per-step batch-composition probe

**Date**: 2026-05-30
**Branch**: `feat/switch_mode`
**GPU**: worker-0 (`942015`), 8× H100-SXM-80GB
**Tooling**: `benchmarks/batch_probe/` (`run_matrix.sh` + `vexact_probe.py` / `vllm_probe.py` + `analyze.py`)

## One-line conclusion

VeXact is slow because **prefill admission only lets in 1 sequence at a time**: the
scheduler admits at most **1** new prefill per step, so batches are never filled and
VeXact runs **22% more** forward steps than vllm. Those extra forwards account for
essentially the entire throughput gap against vllm.

The root cause is a single config default: `max_num_prefill_seqs = 1` at
`vexact/config.py:240`.

## Experiment setup

| Item             | Value                                                                            |
| ---------------- | -------------------------------------------------------------------------------- |
| Model            | `Qwen3-1.7B` (`/mnt/hdfs/neiwen/models/Qwen3-1.7B`)                              |
| Dataset          | `ShareGPT_V3_unfiltered_cleaned_split.json`                                      |
| Requests         | 256                                                                              |
| Scheduler params | `max_num_batched_tokens=2048`, `max_num_seqs=512`, `max_model_len=2048`          |
| Matrix           | 2 engines × 2 chunked × 2 eager = 8 probes, each on its own GPU, run in parallel |

The probe does not measure throughput. Instead it captures, for **every scheduler step,
the actual batch composition that was scheduled** (prefill / decode seq counts and token
counts), writes it as JSONL, and compares with `analyze.py`. On the VeXact side the
recording is gated by the `VEXACT_BATCH_PROBE` env var in `vexact/core/scheduler.py`; on
the vllm side we monkeypatch `Scheduler.schedule` to record with the **identical rule**
(`tokens==1 ⇒ decode`, otherwise prefill), making it an apples-to-apples comparison.

Reproduce:

```bash
bash benchmarks/batch_probe/run_matrix.sh   # on the 8-GPU worker
python benchmarks/batch_probe/analyze.py benchmarks/batch_probe/results
```

## Core data

The **total work scheduled by the two engines is identical** (both 110808 tokens, 256
prompts, prefill/decode = 55265/55543 ≈ 49.9%/50.1%), so the batch shapes can be compared
directly:

| Metric                              | VeXact               | vllm                   | Meaning                                           |
| ----------------------------------- | -------------------- | ---------------------- | ------------------------------------------------- |
| **Total scheduler steps**           | **971**              | **795**                | VeXact runs **22% (+176) more** forwards          |
| Steps containing prefill            | **256**              | **29**                 | VeXact gives each prompt its own step             |
| Max prefills in one step            | **1**                | **16**                 | ← the core difference                             |
| Avg prefill tokens per prefill-step | 216                  | **1906**               | VeXact uses ~10% of the budget; vllm fills 93%    |
| Max tokens/step                     | 1054 (51% of budget) | **2048 (budget full)** | even VeXact's busiest batch is only half full     |
| Mean tokens/step                    | 114                  | 139                    |                                                   |
| Mean decode seqs/step               | 57                   | 70                     | prompts enter slowly, so fewer concurrent decodes |
| Max seqs/step                       | 142                  | 228                    |                                                   |

VeXact `prefill_seqs/step` distribution: `{0: 715 steps, 1: 256 steps}` — never exceeds 1.
vllm `prefill_seqs/step` distribution: `{0: 766 steps, the rest 7–16 per step}`.

## Root-cause analysis

`vexact/config.py:240`

```python
max_num_prefill_seqs: int = field(default=1, ...)
```

In `schedule()` of `vexact/core/scheduler.py` this value becomes `prefill_seqs_budget`,
which caps the admission rate in the new-request admission loop:

- Line 164: `prefill_seqs_budget = self._max_num_prefill_seqs` (= 1)
- Line 180: in-flight prefills consume the budget first
- Line 193 / 215: `while ... and prefill_seqs_budget > 0`, decremented by 1 per admitted prefill

So the 256 prompts are fed into the running pool one at a time, "toothpaste" style, taking
256 separate prefill steps. vllm, by contrast, constrains prefill with only the single
`max_num_batched_tokens` token budget, packs many prefills into the 2048-token window, and
drains all of them in just 29 steps.

### Why it is currently fixed to 1

`max_num_prefill_seqs = 1` **is not an oversight — it is a deliberate constraint right now:
VeXact's CUDA graph does not yet support prefill.** A CUDA graph requires every captured
batch to have a fixed shape, whereas prefill's query length varies with the prompt and is
dynamic, so it cannot be folded into a captured graph. The current execution model is
therefore:

- **decode** (1 token each, regular shape) runs through the CUDA graph;
- **prefill** can only run in eager.

Limiting each step to 1 prefill keeps "steps that contain prefill" simple and controllable,
avoiding the complexity of mixing variable-length prefill with the graph path. The cost is
exactly what we see above: prefill admission is throttled hard, batches are not filled, and
the forward-step count is inflated. This also explains the corroborating evidence below —
why the chunk/eager toggles have no effect on scheduling: the admission bottleneck is
determined entirely by this one constraint, independent of whether the graph is on.

The consequence is a causal chain:

1. Slow prefill admission → the running pool accumulates concurrent requests slowly → smaller decode batches (57 vs 70);
1. Each prefill step carries only ~216 tokens, leaving half the GPU idle → more steps needed for the same total;
1. The 176 extra forwards each pay kernel-launch / CUDA-graph-replay / Python scheduling overhead.

The step ratio `971 / 795 = 1.22×` lines up with the existing throughput report
(`benchmarks/batch_invariance_results.md`): VeXact-invariant 27.66s vs vllm-invariant 23.64s
(**1.17× slower**). **The extra forward steps account for essentially the entire gap between
VeXact and vllm.**

## A clean corroboration

VeXact's 4 configs (the full chunk × eager cross product) produce **byte-for-byte identical**
output (971 steps, max=1054, identical histogram), showing that the `enable_chunked_prefill`
and `enforce_eager` toggles have no effect on scheduling whatsoever. The bottleneck is
neither chunking nor CUDA graph — it is purely the **prefill admission rate**.

Reason: this workload's prompts are short enough (prefill mean 216 tokens) that chunked
prefill's `min(budget, remaining)` branch is effectively the whole prefill, so chunking
never actually triggers.

## Recommendation

The ideal end state is to constrain prefill admission with only the single
`max_num_batched_tokens` token budget, like vllm, so a single step can pack multiple
prefills to fill the 2048 budget and the step count drops from 971 toward vllm's ~795.

But because the source of `max_num_prefill_seqs = 1` is that **CUDA graph does not support
variable-length prefill**, lifting it requires solving the execution side first, in two
steps:

1. **Short term (without touching the graph)**: raise `max_num_prefill_seqs` on the
    `enforce_eager` (CUDA-graph-off) path. Prefill already runs eager there, so packing
    multiple prefills has no graph-shape constraint, and we can directly measure the upper
    bound of the "fill the batch ⇒ fewer steps" gain.
1. **Long term (touching the graph)**: make CUDA graph support prefill (e.g. bucket by token
    count and pad to fixed shapes, or use different capture strategies for prefill vs decode),
    then open up admission with the graph on as well.

Next step: using the same `benchmarks/batch_probe` harness, raise `max_num_prefill_seqs` on
the eager path and re-run to verify the step count drops from 971 toward ~795 as expected.
