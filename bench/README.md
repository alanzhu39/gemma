# Benchmarking Gemma 3 inference

Everything here exists so that a change can be judged with a number instead of
an opinion. Read this before proposing an optimization.

## Quick start

```sh
npm run bench:setup      # one-time: Chromium + its system libs (no root needed)
npm run bench:fetch      # one-time: model weights into bench/.cache
npm run bench            # run the full suite
npm run bench:check      # verify the model still computes the right thing
```

## What is measured

Workloads are fixed and pre-tokenized in `bench/workloads.json`, so a run never
depends on the network or on a tokenizer version. Regenerate them with
`npm run bench:workloads` only if the workload design intentionally changes.

| Workload | What it measures |
| --- | --- |
| `prefill-128` / `prefill-512` / `prefill-1024` | Time to process a prompt of N tokens in one pass. Reported as tokens/second. |
| `decode-ctx128` / `decode-ctx1024` | Steady-state per-token latency while generating from an N-token context. Reported as tokens/second (1 / latency). |
| `batch-1/2/4` | Aggregate goodput with B independent conversations advanced in lockstep, dispatched before syncing. |

### Why decode is reported as latency

A decode step processes exactly one token. Throughput in tokens/second is just
the inverse of per-token latency, and latency is what a user feels. Quoting
"tokens/second" for decode can hide that a change made every token slower but
batched more of them together.

### Why there is a batch workload

Single-sequence decode leaves the GPU mostly idle: one token is far too little
work to fill it. The `batch-*` workloads measure how well the engine amortises
that away. Note that today these run the existing single-sequence path for B
conversations concurrently; they do **not** exercise a true batched matmul. If
a real batched engine path is added, `batchSupported` in the results will flip
to 1 and `batchedTotalTps` becomes meaningful.

## Measurement rules

These are enforced by `bench/bench.ts` and should not be relaxed:

1. **Synchronise inside the timed region.** Each sample calls
   `blockUntilReady` and reads one element back, so queued GPU work cannot leak
   into the next sample and make the previous one look fast.
2. **Discard warmup.** Shader compilation and KV-cache growth are one-time
   costs. Warmup iterations run before measurement and are thrown away.
3. **Report the median.** Mean is skewed by GC pauses and scheduler noise. The
   median, min, and p90 are all recorded in the JSON.
4. **Identical work per repeat.** Decode uses a deterministic synthetic token
   sequence, so every repeat does the same amount of work and results are
   comparable across runs.

## Correctness gate

An optimization that changes the model's output is a bug, not a speedup.

```sh
npm run bench:golden     # record the reference (do this once, on a known-good tree)
npm run bench:check      # compare against the reference; non-zero exit on failure
```

The check runs a fixed token sequence, projects the final position to logits,
and compares a 64-point sample of the vocabulary against the golden file. It is
tolerance-based, because GPU floating point is not bitwise reproducible across
shader rewrites — different summation orders legitimately change the last bits.
What must hold: all logits finite, logits not collapsed, argmax unchanged, and
relative error under the tolerance (default 2%).

Goldens are stored per precision (`bench/goldens/f16.json`,
`bench/goldens/f32.json`) because the two paths round differently. A run whose
precision does not match the golden reports `skipped` rather than a false
failure.

## Comparing against a baseline

```sh
npm run bench -- --label baseline
cp bench/results/baseline-*.json bench/baseline.json
npm run bench
node bench/compare.mjs            # human-readable diff
node bench/compare.mjs --gate     # non-zero exit unless the target is met
```

`bench/compare.mjs` refuses to compare runs made at different precision, so a
CPU fallback can never be mistaken for a GPU result.

## The performance target

Defined in `bench/compare.mjs` as `TARGET`:

- **decode-ctx1024: 2.0x** baseline throughput
- **prefill-1024: 1.5x** baseline throughput
- **no workload below 0.9x** baseline (regression floor)

Change these numbers in one place, and update `bench/baseline.json` when a new
target epoch starts.

## Hardware notes

The harness runs headless Chromium. On a machine without access to a real GPU,
Chromium falls back to SwiftShader, which:

- is a **CPU** renderer, so numbers are 100-1000x slower than a GPU and say
  nothing about GPU performance;
- lacks the `shader-f16` feature the FP16 weights need, so the harness widens
  the weights to float32.

Both facts are recorded in every result (`sysInfo.isSoftwareRenderer`,
`sysInfo.precision`) and printed as a warning. Results from a software renderer
are only valid for *relative* comparisons, and only against other software
renderer results at the same precision.

`bench/setup-chrome-deps.sh` exists because this box has no passwordless sudo.
It downloads the `.deb` packages Chromium needs and unpacks them under
`bench/.cache/chrome-deps`; `bench/run.mjs` and `bench/check.mjs` put that
directory on `LD_LIBRARY_PATH`. If Chromium ever fails to launch with a missing
`lib*.so`, that script is the fix.
