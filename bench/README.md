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

> **On a CPU fallback this is slow.** A full `bench:check` takes 10-25 minutes on
> SwiftShader, because it prefills 638 tokens and then decodes 4 steps on a CPU
> rasteriser. Do not wrap it in a short `timeout`: a run that expires mid-way
> produces no result and looks like a failure. Budget 30 minutes, or wait for a
> real GPU.

## Correctness gate

An optimization that changes the model's output is a bug, not a speedup.

```sh
npm run bench:golden     # record the reference (do this once, on a known-good tree)
npm run bench:check      # compare against the reference; non-zero exit on failure
```

Two checks run, and both must pass:

- **prefill** — one pass over a fixed 11-token sequence.
- **decode** — a 638-token prefill, then 4 decode steps. `638` is deliberate: it
  pads the KV cache to 640 during prefill so the *first* decode step crosses the
  boundary and triggers a grow-and-repad, and it is past the 512-slot sliding
  window so the ring buffer is already wrapping.

The check runs a fixed token sequence, projects the final position to logits,
and compares a 64-point sample of the vocabulary against the golden. It is
tolerance-based, because GPU floating point is not bitwise reproducible across
shader rewrites — different summation orders legitimately change the last bits.

### The error metric

Error is normalised by the **scale of the logit vector** (the largest `|logit|`
in that step), not element-wise:

```
err = max over sampled entries of |actual - golden| / max|golden|
```

An element-wise relative error is the wrong metric here. Most of a vocabulary
sits near zero — one decode step has 12 of its 64 sampled logits under 0.5 and a
minimum of 0.0024 — so dividing by 0.0024 turns a `1e-5` absolute difference
into a `0.004` "relative error". That number is produced entirely by the
denominator, and it can fail a numerically sound change for no reason.

### Calibration

Tolerances are fractions of that scale: **0.001 for decode, 0.005 for prefill**
(`--tolerance` overrides both). Chosen from measured headroom:

| signal | magnitude (fraction of scale) |
| --- | --- |
| RoPE rewrite (exact transform) | **0.0** |
| attention rewrite (fusion, 7→4 kernels) | **1.3e-7** |
| **pass/fail threshold** | **1e-3** |
| a genuine 1% breakage | 1e-2 |

That is ~10,000× above observed noise and 10× below a real break. A looser
threshold leaves too little margin to be meaningful: at 0.005 a real break sat
only 2× above the line, which is a coin flip rather than a gate.

Verified against the recorded golden: identical output passes, a `1e-7`
perturbation passes, `1e-5` passes, a 1% scale error fails, a 0.05 absolute
shift fails, and corrupting the smallest 25% of logits by 2% fails.

### Argmax

Argmax must match, with one principled exception. When the golden's top two
tokens differ by less than the error the tolerance already permits
(`tolerance * |top1|`), a swap is expected from any change in summation order
and is counted as a tolerated near-tie rather than a failure. At 0.001 all four
decode steps' argmax comparisons are enforced.

A PASS on the normalised error alone is not the whole evidence. When reporting a
result, also give the **absolute max-abs-diff** and the **argmax match**, since
the normalised metric deliberately trades away sensitivity to small absolute
drift.

Goldens are stored per precision (`bench/goldens/f16.json`,
`bench/goldens/f32.json`) because the two paths round differently, and the
decode reference is `<precision>-decode.json`. A run whose precision does not
match the golden reports `skipped` rather than a false failure.

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

## Getting the GPU to actually work (Linux + NVIDIA)

Getting Chromium to use a real GPU rather than SwiftShader took four separate
fixes. Each was verified individually; any one missing silently leaves you on the
CPU fallback, and the harness still runs, so the failure is easy to miss.

1. **Device access.** `/dev/dri/renderD128` ships as `root:render 0660`. If your
   user is not in `render`, the driver cannot open the device. You do *not* need
   `usermod` + re-login if you are already in `video`: a udev rule that puts
   render nodes in the `video` group takes effect immediately and survives
   reboots.

   ```sh
   sudo sh -c 'echo "SUBSYSTEM==\"drm\", KERNEL==\"renderD*\", GROUP=\"video\", MODE=\"0660\"" \
     > /etc/udev/rules.d/99-render-video.rules'
   sudo udevadm trigger
   ```

2. **Drivers.** Without these there is no Vulkan driver to talk to at all.

   ```sh
   sudo apt install -y mesa-vulkan-drivers libvulkan1
   ```

3. **Chromium flags.** Headless Chromium does not select Vulkan unless told to.
   With a working driver and a readable device it still picks ANGLE's
   SwiftShader backend. `bench/chrome.mjs` passes
   `--enable-features=Vulkan --use-angle=vulkan --use-vulkan=native`. These are
   safe on a machine with no driver, because Chromium falls back by itself.

4. **NVK opt-in on Pascal and older.** Mesa's NVK refuses to probe those GPUs
   unless `NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1` is set, and reports zero devices
   without it. `chromeEnv()` sets it for the browser.

Verify with `node bench/preflight.mjs`. It prints the renderer string and the
`shader-f16` feature.

### Known limitation: no `shader-f16` on NVK/Pascal

NVK does not implement `shaderFloat16` on Pascal (confirmed in `vulkaninfo`, not
only through Chromium), so the FP16 weights cannot run natively there and the
harness widens them to float32. Consequences:

- Results are genuine GPU results, roughly 50-100x faster than the CPU fallback.
- But they are **f32, not f16**, so they move twice the bytes and are not
  identical to the production browser build. Decode is bandwidth-bound, so this
  is a real bias — expect absolute throughput to read lower than an f16 GPU
  would deliver.
- A GPU or driver that supports `shader-f16` (NVIDIA proprietary driver, modern
  AMD, Apple) avoids this entirely.

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
