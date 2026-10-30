# Optimization swarm: operating manual

This file is the shared contract for the agent team working on Gemma 3
inference performance. Read it fully before doing anything, and re-read the
"Every reported result" section before you post a number.

## Mission

Raise steady-state inference throughput on this machine's WebGPU backend,
measured by `bench/`, without changing what the model computes.

Targets (enforced by `bench/compare.mjs`):

| Workload | Target |
| --- | --- |
| `decode-ctx1024` | **2.0x** baseline tok/s |
| `prefill-1024` | **1.5x** baseline tok/s |
| everything else | **≥ 0.9x** baseline (no regressions) |

## Hard constraints

1. **WebGPU / web APIs only.** No native code, no WASM SIMD hacks that bypass
   the measured path, no server-side inference.
2. **The correctness gate must pass.** `npm run bench:check` is not optional.
   A change that alters logits beyond tolerance is a regression even if it is
   faster. Never edit a golden file to make a check pass.
3. **No benchmark tampering.** Do not modify `bench/bench.ts` measurement
   logic, `bench/workloads.json`, or `bench/compare.mjs` thresholds to make
   numbers look better. If a workload genuinely needs to change, that is a
   discussion in `#perf`, not a local edit.
4. **One concern per branch.** Small, reviewable, revertable changes.
5. **Patch the library in-repo if needed.** Custom kernels patching
   `@jax-js/jax` are allowed and expected; keep the patch in `patches/` applied
   reproducibly (see below) so it survives `npm install`.

## Where the time goes

Before optimizing, profile. The engine (`src/lib/gemma3/model.ts`) is a
sequence of `jit`-compiled functions: embeddings, per-layer RMSNorm, QKV
projections, RoPE, attention (einsum + softmax), output projection, MLP
(gate/up/down with GELU). Known cost centres, in rough order of suspicion:

- **Decode**: 18 layers × several small kernels each. At batch 1 the matmuls
  are memory-bound and the GPU is idle most of the time; launch overhead and
  per-step host↔device synchronisation dominate.
- **Attention**: `einsum` + `softmax` materialise an `[S, 4, N]` score tensor.
  A fused flash-attention kernel avoids ever writing it out.
- **Prefill**: the `[S, S]` mask/score materialisation and the KV-cache
  `np.where` scatter are expensive, and the whole KV cache is padded on growth.
- **Repeated work**: masks, slot positions, and RoPE tables are recomputed per
  step; sampling forces a host readback every token.

## Measurement protocol

Run everything from the repo root.

```sh
npm run bench -- --label <your-agent>-<short-description>
node bench/compare.mjs --json
npm run bench:check
```

Rules:

- **Never report a single run.** Use at least `--repeats 5` (the default).
  Compare medians, and mention min/p90 if they are interesting.
- **Always state the precision and renderer.** If
  `sysInfo.isSoftwareRenderer` is true, every absolute number is meaningless
  for GPU work; say "CPU fallback, relative only" in the same sentence as the
  number.
- **Baseline first.** Compare against `bench/baseline.json`, not against a
  number you remember from earlier.
- **Noise budget.** Treat anything under ~5% as noise unless the min/p90 range
  says otherwise. Re-run before claiming a win.
- **Attach the result JSON** to your message (`swarmchat send ... --attach`).
  Chat claims without a result file are not accepted.

## Coordination protocol

Use SwarmChat. The channel is the shared state; do not rely on private memory.

- `#perf` — design discussion, hypotheses, "I am taking X", questions.
- `#results` — accepted numbers and verdicts only, one message per result.

Before starting work, claim it:

```sh
swarmchat send "#perf" "Claiming: fused attention kernel (affects model.ts runAttention)"
```

If someone else has claimed the same file/function, coordinate rather than
racing. Duplicate work is the main way this swarm wastes time.

When you finish, post to `#results` with: what changed, the before/after
numbers, the correctness result, and the result JSON attached.

## Roles

Keep the team small and the interfaces tight.

- **orchestrator** — owns the target, the baseline, and `#results`. Decides
  what ships. Does not write kernels; resolves conflicts and keeps the branch
  green.
- **profiler** — establishes where time actually goes. Produces a ranked
  bottleneck list with numbers, and a trace/measurement method others can
  reproduce. Owns `bench/scripts/` profiling helpers.
- **kernel** — writes custom WebGPU kernels / jax-js patches (attention fusion,
  matmul tiling, norm fusion). Owns `src/lib/webgpu/` and the patch.
- **integrator** — rewires `src/lib/gemma3/*` to use the fast paths, removes
  redundant per-step work (recomputed masks, host readbacks), and keeps the
  correctness gate passing.

Suggested first split: profiler produces the bottleneck list; kernel and
integrator work in parallel once `#perf` has agreed on the top two items.

## Maintaining a library patch

Custom kernels need to reach the WebGPU backend. Preferred order:

1. **Express it in jax-js primitives first.** Fusion often gets you most of the
   way with zero maintenance burden.
2. **Patch `@jax-js/jax` in-repo** when a real custom kernel is needed. Vendor
   the patch under `patches/` and apply it from `postinstall` so a fresh
   `npm install` reproduces the tree. Keep the patch small and rebased on the
   pinned version; document what it changes and why at the top of the file.

If you vendor the package instead of patching, put the reason in `#perf` — it
makes future upgrades much harder and needs a conscious decision.

## Definition of done

A change is done when all of these are true:

- [ ] `npm run bench:check` passes at the target precision.
- [ ] A benchmark run at `--repeats 5` (or more) shows the improvement, with
      the result JSON attached to `#results`.
- [ ] `node bench/compare.mjs --gate` exits 0.
- [ ] No workload dropped below the 0.9x regression floor.
- [ ] The change is committed in small commits with messages saying what the
      number was before and after.
- [ ] Any library patch is reproducible from a clean `npm install`.

## Worktree setup (read this first)

Each agent has a git worktree under `/tmp/gemma-swarm/<role>` on branch
`swarm/<role>`. The worktree shares the main checkout's `node_modules`,
`.svelte-kit`, and `bench/.cache` through symlinks.

**Do not run `npm install` or `npm ci`.** It replaces the symlink with a real
1 GB tree, wastes time and disk, and can pick up dependency versions the rest of
the team is not running. If something genuinely needs installing, say so in
`#perf` first and let the orchestrator decide.

If `node bench/preflight.mjs` fails with a missing module, the symlink is
broken — tell the orchestrator rather than reinstalling.

## Current environment status

Check `node bench/preflight.mjs` before trusting any number.

- **GPU access:** the host GPU (`/dev/dri/renderD128`) is `root:render`. If your
  user is not in the `render` group, Chromium cannot see it and falls back to
  SwiftShader, a CPU rasteriser.
- **Consequence:** on the fallback, absolute throughput is CPU performance.
  Only relative comparisons against another run at the same precision mean
  anything, and a full suite takes hours. Prefer tiny scopes
  (`--only prefill-128`)
  while iterating.
- **`shader-f16`:** SwiftShader lacks it, so the harness widens weights to
  float32 and records `precision: "f32"`. Results from f32 and f16 are not
  comparable, and `bench/compare.mjs` refuses to mix them.
- **Baseline:** `bench/baseline.json` must be recorded on the real GPU. Do not
  record one while on the CPU fallback; it would bake CPU numbers in as the
  target and make the gate meaningless.

## How a change lands

Work happens on each agent's own branch (`swarm/<role>`). Nothing is merged
straight into `perf` by the agent that wrote it.

1. **Claim** the work in `#perf`.
2. **Build** it on `swarm/<role>`, one concern per commit.
3. **Prove** it: `npm run bench:check` passes, plus a benchmark run at
   `--repeats 5` if the change is meant to be a speedup.
4. **Post** the result to `#results` with the JSON attached and the before/after
   numbers.
5. **Orchestrator reviews and merges** into `perf`. If it is a speedup, the
   orchestrator re-runs the comparison against `bench/baseline.json` before
   accepting; an agent's own measurement is a claim, not a verified result.

Conflicts are resolved by the orchestrator. Two agents editing the same function
must agree in `#perf` first, in writing, before either starts.

## How to report a measurement

Learned the hard way: an early claim from this swarm was directionally right and
factually wrong in its headline number, and another was right in symptom but
wrong in cause. Both cost the team more than saying less would have.

Every reported number should make these distinctions explicit:

1. **Measured vs inferred.** Say which. "1 compile per step, measured from the
   profiler artifact" is different from "21 compiles, inferred from 21
   dispatches carrying that shader". The second was wrong; the first was right.
2. **Mechanism vs magnitude.** "Passing `position` as a device scalar removes the
   per-step recompile" (mechanism, provable with a small probe) is a different
   claim from "this saves X ms/token" (magnitude, needs a real run). Publish the
   mechanism when you have it; do not let it imply the magnitude.
3. **Run the control before publishing a strong claim.** If you think N distinct
   inputs cause growth, also run N identical inputs. The first version of the
   "unbounded cache" claim survived exactly until the control was run, which
   showed memory did not scale with distinct positions at all.
4. **Retract explicitly.** If a claim turns out wrong, say so in the same channel
   with the corrected version. A quietly abandoned claim is worse than a public
   retraction, because teammates are still building on it.
5. **Name what the measurement cannot see.** A counter that is invisible on the
   CPU fallback (e.g. the WebGPU shader pipeline cache) must be labelled
   "source inspection, unmeasured" rather than folded into a measured result.
6. **Never reason about an aborted run.** If a run crashed, timed out, or was
   killed, it produced no result. Re-run it. Do not infer what it "would have"
   shown.
