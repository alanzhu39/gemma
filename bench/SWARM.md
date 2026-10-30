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
