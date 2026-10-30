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

## What the profile actually says (as of the first full analysis)

Structural counts are hardware-independent and are the ones to trust on the CPU
fallback. Wall-clock on SwiftShader is not.

Decode at ctx-1024: **723 dispatches**, 39 per layer (702 of them), plus 21
step-level. Two orderings disagree, and the disagreement is the finding:

Measured per-dispatch and per-byte cost by model region (decode, ctx 1024, a
**quiet** run — `rank.mjs` reports `other-process utilisation 0.47`):

| region | dispatches | share | MiB | note |
| --- | --- | --- | --- | --- |
| norm | 218 | 30.2% | 1.1 | pure launch overhead |
| RoPE | 114 | 15.8% | 0.4 | pure launch overhead |
| attention core | 110 | 15.2% | 23.4 | |
| KV write | 72 | 10.0% | 87.3 | now row-write |
| QKV proj | 54 | 7.5% | 67.7 | |
| MLP | 54 | 7.5% | 270.7 | |
| mask/slots | 40 | 5.5% | 0.2 | rebuilt per step |
| embed | 2 | 0.3% | 640.0 | the accepted wall |

The inversion is the point: **norm and RoPE are 46% of dispatches for 1.5 MiB of
traffic**, while embed is 0.3% of dispatches for 640 MiB. Optimise those two
regimes differently — launch count for the first, bytes for the second.

| by wall-clock | | by dispatch count | |
| --- | --- | --- | --- |
| logits matvec | 22.2% | norm | 30.2% |
| KV write | 20.4% | RoPE | 15.8% |
| MLP | 19.4% | MLP | ~7.5% |
| attention core | 13.4% | attention | ~15% |

**84% of dispatches use ≤4 workgroups and consume 55.5% of kernel time.** At
batch 1 this engine is *launch-bound*, not FLOP-bound. That is why optimising
the biggest wall-clock line item alone will disappoint: the win is in fewer,
larger dispatches.

Other measured facts worth not re-deriving:

- KV write is **1869× amplified**: 36 KiB of real change against 65.7 MiB of
  bound work, because `np.where` runs over the whole cache capacity.
- Position-only tables (masks, slots, RoPE) are rebuilt every step: ~154
  dispatches, ~21% of launches.
- One `createBuffer` + `destroy` per dispatch, 46.1 MiB per decode step.
- Host readback is **not** a problem: 1 per step. The earlier suspicion that
  sampling serialised the pipeline was wrong.
- Pad-on-growth is correctly confined to warmup.

Prefill-1024: MLP 41.8%, attention 39.1%, QKV 10.3%. Attention binds 2666 MiB =
48% of bytes. Decomposed, **QK is 2.00× PV at identical FLOPs**, purely from
writing the score tensor. So restructuring helps PV, but QK's cost is only
removable by a fused kernel — that is the argument for P5, measured rather than
asserted.

**Caveat that must travel with these numbers:** the ms ordering above is CPU
fallback and was contended. Only the dispatch and byte rankings are exact.

## The memory-bandwidth floor (read this before setting expectations)

Decode at batch 1 is a streaming problem: every token must read the weights.
From the actual safetensors shapes, at f16, with ctx 1024:

All bandwidths below are **decimal GB/s**, which is NVIDIA's convention for the
1060's 192 GB/s. Mixing GiB/s and GB/s shifts these figures by ~7%, which is
enough to matter if the number is used to retarget a goal.
| --- | --- | --- |
| **logits matvec (tied embedding, 262144x640)** | **320.0** | **61%** |
| 18 decoder layers of weights | 191.4 | 37% |
| KV cache reads (15 sliding @512, 3 global @1024) | 10.5 | 2% |
| **total** | **521.9** | |

So two thirds of every decode step is one matrix-vector product against the
vocabulary table. That has consequences:

- **A bandwidth floor exists and is not negotiable by kernel work.** On a
  GTX 1060 (192 GB/s) the floor is ~2.97 ms/token ≈ **351 tok/s**. At 1 TB/s it
  is ~1827 tok/s. No amount of fusion, dispatch batching or tiling beats it; the
  kernel already reads its own weights exactly once in one dispatch.
- **The floor is a ceiling, not a forecast.** A launch-bound engine lands far
  below it. For calibration: a 270M f16 model on a 1060 realistically manages
  ~30-40 tok/s, roughly 10% of the floor. If the measured baseline is ~35, then
  2x = 70 is only 20% of the floor and is reachable by kernel work alone. So the
  floor does not threaten the target — but `baseline / floor` must still be
  computed on the first real GPU run before anyone commits to the number.
- **Check the target against the floor before chasing it.** A 2x decode target is
  reachable only if the measured baseline sits meaningfully below the floor for
  that GPU. If the baseline is already close, 2x is arithmetically impossible
  without reducing bytes moved.
- **The one real lever is quantisation of the tied embedding.** int8 halves the
  table to 160 MiB, int4 to 80 MiB. Note that quantising the embedding *alone*
  is not enough to double the floor: int4-embed gives ~650 tok/s against a 702
  tok/s requirement for 2x of the f16 floor. Quantising layer weights too is what
  clears it. Because the embedding is tied, quantising the
  output head also quantises the input gather, which reads a single row and does
  not care how wide the table is — nearly all the saving lands on the matvec.
  This is a correctness/golden task before it is a performance task: it changes
  logits, and the gate requires a stable argmax.
- **There is no safe shortcut on the vocabulary.** Exact argmax needs all 262144
  dot products; a decaying-vocabulary or approximate search changes the tail
  winner and would silently degrade output.

Implication for the swarm: do not spend kernel cycles on the logits matvec. It is
an accepted wall unless someone takes on embedding quantisation with a gate
measurement.
