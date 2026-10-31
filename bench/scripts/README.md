# Diagnostic probes

Standalone measurement tools, each isolating one question. Every one takes
`bench/lock.mjs` before touching the GPU.

| probe | question |
| --- | --- |
| `bandwidth-probe.ts` | Sustained read vs copy bandwidth |
| `gemv-compare.ts` | Which GEMV access pattern wins at the real working set |
| `probe-transpose-cost.ts` | Is `weight.transpose()` a strided read? (the 3.8x) |
| `probe-projection-real.ts` | Why did probes disagree by 2.8x (`np.ones` elision) |
| `probe-decode-breakdown.ts` | Full per-phase breakdown, timestamp-query based |
| `probe-final-ab.ts` | Interleaved A/B of the candidate fixes |
| `probe-post-fix.ts` | Re-ranking after the projection fix |
| `dispatch-cost.ts` | Cost per dispatch, in the shape actually run |
| `submit-shape.ts` | Does batching submits help with real work between |
| `submit-hazard.ts` | What happens if a buffer is destroyed before its submit |
| `gpu-error-detect.ts` | Does the gate's error detector actually fire |

## Rules, each learned by getting it wrong

1. **Force completion.** Read a value back. Without it the work is elided and you
   measure nothing.
2. **Reject anything above hardware peak** (192 GB/s here). Always a probe bug.
3. **`np.ones(...)` is a compile-time literal.** A kernel consuming it folds the
   input away and never reads the weight. This single mistake produced the 2.8x
   disagreement between two probes (27 ms vs 73 ms) and hid 43 ms of projection
   from the trace. Use a real host buffer: `np.array([...])`.
4. **Request raised device limits**, or large buffers silently no-op.
5. **Warm up three times.** One pass is not enough for a 640 MiB tensor to be
   resident.
6. **Measure the shape you actually run.** Batched vs per-submit differs 8x;
   160 MiB vs 640 MiB GEMV differs 2.4x.
7. **State the shape with the number.**
