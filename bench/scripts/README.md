# Diagnostic probes

Standalone measurement tools. Each isolates one question and reports numbers with
the shape they were measured in.

| probe | question |
| --- | --- |
| `bandwidth-probe.ts` | What read / copy bandwidth can this card sustain? |
| `gemv-compare.ts` | Which GEMV access pattern wins, at the real working-set size? |
| `probe-sampling.ts` | How much of a step is token sampling? |
| `probe-step-phases.ts` | Where does a real decode step spend its time? |
| `probe-gemv-variants.ts` | Which part of the projection is slow? |
| `gpu-error-detect.ts` | Does the gate's error detector actually fire? |
| `dispatch-cost.ts` | What does one dispatch cost, in the shape we actually run? |
| `submit-shape.ts` | Does batching submits help with real work between them? |
| `submit-hazard.ts` | What happens if a buffer is destroyed before its submit? |

## Rules these all follow, learned the hard way

1. **Force completion.** Read a value back out. Without it jax-js never dispatches
   and you measure nothing — one probe reported 1973 GB/s on a 192 GB/s card.
2. **Reject anything above hardware peak.** It is always a bug, never a fast kernel.
3. **Request raised device limits.** A bare `requestDevice()` gives 256 MB and
   16 KiB workgroup storage; larger working sets then silently do nothing.
4. **Warm up properly.** One warm pass is not enough for a 640 MiB tensor to be
   resident. A single-warmup probe once made a bare `np.dot` appear 2.6x faster
   than the jit'd version; with three warmups they are identical (~70 ms).
5. **Measure the shape you actually run.** Batched-submit cost and per-submit cost
   differ by 8x. 160 MiB and 640 MiB GEMV differ by 2.4x.
6. **Verify output before timing.** An unverified kernel can be elided.
7. **State the shape with the number.** A figure without its configuration is not
   reproducible.
