# The committed baseline is void

`bench/baseline.json` was recorded on the dev machine's GTX 1060 through NVK.
That adapter is not usable for performance work, so the numbers in it do not
describe this engine.

## Why: nouveau cannot reclock Pascal

Nouveau's own power-management support matrix lists, for Pascal (GP106):

| feature | status |
| --- | --- |
| Engine reclocking | **TODO** |
| Memory reclocking | **TODO** |
| Automatic reclocking | **TODO** |
| Voltage adjusting | **TODO** |

Without PMU firmware the card never leaves its boot clocks — around **139 MHz**
core against a **1709 MHz** boost clock, roughly 8% of what the hardware can do.
Memory also never reclocks. This is a documented, long-standing nouveau
limitation on Pascal, not a fault in this machine.

That explains why everything measured low by a similar factor, rather than one
subsystem being slow.

Measured on the same adapter, same session:

| operation | measured | GTX 1060 nominal | fraction |
| --- | --- | --- | --- |
| elementwise mul | 9.4 GB/s | ~192 GB/s | 5% |
| reduction | 1.5 GB/s | ~192 GB/s | 0.8% |
| compute (FMA loop) | ~100-151 GFLOPS | ~4370 GFLOPS | 2-3% |

Memory and compute are both crippled by a similar factor, so this is the driver
or device configuration, not our kernels. Mesa gates Pascal support behind
`NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1`, which is an accurate label.

Suspects that were tested and eliminated:

- **Transpose**: strided `W^T` 461 ms vs a contiguous transposed copy 463 ms.
- **Arithmetic intensity**: M=1 batched to M=4 gave 462 vs 463 ms.
- **Reduction pattern**: plain elementwise over the same buffer is also ~9 GB/s.

## What is still trustworthy here

- Dispatch, workgroup, allocation and readback counts (structural, hardware-independent).
- Byte-traffic accounting derived from tensor shapes.
- Correctness: `npm run bench:check`. Note NVK on Pascal lacks `shader-f16`, so
  local runs take the f32 path and cannot validate f16 code at all.
- Anything measured on the software path, for relative comparison only.

## What is not

Any ms, tok/s, GB/s or speedup figure. Those describe NVK on Pascal.

## To produce a real baseline

Run `npm run bench -- --label baseline` on hardware with a working Vulkan driver
and `shader-f16` (NVIDIA proprietary, modern AMD, Apple, or Intel). Then confirm
`sysInfo.rendererString` names the real GPU and `sysInfo.precision` is `f16`,
and sanity-check achieved GB/s against the card's nominal bandwidth before
trusting any of it.

## A better result: the production design is already right

While invalidating the baseline, an audit of jax-js 0.1.11's dtype rules turned
up something worth recording, because it means the fp16-storage / fp32-compute
design **already does what it was intended to do**.

`promoteTypes` (`backend-DZvR7mZV.js:375`) promotes `float16 < float32` to
float32, and `dot` lowers to a multiply plus a reduce whose accumulator uses the
*promoted* dtype (`index.js:2227`). But the promotion is emitted as an in-shader
`AluOp.Cast` on the loaded value, not as a separate conversion pass:

```wgsl
var<storage, read> w : array<f16>;   // weight stays fp16 in memory
... acc += f32(w[k]) * x[...];       // widened in register, f32 accumulate
```

There is even an explicit f16→f32 accumulator promotion for the pure-f16 case
(`backend-DZvR7mZV.js:1373`), which is JAX's mixed-precision behaviour exactly.

So on any adapter with `shader-f16`: weights are read as fp16 (half the bytes),
widened in-register, and accumulated in fp32. No 320 MiB read plus 640 MiB write
per projection — that scenario does not occur.

Two consequences:

1. **Memory traffic for weights is already halved** relative to an f32-store
   design. Since batch-1 decode is bandwidth-bound, that is the single biggest
   win available, and it is already in place.
2. **`shader-f16` is mandatory, not optional.** Without it the shader *cannot*
   declare `array<f16>`, so jax-js throws rather than falling back. Devices
   without the feature (and this dev box, on NVK/Pascal) simply cannot run the
   fp16 path. `bench/f32-model.ts` exists only to keep the harness usable there;
   it is not representative of production.
