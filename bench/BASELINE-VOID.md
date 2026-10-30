# The committed baseline is void

`bench/baseline.json` was recorded on the dev machine's GTX 1060 through NVK.
That adapter is not usable for performance work, so the numbers in it do not
describe this engine.

## Why

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
