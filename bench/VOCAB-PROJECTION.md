# Vocabulary projection: the kernel works, and it ties a one-line fix

Measured on the GTX 1060 (f32 weights, official harness, readback forced).

## What was built

`src/lib/gemma3/vocab-projection.ts`: a warp-per-row GEMV. The 32 lanes of a
subgroup cooperate on one output row, each striding k by 32, reduced with
`subgroupShuffleXor`. This needs `subgroups` as a *device* feature, which
jax-js did not request, so `patches/jax-js-subgroups.patch` adds it.

## What it achieves

Isolated on resident data, 15 reps, minimum over reps:

| variant | ms | GB/s |
| --- | --- | --- |
| strided `np.dot` (what the code did) | 68.5 | 9.8 |
| contiguous `np.dot` (`tokenEmbedT`) | 5.3 | 126.6 |
| warp-per-row kernel | 5.2 | 129.1 |

Read ceiling on this card is 101-140 GB/s, so both fast paths are at the wall.

Whole model, `decode-ctx1024`:

| | tok/s | ms/token |
| --- | --- | --- |
| `baseline.json` (older tree) | 8.29 | 120.7 |
| stock, contiguous copy | 18.65 | 53.6 |
| warp-per-row kernel | 18.34 | 54.5 |

## The finding

**The kernel does not beat the contiguous-copy fix. It ties it.** Interleaved
A/B, 9 rounds, order rotated every round: `kernelVsStockT` median **0.98**, and
two whole-suite runs agree (18.34 vs 18.65 tok/s).

That is not a failure of the kernel. It is that both changes remove the *same*
deficit by different means:

- the stock path read the embedding as a *strided view* of a [vocab, dim]
  matrix, 2560 B between consecutive elements, so every 128 B line delivered
  4 B. Materialising a contiguous [dim, vocab] copy once at load fixes that.
- the kernel reads the same 640 MiB with fully coalesced row-major loads.

Once the bytes arrive coalesced, the card is at its read ceiling and there is
nothing left to win. The two are not additive, and no amount of further tuning
of the kernel can help: 126-129 GB/s IS the wall.

Against the recorded baseline both are ~2.2x, but that baseline predates other
merged work, so it is not a clean control for this change.

## Which to keep

The contiguous copy is ~10 lines, needs no patch to a dependency, and cannot
subtly break. The kernel is ~250 lines, needs a library patch and a device
feature, and ties it. **Recommendation: keep the copy, and keep the kernel only
as the record of what the access pattern was worth** - unless the f16 path (not
testable here: this adapter has no `shader-f16`) behaves differently, which is
untested.

The kernel is nevertheless correct and verified: 0 bad rows over all 262144,
gate PASSES with decode norm err 5.87e-6 and argmax 4/4 enforced, and a probe
confirms the production path really runs it (91% of elements differ bitwise from
`np.dot` while the error stays at 1.15e-6).

## Measurement caveat

This box runs at 95-99% external CPU load. One variant's per-round times ranged
39-219 ms for identical work, so medians cannot resolve a 5% difference on a
whole-step measurement. The isolated table uses the **minimum** over reps
(interference only adds time, so the minimum is the least disturbed run); the
whole-step claim rests on two independent whole-suite runs agreeing.

## The memory cost of the fix

The contiguous copy is a second copy of the tied embedding. That is deliberate and
unavoidable:

- the **input gather** (`tokenEmbed.slice(tokens)`) needs `[vocab, dim]`, because
  a gather indexes the leading axis;
- the **output projection** needs `[dim, vocab]` contiguous, because a matvec
  contracts the trailing axis.

One layout cannot serve both. Alternatives considered: keeping only the
transposed copy and gathering with a per-token kernel (more code, slower load), or
gathering on the host from the f16 source (a PCIe round trip per token).

Cost: **+320 MiB on the f16 path, +640 MiB on the f32 fallback**, against a
511 MiB weight set. Fine on a 6 GiB card; worth knowing on a small integrated GPU.
Load time cost: ~336 ms once, which pays back after about five tokens.
