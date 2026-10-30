// Usage: node bench/scripts/attn-equiv.mjs   (exit 0 = restructured core matches the original)
// Old vs restructured attention core on decode + prefill shapes (f32, CPU backend).
import { init, jit, nn, numpy as np, random, defaultDevice } from "@jax-js/jax";
await init("cpu"); defaultDevice("cpu");
const old = jit((q, k, v, mask) =>
  np.einsum("HSN,ND->SHD", nn.softmax(np.where(mask, np.einsum("SHD,ND->HSN", q, k).mul(1 / 16), -Infinity)), v));
const neu = jit((q, k, v, mask) => {
  const [S] = q.shape, N = k.shape[0];
  const bias = np.where(mask.reshape([S, 1, N]), 0, -Infinity);
  const s = np.einsum("SHD,ND->SHN", q, k).mul(1 / 16).add(bias);
  const m = np.maximum(s.ref.max(-1), -3e38);
  const l = np.maximum(np.exp(s.ref.sub(m.ref.reshape([S, 4, 1]))).sum(-1), 1e-30);
  return np.einsum("SHN,ND->SHD", np.exp(s.sub(m.reshape([S, 4, 1]))), v).div(l.reshape([S, 4, 1]));
});
let worst = 0;
for (const [S, N, valid, scale] of [[1, 512, 37, 4], [1, 512, 512, 4], [1, 1024, 900, 8], [1, 128, 1, 4], [64, 64, -1, 4]]) {
  const key = random.key(S * 7919 + N);
  const [k1, k2, k3] = random.split(key, 3);
  const q = random.normal(k1, [S, 4, 256]).mul(scale), k = random.normal(k2, [N, 256]).mul(scale), v = random.normal(k3, [N, 256]);
  const mask = valid < 0 ? np.tri(S, S, 0, { dtype: np.bool }) : np.arange(N).less(valid).reshape([1, N]);
  const a = await old(q.ref, k.ref, v.ref, mask.ref).data(), b = await neu(q, k, v, mask).data();
  let err = 0, mx = 0;
  for (let i = 0; i < a.length; i++) { err = Math.max(err, Math.abs(a[i] - b[i])); mx = Math.max(mx, Math.abs(a[i])); if (!Number.isFinite(b[i])) throw new Error("non-finite"); }
  console.log(`S=${S} N=${N} valid=${valid}: max abs err ${err.toExponential(2)} (max |out| ${mx.toFixed(3)})`);
  worst = Math.max(worst, err / mx);
}
// Masked/padded slots holding huge (finite) K/V, plus one fully masked query row.
{
  const N = 512, S = 2;
  const [k1, k2, k3] = random.split(random.key(1), 3);
  const q = random.normal(k1, [S, 4, 256]).mul(4);
  const pad = np.arange(N).greaterEqual(100).reshape([N, 1]);
  const k = np.where(pad.ref, 1e18, random.normal(k2, [N, 256]).mul(4));
  const v = np.where(pad, 1e18, random.normal(k3, [N, 256]));
  const mask = np.stack([np.arange(N).less(100), np.zeros([N], { dtype: np.bool })]); // row 1 fully masked
  const o = await old(q.ref, k.ref, v.ref, mask.ref).data(), b = await neu(q, k, v, mask).data();
  const half = b.length / 2;
  const nanOld = o.filter(Number.isNaN).length, nanNew = b.filter((x) => !Number.isFinite(x)).length;
  const row1Zero = b.subarray(half).every((x) => x === 0);
  let err = 0; for (let i = 0; i < half; i++) err = Math.max(err, Math.abs(o[i] - b[i]));
  console.log(`padded slots + fully-masked row: old NaN/inf count ${nanOld}, new non-finite count ${nanNew}, masked row all-zero ${row1Zero}, valid-row max abs err vs old ${err.toExponential(2)}`);
  if (nanNew || !row1Zero || err > 1e-5) worst = Infinity;
}
console.log("worst relative:", worst.toExponential(2)); process.exit(worst < 1e-4 ? 0 : 1);
