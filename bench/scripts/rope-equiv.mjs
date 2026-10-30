// Usage: node bench/scripts/rope-equiv.mjs [S] [pos]   (exit 0 = new RoPE matches the original)
//
// Compares the original rotate-half RoPE (split / negate / concat) against the
// flip-view + pre-negated-sine version now in src/lib/gemma3/model.ts, on the
// exact shapes the model uses and with the model's own precomputeRoPECache math.
// Also dumps the jax-js kernel list for RoPE alone, old vs new, for both the
// 3-D q shape and the 2-D k shape.
import { init, jit, numpy as np, setDebug, defaultDevice, DType } from "@jax-js/jax";
await init("cpu"); defaultDevice("cpu");
const HALF = 128;
const S = Number(process.argv[2] ?? 1);
const POS = Number(process.argv[3] ?? 129);

// --- precomputeRoPECache: old returns sin, new returns the pre-negated sin ---
function ropeTables(offset, seqLen, headDim, isSliding, variant) {
  const base = isSliding ? 10000 : 1000000;
  const invFreq = np.exp(
    np.arange(0, headDim, 2, { dtype: DType.Float32 }).mul(-Math.log(base) / headDim),
  );
  const positions = np.arange(seqLen).add(offset);
  let angles = np.outer(positions, invFreq);
  angles = np.concatenate([angles.ref, angles], -1);
  const cos = np.cos(angles.ref).astype(DType.Float32);
  if (variant === "old") return { cos, sin: np.sin(angles).astype(DType.Float32) };
  const sign = np.where(
    np.arange(headDim).less(headDim / 2),
    np.array(-1, { dtype: DType.Float32 }),
    np.array(1, { dtype: DType.Float32 }),
  );
  return { cos, sin: np.sin(angles).astype(DType.Float32).mul(sign) };
}

function broadcastTo(t, x) {
  const dim = x.shape[x.ndim - 1];
  return x.ndim === 3 ? t.reshape([x.shape[0], 1, dim]) : t;
}
function oldRoPE(x, cos, sin) {
  const [x1, x2] = np.split(x.ref, [HALF], -1);
  return x.ref.mul(broadcastTo(cos, x)).add(np.concatenate([x2.neg(), x1], -1).mul(broadcastTo(sin, x)));
}
function newRoPE(x, cos, sinSigned) {
  const dim = x.shape[x.ndim - 1];
  const lead = x.shape.slice(0, -1);
  const rot = np.flip(x.ref.reshape([...lead, 2, HALF]), -2).reshape([...lead, dim]);
  return x.mul(broadcastTo(cos, x)).add(rot.mul(broadcastTo(sinSigned, x)));
}

// Deterministic input, independent of the rotation tables.
const vals = (n) => {
  const a = new Float32Array(n);
  let s = 12345;
  for (let i = 0; i < n; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    a[i] = (s / 0x7fffffff - 0.5) * 4;
  }
  return a;
};
const q0 = np.array(vals(S * 4 * 256), { dtype: np.float32 }).reshape([S, 4, 256]);
const k0 = np.array(vals(S * 256), { dtype: np.float32 }).reshape([S, 256]);

// --- kernel counts first, so jit has not compiled these shapes yet ---
console.log(`--- kernel lists (${S === 1 ? "decode" : "prefill"} shapes, S=${S}) ---`);
for (const variant of ["old", "new"]) {
  const rope = variant === "old" ? oldRoPE : newRoPE;
  for (const [label, x] of [["q", q0], ["k", k0]]) {
    const t = ropeTables(POS, S, 256, true, variant);
    const f = jit((a, c, s) => rope(a, c, s));
    setDebug(4);
    const out = f(x.ref, t.cos.ref, t.sin.ref);
    setDebug(0);
    await out.data();
    console.log(`--- ${variant} ${label} ${JSON.stringify(x.shape)}`);
  }
}

// --- equivalence, sliding and global tables ---
let worst = 0;
const rel = (a, b) => {
  let e = 0, m = 0;
  for (let i = 0; i < a.length; i++) {
    e = Math.max(e, Math.abs(a[i] - b[i]));
    m = Math.max(m, Math.abs(a[i]));
  }
  return { e, r: e / m };
};
for (const isSliding of [true, false]) {
  const tOld = ropeTables(POS, S, 256, isSliding, "old");
  const tNew = ropeTables(POS, S, 256, isSliding, "new");
  const tOld2 = ropeTables(POS, S, 256, isSliding, "old");
  const aq = await jit((x, c, s) => oldRoPE(x, c, s))(q0.ref, tOld.cos.ref, tOld.sin.ref).data();
  const ak = await jit((x, c, s) => oldRoPE(x, c, s))(k0.ref, tOld2.cos, tOld2.sin).data();
  const bq = await jit((x, c, s) => newRoPE(x, c, s))(q0.ref, tNew.cos.ref, tNew.sin.ref).data();
  const bk = await jit((x, c, s) => newRoPE(x, c, s))(k0.ref, tNew.cos, tNew.sin).data();
  const rq = rel(aq, bq), rk = rel(ak, bk);
  console.log(
    `${isSliding ? "sliding" : "global "} S=${S} pos=${POS}: q max abs err ${rq.e.toExponential(2)} (rel ${rq.r.toExponential(2)}), k max abs err ${rk.e.toExponential(2)} (rel ${rk.r.toExponential(2)})`,
  );
  worst = Math.max(worst, rq.r, rk.r);
}
console.log(`worst relative: ${worst.toExponential(2)}`);
process.exit(worst < 1e-6 ? 0 : 1);
