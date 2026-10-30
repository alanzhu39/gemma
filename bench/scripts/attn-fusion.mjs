// Usage: node bench/scripts/attn-fusion.mjs <N> <orig|restruct|matmul|fused>
// Dump the fused kernel list jax-js produces for the decode attention core.
import { init, jit, nn, numpy as np, setDebug, defaultDevice } from "@jax-js/jax";
await init("cpu"); defaultDevice("cpu");
const N = Number(process.argv[2] ?? 1024), S = 1;
const variant = process.argv[3] ?? "orig";
const f = jit(function attn(q, k, v, mask) {
  if (variant === "orig") {
    const w = nn.softmax(np.where(mask, np.einsum("SHD,ND->HSN", q, k).mul(1 / 16), -Infinity));
    return np.einsum("HSN,ND->SHD", w, v);
  }
  if (variant === "fused") {
    // Mask as additive bias (fuses into the QK^T epilogue); exp recomputed per consumer.
    const bias = np.where(mask.reshape([S, 1, N]), 0, -Infinity);
    const s = np.einsum("SHD,ND->SHN", q, k).mul(1 / 16).add(bias);
    const m = np.maximum(s.ref.max(-1), -3e38);
    const l = np.maximum(np.exp(s.ref.sub(m.ref.reshape([S, 4, 1]))).sum(-1), 1e-30);
    return np.einsum("SHN,ND->SHD", np.exp(s.sub(m.reshape([S, 4, 1]))), v).div(l.reshape([S, 4, 1]));
  }
  // Restructured: no transposes, unnormalised P@V with the 1/l in the epilogue.
  const s = np.where(mask.reshape([S, 1, N]), np.einsum("SHD,ND->SHN", q, k).mul(1 / 16), -Infinity);
  const m = s.ref.max(-1, { keepdims: true });
  const p = np.exp(s.sub(m));
  if (variant === "restruct") {
    const l = p.ref.sum(-1, { keepdims: true });
    return np.einsum("SHN,ND->SHD", p, v).div(l);
  }
  if (variant === "matmul") {
    const l = p.ref.sum(-1, { keepdims: true });
    return np.matmul(p, v).div(l);
  }
});
const q = np.ones([S, 4, 256]), k = np.ones([N, 256]), v = np.ones([N, 256]);
const mask = np.ones([S, N], { dtype: np.bool });
setDebug(4);
const out = f(q, k, v, mask);
setDebug(0);
await out.data();
