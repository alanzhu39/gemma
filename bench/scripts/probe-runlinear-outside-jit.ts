/**
 * Why does calling a jitted `runLinear` from outside a jit fail?
 *
 * The fused layer calls `runLinear` while tracing another `jit`, so its
 * arguments are tracers whose lifetime the outer program manages. The prefill
 * path calls it from ordinary code, with the model's real weight arrays and a
 * `Linear` whose `bias` key is simply absent. This probe separates the two
 * differences instead of guessing which one matters.
 */
import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { runLinear } from "../../src/lib/gemma3/model";

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");

	const mkX = () =>
		np.array(Float32Array.from({ length: 14 }, (_, i) => Math.sin(i)), { dtype: np.float32 })
			.reshape([2, 7]);
	const mkW = () =>
		np.array(Float32Array.from({ length: 21 }, (_, i) => Math.cos(i)), { dtype: np.float32 })
			.reshape([3, 7]);

	const ref = (w: np.Array, x: np.Array) => np.dot(x, w.transpose());

	const out: Record<string, unknown> = {};

	// 1. Straight through, weight not ref'd -- the shapes/ownership question.
	try {
		const w = mkW();
		const got = runLinear({ weight: w, bias: undefined }, mkX());
		const a = got.dataSync() as Float32Array;
		out.direct = { ok: true, first: +a[0].toFixed(5), len: a.length };
	} catch (e) {
		out.direct = { ok: false, err: String((e as Error).message).slice(0, 200) };
	}

	// 2. With an explicit ref on the weight.
	try {
		const w = mkW();
		const got = runLinear({ weight: tree.ref(w), bias: undefined }, mkX());
		const a = got.dataSync() as Float32Array;
		out.withRef = { ok: true, first: +a[0].toFixed(5), len: a.length };
		w.dispose();
	} catch (e) {
		out.withRef = { ok: false, err: String((e as Error).message).slice(0, 200) };
	}

	// 3. An object that genuinely has no `bias` key at all.
	try {
		const w = mkW();
		const linear: { weight: np.Array } = { weight: tree.ref(w) };
		const got = runLinear(linear, mkX());
		const a = got.dataSync() as Float32Array;
		out.withoutBiasKey = { ok: true, first: +a[0].toFixed(5), len: a.length };
		w.dispose();
	} catch (e) {
		out.withoutBiasKey = { ok: false, err: String((e as Error).message).slice(0, 200) };
	}

	// 4. Does calling it twice reuse the weights, i.e. is the weight consumed?
	try {
		const w = mkW();
		const r1 = runLinear({ weight: tree.ref(w), bias: undefined }, mkX());
		const r2 = runLinear({ weight: tree.ref(w), bias: undefined }, mkX());
		const a = r1.dataSync() as Float32Array;
		const b = r2.dataSync() as Float32Array;
		// Reference values.
		const w2 = mkW();
		const want = (ref(w2, mkX()).dataSync() as Float32Array)[0];
		w2.dispose();
		out.twice = { ok: true, a0: +a[0].toFixed(5), b0: +b[0].toFixed(5), want: +want.toFixed(5) };
		w.dispose();
	} catch (e) {
		out.twice = { ok: false, err: String((e as Error).message).slice(0, 200) };
	}

	return out;
}
