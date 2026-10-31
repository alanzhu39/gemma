/**
 * Is the tiled kernel worth it for the PV stage of prefill attention?
 *
 * After the projections and QK moved to the kernel, PV is the largest remaining
 * GPU region: 89.4 ms of the 163.6 ms total at S=1024 (55%). It was left on
 * `np.einsum` because the earlier probe priced it against a transposed copy it
 * does not need -- v is already [N, 256] and PV reduces over N, so v is exactly
 * the [K, out] operand the kernel's "in,out" layout wants.
 *
 * This measures PV alone both ways, on GPU timestamps, and checks agreement.
 * Shapes come from the model: probs [S*4, N], v [N, 256], out [S*4, 256].
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { matmul, matmulGpuMs } from "../../src/lib/gemma3/matmul";

const PEAK_AT_1506 = 3855;
const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const round = (x: number, n = 3) => +x.toFixed(n);

async function jaxGpuMs(fn: () => np.Array, reps = 8): Promise<number> {
	const { profiler } = await import("@jax-js/jax");
	const measures: PerformanceMeasure[] = [];
	const obs = new PerformanceObserver((l) => {
		for (const e of l.getEntries()) measures.push(e as PerformanceMeasure);
	});
	try {
		obs.observe({ type: "measure" });
	} catch {}
	for (let i = 0; i < 3; i++) (await fn().slice([0]).data());
	let last = -1;
	for (let i = 0; i < 40 && measures.length !== last; i++) {
		last = measures.length;
		await new Promise((r) => setTimeout(r, 40));
	}
	measures.length = 0;
	profiler.startTrace();
	const keep: np.Array[] = [];
	for (let i = 0; i < reps; i++) keep.push(fn());
	await blockUntilReady(keep);
	profiler.stopTrace();
	let stable = 0;
	last = -1;
	for (let i = 0; i < 80; i++) {
		const n = measures.filter((e) => e.name.startsWith("Kernel[")).length;
		if (n >= reps && measures.length === last) {
			if (++stable >= 2) break;
		} else stable = 0;
		last = measures.length;
		await new Promise((r) => setTimeout(r, 40));
	}
	obs.disconnect();
	for (const k of keep) k.dispose();
	const named = measures.filter((e) => e.name.startsWith("Kernel["));
	if (named.length < reps) throw new Error(`captured ${named.length}/${reps}`);
	return named.reduce((a, e) => a + e.duration, 0) / named.length;
}

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");

	const rows: Record<string, unknown>[] = [];
	for (const S of [128, 512, 1024]) {
		const N = S;
		// probs: real positive values, not a literal that can be folded away.
		const probs = np
			.array(
				Float32Array.from({ length: S * 4 * N }, (_, i) => 0.5 + 0.5 * Math.sin(i * 0.003)),
				{ dtype: np.float32 },
			)
			.reshape([S * 4, N]);
		const v = np
			.array(Float32Array.from({ length: N * 256 }, (_, i) => Math.cos(i * 0.0017)), {
				dtype: np.float32,
			})
			.reshape([N, 256]);

		// Current path: [S*4,N] x [N,256] through einsum, as the model does.
		const einsumMs = round(
			await jaxGpuMs(() => np.einsum("SN,ND->SD", probs.ref, v.ref)),
		);
		// The same product as a dot, in case einsum's path differs.
		const dotMs = round(await jaxGpuMs(() => np.dot(probs.ref, v.ref)));

		// Tiled: v is [K, out], which is the "in,out" layout. No copy.
		const t: number[] = [];
		for (let i = 0; i < 8; i++) {
			const r = matmul(tree.ref(v), probs.ref, "in,out", { timestamps: true })!;
			t.push(await matmulGpuMs(r));
			r.dispose();
		}
		const tiledMs = round(median(t));

		// Agreement.
		const a = (await np.dot(probs.ref, v.ref).data()) as Float32Array;
		const got = matmul(tree.ref(v), probs.ref, "in,out")!;
		const b = (await got.data()) as Float32Array;
		let mag = 0;
		let maxAbs = 0;
		for (let i = 0; i < a.length; i++) {
			mag = Math.max(mag, Math.abs(a[i]));
			maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
		}

		const flops = 2 * S * 4 * N * 256;
		if (flops / 1e6 / Math.min(einsumMs, dotMs, tiledMs) > PEAK_AT_1506) {
			throw new Error("PV above peak");
		}

		rows.push({
			S,
			shape: `[${S * 4},${N}]x[${N},256]`,
			gflop: round(flops / 1e9, 3),
			einsumMs,
			einsumGf: round(flops / 1e6 / einsumMs, 0),
			dotMs,
			dotGf: round(flops / 1e6 / dotMs, 0),
			tiledMs,
			tiledGf: round(flops / 1e6 / tiledMs, 0),
			vsEinsum: round(einsumMs / tiledMs, 2),
			vsDot: round(dotMs / tiledMs, 2),
			err: +(maxAbs / mag).toExponential(1),
		});
	}
	return { rows };
}
