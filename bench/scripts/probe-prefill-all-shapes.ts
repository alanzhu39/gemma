/**
 * Every matmul shape in a prefill pass: the engine's current lowering vs a
 * tiled shared-memory kernel, both measured on GPU timestamps.
 *
 * The engine's arms are timed with jax-js's own trace (it only instruments its
 * own submits); the tiled kernel is timed with its own query set, for the same
 * reason. Both are GPU-side numbers, so the 85-99% CPU load on this box cannot
 * inflate either.
 *
 * Each row also states the achieved GFLOPS against the card's measured peak
 * (1280 lanes x 1506 MHz x 2). Anything at or above peak means the work was
 * elided, and the probe refuses it.
 */
import { blockUntilReady, defaultDevice, getWebGPUDevice, init, numpy as np } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import type { Gemma3 } from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";
import { tiledGemm, tiledGpuMs } from "./tiled-gemm";

const PEAK_AT_1506 = 3855;
const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const round = (x: number, n = 2) => +x.toFixed(n);

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
	if (named.length < reps) throw new Error(`captured ${named.length}/${reps} kernels`);
	return named.reduce((a, e) => a + e.duration, 0) / named.length;
}

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const adapter = await navigator.gpu.requestAdapter();
	const res = await fetch("/weights/model.safetensors");
	const model: Gemma3 = loadModelF32(safetensors.parse(await res.arrayBuffer()));

	const S = 1024;
	const layer = model.layers[0];
	const mk = (r: number, c: number, f: number) =>
		np.array(Float32Array.from({ length: r * c }, (_, i) => Math.sin(i * f)), { dtype: np.float32 })
			.reshape([r, c]);

	const cases = [
		{ name: "qProj", w: layer.selfAttn.qProj.weight, x: mk(S, 640, 0.001), perLayer: 1 },
		{ name: "kProj", w: layer.selfAttn.kProj.weight, x: mk(S, 640, 0.0011), perLayer: 1 },
		{ name: "vProj", w: layer.selfAttn.vProj.weight, x: mk(S, 640, 0.0012), perLayer: 1 },
		{ name: "oProj", w: layer.selfAttn.oProj.weight, x: mk(S, 1024, 0.0013), perLayer: 1 },
		{ name: "gateProj", w: layer.mlp.gateProj.weight, x: mk(S, 640, 0.0014), perLayer: 1 },
		{ name: "upProj", w: layer.mlp.upProj.weight, x: mk(S, 640, 0.0015), perLayer: 1 },
		{ name: "downProj", w: layer.mlp.downProj.weight, x: mk(S, 2048, 0.0016), perLayer: 1 },
	];

	const rows: Record<string, unknown>[] = [];
	for (const c of cases) {
		const [outDim, inDim] = c.w.shape;
		const flops = 2 * S * inDim * outDim;

		const viewMs = round(await jaxGpuMs(() => np.dot(c.x.ref, c.w.ref.transpose())), 3);
		const wT = c.w.ref.transpose().mul(1) as np.Array;
		const matTMs = round(await jaxGpuMs(() => np.dot(c.x.ref, wT.ref)), 3);
		wT.dispose();

		const t: number[] = [];
		for (let i = 0; i < 10; i++) {
			const r = tiledGemm(c.x.ref, c.w.ref, { timestamps: true })!;
			t.push(await tiledGpuMs(r.out));
			r.out.dispose();
		}
		const tiledMs = round(median(t), 3);

		// Agreement.
		const a = (await np.dot(c.x.ref, c.w.ref.transpose()).data()) as Float32Array;
		const tr = tiledGemm(c.x.ref, c.w.ref)!;
		// `.data()` consumes its receiver, so nothing else may dispose `tr.out`.
		const d = (await tr.out.data()) as Float32Array;
		let mag = 0;
		let ad = 0;
		for (let i = 0; i < a.length; i++) {
			mag = Math.max(mag, Math.abs(a[i]));
			ad = Math.max(ad, Math.abs(a[i] - d[i]));
		}
		if (!(mag > 1e-6)) throw new Error(`${c.name}: zero output`);
		if (ad / mag > 2e-4) throw new Error(`${c.name}: tiled err ${ad}/${mag}`);

		const gf = flops / 1e6;
		if (gf / tiledMs > PEAK_AT_1506) throw new Error(`${c.name}: above peak`);
		rows.push({
			name: c.name,
			shape: `[${S},${inDim}]x[${inDim},${outDim}]`,
			gflop: round(flops / 1e9, 3),
			viewMs,
			viewGf: round(gf / viewMs, 0),
			matTMs,
			matTGf: round(gf / matTMs, 0),
			tiledMs,
			tiledGf: round(gf / tiledMs, 0),
			vsView: round(viewMs / tiledMs, 2),
			vsMatT: round(matTMs / tiledMs, 2),
			err: +(ad / mag).toExponential(1),
		});
	}

	// ---- attention core -----------------------------------------------------
	// QK: per head [S,256] x [N,256]^T. PV: [S,N] x [N,256] (v row-major in k).
	const N = 1024;
	const q = mk(S * 4, 256, 0.0009);
	const k = mk(N, 256, 0.0011);
	const p = np
		.array(Float32Array.from({ length: S * 4 * N }, (_, i) => 0.5 + 0.5 * Math.sin(i * 0.003)), {
			dtype: np.float32,
		})
		.reshape([S * 4, N]);
	const v = mk(N, 256, 0.0017);

	// QK all four heads at once: [S*4, 256] x [256, N] with the same k for each
	// head, which is what the model does (heads differ only in q).
	const qkFlops = 2 * S * 4 * N * 256;
	const qkViewMs = round(
		await jaxGpuMs(() => np.dot(q.ref, k.ref.transpose()).reshape([S, 4, N])),
		3,
	);
	const qkTiled: number[] = [];
	for (let i = 0; i < 8; i++) {
		const r = tiledGemm(q.ref, k.ref, { timestamps: true })!;
		qkTiled.push(await tiledGpuMs(r.out));
		r.out.dispose();
	}
	const qkTiledMs = round(median(qkTiled), 3);
	if (qkFlops / 1e6 / Math.min(qkViewMs, qkTiledMs) > PEAK_AT_1506) {
		throw new Error("QK above peak");
	}

	// PV: [S*4, N] x [N, 256], second operand contiguous along the reduction
	// axis already (V is [N,256] row-major, reduction over N).
	const pvFlops = 2 * S * 4 * N * 256;
	const pvViewMs = round(await jaxGpuMs(() => np.dot(p.ref, v.ref).reshape([S, 4, 256])), 3);
	// The tiled kernel wants w as [out, in]; V is [N,256] = [in, out], so it
	// needs the transposed copy, which is the honest comparison for this stage.
	const vT = v.ref.transpose().mul(1) as np.Array;
	const pvTiled: number[] = [];
	for (let i = 0; i < 8; i++) {
		const r = tiledGemm(p.ref, vT.ref, { timestamps: true })!;
		pvTiled.push(await tiledGpuMs(r.out));
		r.out.dispose();
	}
	const pvTiledMs = round(median(pvTiled), 3);

	const attnRows = [
		{
			name: "QK (4 heads)",
			shape: `[${S * 4},256]x[256,${N}]`,
			gflop: round(qkFlops / 1e9, 3),
			viewMs: qkViewMs,
			viewGf: round(qkFlops / 1e6 / qkViewMs, 0),
			tiledMs: qkTiledMs,
			tiledGf: round(qkFlops / 1e6 / qkTiledMs, 0),
			vsView: round(qkViewMs / qkTiledMs, 2),
		},
		{
			name: "PV (4 heads)",
			shape: `[${S * 4},${N}]x[${N},256]`,
			gflop: round(pvFlops / 1e9, 3),
			viewMs: pvViewMs,
			viewGf: round(pvFlops / 1e6 / pvViewMs, 0),
			tiledMs: pvTiledMs,
			tiledGf: round(pvFlops / 1e6 / pvTiledMs, 0),
			vsView: round(pvViewMs / pvTiledMs, 2),
		},
	];

	const tot = (k: string) => round(rows.reduce((a, r) => a + (r[k] as number), 0), 2);
	return {
		config: {
			S,
			precision: "f32",
			peakGflopsAt1506: PEAK_AT_1506,
			f16: adapter!.features.has("shader-f16" as GPUFeatureName),
		},
		projections: rows,
		projectionsPerLayer: {
			gflop: round(rows.reduce((a, r) => a + (r.gflop as number), 0), 2),
			viewMs: tot("viewMs"),
			matTMs: tot("matTMs"),
			tiledMs: tot("tiledMs"),
			viewGf: round((rows.reduce((a, r) => a + (r.gflop as number), 0) * 1e3) / tot("viewMs"), 0),
			tiledGf: round((rows.reduce((a, r) => a + (r.gflop as number), 0) * 1e3) / tot("tiledMs"), 0),
		},
		attention: attnRows,
		projectedPrefill: {
			note:
				"measured GPU time of the parts a GEMM rewrite touches, per layer, x18. " +
				"Compare with the 960 ms total from probe-prefill-breakdown.",
			viewMs: round((tot("viewMs") + attnRows[0].viewMs + attnRows[1].viewMs) * 18, 0),
			tiledMs: round((tot("tiledMs") + attnRows[0].tiledMs + attnRows[1].tiledMs) * 18, 0),
		},
	};
}
