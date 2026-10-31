/**
 * The real GPU total for a prefill pass, including the custom matmul.
 *
 * jax-js's `profiler.startTrace()` only instruments kernels jax-js submits, so
 * every trace-based breakdown in this repo *under-reports* once a hand-written
 * kernel is in the model. After the tiled matmul landed, the trace said 74.8 ms
 * of GPU time for a pass whose wall time was 405 ms -- a gap large enough that
 * using it would badly understate how much GPU work remains.
 *
 * This probe sums both sources over one pass:
 *
 *   jaxKernels  from the jax-js timestamp trace
 *   customMs    from the matmul's own query sets (accumulated through
 *               `matmulGpuMs`, awaited after the pass)
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { createGemma3State, runGemma3Step } from "../../src/lib/gemma3/model";
import type { Gemma3, Gemma3State } from "../../src/lib/gemma3/model";
import { setMatmulTimestamps, takeMatmulTimes } from "../../src/lib/gemma3/matmul";
import { loadModelF32 } from "../f32-model";

const round = (x: number, n = 2) => +x.toFixed(n);
const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

function dispose(s: Gemma3State) {
	for (const c of s.kvCaches) {
		c.k.dispose();
		c.v.dispose();
	}
	s.positionTensor.dispose();
}

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model: Gemma3 = loadModelF32(safetensors.parse(await res.arrayBuffer()));

	const workloads = await (await fetch("/bench/workloads.json")).json();
	const tokens: number[] = workloads.prefill
		.find((w: { name: string }) => w.name === "prefill-1024")
		.tokens.slice(0, 1024);

	const wallRun = async () => {
		const state = createGemma3State(model);
		const t0 = performance.now();
		const out = runGemma3Step(tree.ref(model), state, np.array(tokens, { dtype: np.uint32 }));
		await blockUntilReady(out.latent);
		await out.latent.ref.slice([-1]).data();
		const dt = performance.now() - t0;
		out.latent.dispose();
		dispose(out.state);
		return dt;
	};

	for (let i = 0; i < 3; i++) await wallRun();
	const wall = median([await wallRun(), await wallRun(), await wallRun(), await wallRun()]);

	// ---- with the custom kernel instrumented ----
	setMatmulTimestamps(true);
	// Warm the timestamp path so its query sets are compiled.
	for (let i = 0; i < 2; i++) await wallRun();

	const { profiler } = await import("@jax-js/jax");
	const measures: PerformanceMeasure[] = [];
	const obs = new PerformanceObserver((l) => {
		for (const e of l.getEntries()) measures.push(e as PerformanceMeasure);
	});
	try {
		obs.observe({ type: "measure" });
	} catch {}
	await new Promise((r) => setTimeout(r, 100));
	measures.length = 0;

	takeMatmulTimes();
	profiler.startTrace();
	const state = createGemma3State(model);
	const t0 = performance.now();
	const out = runGemma3Step(tree.ref(model), state, np.array(tokens, { dtype: np.uint32 }));
	await blockUntilReady(out.latent);
	await out.latent.ref.slice([-1]).data();
	const tracedWall = performance.now() - t0;
	profiler.stopTrace();
	const custom = await takeMatmulTimes();
	let stable = 0;
	let last = -1;
	for (let i = 0; i < 50; i++) {
		if (measures.length === last) {
			if (++stable >= 3) break;
		} else stable = 0;
		last = measures.length;
		await new Promise((r) => setTimeout(r, 40));
	}
	obs.disconnect();
	out.latent.dispose();
	dispose(out.state);
	setMatmulTimestamps(false);

	const jaxKernels = measures.filter((e) => e.name.startsWith("Kernel["));
	const jaxMs = jaxKernels.reduce((a, e) => a + e.duration, 0);

	return {
		ctxLen: tokens.length,
		wallMs: round(tracedWall),
		baselineWallMs: round(wall),
		gpu: {
			jaxKernels: jaxKernels.length,
			jaxMs: round(jaxMs),
			customMatmuls: custom.count,
			customMs: round(custom.total),
			totalMs: round(jaxMs + custom.total),
			customSharePct: round((100 * custom.total) / Math.max(1e-9, jaxMs + custom.total), 1),
		},
		accounting: {
			gpuTotalMs: round(jaxMs + custom.total),
			wallMs: round(tracedWall),
			ratio: round(tracedWall / Math.max(1e-9, jaxMs + custom.total), 2),
		},
	};
}
