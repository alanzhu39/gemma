/**
 * Three arms at every prefill size, so the cost of un-fusing the layer can be
 * separated from the gain of the tiled matmul.
 *
 *   fused      the original `runDecoderLayer`: everything inside one jit
 *   split      the restructured layer, stock np.dot  -> isolates restructuring
 *   kernel     the restructured layer + tiled matmul -> the candidate
 *
 * The reason to measure all three: the restructured layer issues more
 * dispatches (each matmul is now its own submit), which costs at small S, while
 * the tiled matmul removes far more GPU time at large S. If those were reported
 * as one number, a regression at S=128 would look like the kernel being slow.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { createGemma3State, runGemma3Step } from "../../src/lib/gemma3/model";
import type { Gemma3, Gemma3State } from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const round = (x: number, n = 2) => +x.toFixed(n);

function dispose(state: Gemma3State) {
	for (const c of state.kvCaches) {
		c.k.dispose();
		c.v.dispose();
	}
	state.positionTensor.dispose();
}

type Arm = "fused" | "split" | "kernel";

function select(arm: Arm) {
	const g = globalThis as { __noTiledMatmul?: boolean; __noPrefillLayer?: boolean };
	g.__noPrefillLayer = arm === "fused";
	g.__noTiledMatmul = arm !== "kernel";
}

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model: Gemma3 = loadModelF32(safetensors.parse(await res.arrayBuffer()));

	const workloads = await (await fetch("/bench/workloads.json")).json();
	const longest: number[] = workloads.prefill.find(
		(w: { name: string }) => w.name === "prefill-1024",
	).tokens.slice(0, 1024);

	/** bench.ts's exact region. */
	const once = async (tokens: number[]) => {
		const state = createGemma3State(model);
		const input = np.array(tokens, { dtype: np.uint32 });
		const t0 = performance.now();
		const out = runGemma3Step(tree.ref(model), state, input);
		await blockUntilReady(out.latent);
		await out.latent.ref.slice([-1]).data();
		const dt = performance.now() - t0;
		out.latent.dispose();
		dispose(out.state);
		return dt;
	};

	const trace = async (tokens: number[]) => {
		const { profiler } = await import("@jax-js/jax");
		const measures: PerformanceMeasure[] = [];
		const obs = new PerformanceObserver((l) => {
			for (const e of l.getEntries()) measures.push(e as PerformanceMeasure);
		});
		try {
			obs.observe({ type: "measure" });
		} catch {}
		await new Promise((r) => setTimeout(r, 120));
		measures.length = 0;
		profiler.startTrace();
		const state = createGemma3State(model);
		const out = runGemma3Step(tree.ref(model), state, np.array(tokens, { dtype: np.uint32 }));
		await blockUntilReady(out.latent);
		profiler.stopTrace();
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
		const k = measures.filter((e) => e.name.startsWith("Kernel["));
		return {
			jaxKernels: k.length,
			jaxGpuMs: round(k.reduce((a, e) => a + e.duration, 0), 1),
		};
	};

	const rows: Record<string, unknown>[] = [];

	// Clock warm-up on the longest input.
	select("fused");
	for (let i = 0; i < 3; i++) await once(longest);

	for (const S of [128, 192, 256, 320]) {
		const tokens = longest.slice(0, S);
		const row: Record<string, unknown> = { S };
		for (const arm of ["fused", "split", "kernel"] as Arm[]) {
			select(arm);
			for (let i = 0; i < 3; i++) await once(tokens);
			const t: number[] = [];
			for (let i = 0; i < 7; i++) t.push(round(await once(tokens)));
			row[`${arm}Ms`] = median(t);
			row[`${arm}Tps`] = round((S * 1000) / median(t), 1);
			const tr = await trace(tokens);
			row[`${arm}Kernels`] = tr.jaxKernels;
			row[`${arm}GpuMs`] = tr.jaxGpuMs;
		}
		row.splitVsFused = round((row.fusedMs as number) / (row.splitMs as number), 2);
		row.kernelVsFused = round((row.fusedMs as number) / (row.kernelMs as number), 2);
		row.kernelVsSplit = round((row.splitMs as number) / (row.kernelMs as number), 2);
		rows.push(row);
	}

	select("kernel");
	return { rows };
}
