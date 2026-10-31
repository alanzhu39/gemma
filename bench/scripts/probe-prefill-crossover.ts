/**
 * Where does the tiled-matmul prefill path start to win, and why?
 *
 * The first full measurement after integrating the kernel showed:
 *
 *   prefill-128    900.1 tok/s   (was 1183)   <- regression
 *   prefill-512   2414.0 tok/s   (was 1188)
 *   prefill-1024  2246.6 tok/s   (was 1020)
 *
 * A regression that appears only at the smallest size points at launch overhead,
 * not arithmetic: at S=128 every kernel is tiny, and the prefill path issues more
 * dispatches than the fused one because the layer is split around the custom
 * matmul. This probe measures the wall time and the GPU-trace dispatch count of
 * both arms at each S the suite benchmarks, so the switch-over point is chosen
 * from data instead of assumed.
 *
 * The two arms share everything except the matmul: `__noTiledMatmul` selects the stock
 * `np.dot` lowering inside the same restructured layer, so this is an honest
 * like-for-like comparison and not "old code vs new code".
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

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model: Gemma3 = loadModelF32(safetensors.parse(await res.arrayBuffer()));

	const workloads = await (await fetch("/bench/workloads.json")).json();
	const longest: number[] = workloads.prefill.find(
		(w: { name: string }) => w.name === "prefill-1024",
	).tokens;

	// bench.ts's exact region: step, sync, read one element back.
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

	// Dispatch count, from jax-js's own trace.
	const countKernels = async (tokens: number[]) => {
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

		profiler.startTrace();
		const state = createGemma3State(model);
		const input = np.array(tokens, { dtype: np.uint32 });
		const out = runGemma3Step(tree.ref(model), state, input);
		await blockUntilReady(out.latent);
		profiler.stopTrace();
		let stable = 0;
		let last = -1;
		for (let i = 0; i < 40; i++) {
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
			kernels: k.length,
			gpuMs: round(k.reduce((a, e) => a + e.duration, 0), 1),
		};
	};

	// Clock warm-up.
	for (let i = 0; i < 3; i++) await once(longest);

	const rows: Record<string, unknown>[] = [];
	for (const S of [128, 256, 512, 1024]) {
		const tokens = longest.slice(0, S);
		const arm: Record<string, unknown> = { S };
		for (const noTiled of [false, true]) {
			(globalThis as { __noTiledMatmul?: boolean }).__noTiledMatmul = noTiled;
			const name = noTiled ? "stock" : "kernel";
			// Warm this arm.
			for (let i = 0; i < 3; i++) await once(tokens);
			const times: number[] = [];
			for (let i = 0; i < 7; i++) times.push(round(await once(tokens)));
			arm[`${name}Ms`] = median(times);
			arm[`${name}Tps`] = round((S * 1000) / median(times), 1);
			arm[`${name}MinMs`] = Math.min(...times);
		}
		(globalThis as { __noTiledMatmul?: boolean }).__noTiledMatmul = false;
		arm.kernelDispatch = await countKernels(tokens);
		(globalThis as { __noTiledMatmul?: boolean }).__noTiledMatmul = true;
		arm.stockDispatch = await countKernels(tokens);
		(globalThis as { __noTiledMatmul?: boolean }).__noTiledMatmul = false;
		arm.speedup = round((arm.stockMs as number) / (arm.kernelMs as number), 2);
		rows.push(arm);
	}

	return { rows };
}
