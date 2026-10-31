/**
 * Account for every millisecond of a prefill pass.
 *
 * `prefill-1024` measures 1004 ms for 1024 tokens (0.98 ms/token) and ~469
 * dispatches, so per-dispatch launch cost cannot explain it. The bench region
 * is `runGemma3Step` + a sync + a 640-float readback; there is no vocabulary
 * projection in it at all, which is worth stating because the decode work made
 * it the obvious suspect.
 *
 * This probe measures:
 *
 *   A. the real bench region, timed exactly as bench.ts times it
 *   B. GPU kernel time per kernel *shape* and per op expression, from jax-js's
 *      timestamp-query trace (immune to CPU contention on this box)
 *   C. the same pass with one sub-piece repeated N times, to price a phase by
 *      difference without changing the code under test
 *
 * Guard rails, same as the decode probe:
 *  - bandwidth above the card's ~192 GB/s peak means work was elided, not won;
 *  - every phase that reports a rate has its byte count stated with it.
 */
import {
	blockUntilReady,
	defaultDevice,
	getWebGPUDevice,
	init,
	numpy as np,
	tree,
} from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import {
	createGemma3State,
	runGemma3Step,
	type Gemma3,
	type Gemma3State,
} from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

const PEAK_GBPS = 192;

const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const round = (x: number, n = 2) => +x.toFixed(n);

function disposeState(state: Gemma3State) {
	for (const c of state.kvCaches) {
		c.k.dispose();
		c.v.dispose();
	}
	state.positionTensor.dispose();
}

/**
 * Exactly bench.ts's `timePrefill` region: step, sync, read one element back.
 * Duplicated on purpose so this probe cannot drift from the thing it explains.
 */
async function timePrefill(model: Gemma3, tokens: number[]) {
	const state = createGemma3State(model);
	const input = np.array(tokens, { dtype: np.uint32 });
	const t0 = performance.now();
	const out = runGemma3Step(tree.ref(model), state, input);
	await blockUntilReady(out.latent);
	await out.latent.ref.slice([-1]).data();
	const dt = performance.now() - t0;
	out.latent.dispose();
	disposeState(out.state);
	return dt;
}

/**
 * Collapse an AluExp dump to an op skeleton so equal ops group together.
 * Buffer shapes and scalar constants are kept: without them two different
 * matmuls (q vs down-projection) collapse into one row and the breakdown says
 * "matmul, 495 ms" instead of which matmul.
 */
function opKey(exp: string): string {
	return exp
		.replace(/0x[0-9a-f]+/g, "0xN")
		.replace(/-?\d+\.\d+e[-+]?\d+/g, "k")
		.replace(/-?\d+\.\d+/g, "k")
		.replace(/\$gidx:int32/g, "$g")
		.replace(/\$ridx:int32/g, "$r")
		.replace(/\s+/g, " ")
		.slice(0, 300);
}

/** A short shape signature: the distinct buffer shapes the kernel touches. */
function shapeSig(exp: string): string {
	const shapes = [...exp.matchAll(/\{([\d,]+):[^}]*\}/g)].map((m) => m[1]);
	return [...new Set(shapes)].join(" | ");
}

type Agg = { count: number; totalMs: number; sizes: Set<number>; sample: string };

export async function probe() {
	const ctxLen = (globalThis as { __ctx?: number }).__ctx ?? 1024;

	await init("webgpu");
	defaultDevice("webgpu");
	const dev = getWebGPUDevice();
	const adapter = await navigator.gpu.requestAdapter();
	const info = {
		hasTimestampQuery: dev.features.has("timestamp-query"),
		f16: adapter!.features.has("shader-f16" as GPUFeatureName),
		renderer: String((adapter!.info as { description?: string })?.description ?? ""),
	};

	const res = await fetch("/weights/model.safetensors");
	const file = safetensors.parse(await res.arrayBuffer());
	const model: Gemma3 = loadModelF32(file);

	const workloads = await (await fetch("/bench/workloads.json")).json();
	const wl = workloads.prefill.find((w: { name: string }) => w.name === `prefill-${ctxLen}`);
	if (!wl) throw new Error(`no workload prefill-${ctxLen}`);
	const tokens: number[] = wl.tokens.slice(0, ctxLen);

	// Clock warm-up: a Pascal card idles at 139 MHz against 1911 MHz boost.
	for (let i = 0; i < 3; i++) await timePrefill(model, tokens);

	// ---- A. the real region -------------------------------------------------
	const wall: number[] = [];
	for (let i = 0; i < 7; i++) wall.push(round(await timePrefill(model, tokens)));
	const wallMedian = median(wall);

	// ---- B. GPU kernel time from the timestamp trace -------------------------
	type TraceRow = { name: string; ms: number; props: Record<string, string> };
	const grab = (measures: PerformanceMeasure[]): TraceRow[] =>
		measures.map((e) => {
			const d = (e as unknown as { detail?: { devtools?: { properties?: [string, string][] } } })
				.detail;
			const props: Record<string, string> = {};
			for (const [k, v] of d?.devtools?.properties ?? []) props[k] = v;
			return { name: e.name, ms: e.duration, props };
		});

	let trace: {
		wallMs: number;
		kernelCount: number;
		totalGpuMs: number;
		byOp: { key: string; count: number; totalMs: number; avgMs: number; share: number; sample: string }[];
		byShape: { name: string; count: number; totalMs: number; share: number }[];
		routines: { name: string; count: number; totalMs: number; share: number }[];
	} | null = null;

	if (info.hasTimestampQuery) {
		const { profiler } = await import("@jax-js/jax");
		const measures: PerformanceMeasure[] = [];
		const obs = new PerformanceObserver((list) => {
			for (const e of list.getEntries()) measures.push(e as PerformanceMeasure);
		});
		try {
			obs.observe({ type: "measure" });
		} catch {
			/* getEntries fallback below */
		}

		profiler.startTrace();
		const state = createGemma3State(model);
		const input = np.array(tokens, { dtype: np.uint32 });
		const t0 = performance.now();
		const out = runGemma3Step(tree.ref(model), state, input);
		await blockUntilReady(out.latent);
		const t1 = performance.now();
		for (let i = 0; i < 24; i++) await new Promise((r) => setTimeout(r, 50));
		profiler.stopTrace();
		for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 50));
		obs.disconnect();
		out.latent.dispose();
		disposeState(out.state);

		const all = measures.length
			? measures
			: (performance.getEntriesByType("measure") as unknown as PerformanceMeasure[]).filter((e) =>
					/^(Kernel|Routine)\[/.test(e.name),
				);
		const rows = grab(all);
		const kernels = rows.filter((r) => r.name.startsWith("Kernel["));
		const routineRows = rows.filter((r) => r.name.startsWith("Routine["));

		const totalGpuMs = kernels.reduce((a, r) => a + r.ms, 0);
		const byOpMap = new Map<string, Agg>();
		for (const r of kernels) {
			const key = opKey(r.props.exp ?? r.name);
			const a = byOpMap.get(key) ?? { count: 0, totalMs: 0, sizes: new Set<number>(), sample: r.props.exp ?? "" };
			a.count++;
			a.totalMs += r.ms;
			a.sizes.add(Number(r.props.size ?? 0));
			byOpMap.set(key, a);
		}
		const byOp = [...byOpMap.entries()]
			.map(([key, a]) => ({
				key,
				count: a.count,
				totalMs: round(a.totalMs),
				avgMs: round(a.totalMs / a.count, 3),
				share: round((100 * a.totalMs) / Math.max(1e-9, totalGpuMs), 1),
				shapes: shapeSig(a.sample),
				sample: a.sample.slice(0, 700),
			}))
			.sort((x, y) => y.totalMs - x.totalMs);

		const byShapeMap = new Map<string, { count: number; totalMs: number }>();
		for (const r of kernels) {
			const a = byShapeMap.get(r.name) ?? { count: 0, totalMs: 0 };
			a.count++;
			a.totalMs += r.ms;
			byShapeMap.set(r.name, a);
		}
		const byShape = [...byShapeMap.entries()]
			.map(([name, a]) => ({
				name,
				count: a.count,
				totalMs: round(a.totalMs),
				share: round((100 * a.totalMs) / Math.max(1e-9, totalGpuMs), 1),
			}))
			.sort((x, y) => y.totalMs - x.totalMs);

		const routineMap = new Map<string, { count: number; totalMs: number }>();
		for (const r of routineRows) {
			const a = routineMap.get(r.name) ?? { count: 0, totalMs: 0 };
			a.count++;
			a.totalMs += r.ms;
			routineMap.set(r.name, a);
		}
		const routines = [...routineMap.entries()]
			.map(([name, a]) => ({
				name,
				count: a.count,
				totalMs: round(a.totalMs),
				share: round((100 * a.totalMs) / Math.max(1e-9, totalMs), 1),
			}))
			.sort((x, y) => y.totalMs - x.totalMs);

		trace = {
			wallMs: round(t1 - t0),
			kernelCount: kernels.length,
			totalGpuMs: round(totalGpuMs),
			byOp: byOp.slice(0, 24),
			byShape: byShape.slice(0, 20),
			routines,
			order: kernels.map((r) => ({
				ms: round(r.ms, 3),
				size: Number(r.props.size ?? 0),
				shapes: shapeSig(r.props.exp ?? ""),
				op: (r.props.exp ?? "").slice(0, 60),
				passes: Number(r.props.passes ?? 0),
				source: r.props.source ?? "",
			})),
		};
	}

	return {
		config: { ctxLen, precision: "f32", ...info },
		benchRegion: {
			runs: wall,
			medianMs: wallMedian,
			minMs: Math.min(...wall),
			tokPerSec: round((ctxLen * 1000) / wallMedian, 2),
		},
		trace,
	};
}
