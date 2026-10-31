/**
 * After the projection fix: what is left, and where?
 *
 * The pre-fix accounting put 68 of 94 ms of GPU kernel time into the single
 * vocabulary-projection kernel, plus ~48 ms of GPU idle between the 469 submits,
 * plus ~16 ms of sampling. This runs the same accounting on the production path
 * (`runLogitsProjection`, which the tree now uses) so the new ranking is measured
 * rather than derived by subtraction.
 *
 * Same method, same caveats: kernel time from GPU timestamps (contention-immune),
 * wall clock with contention reported, entries clustered by trace batch so a
 * split flush cannot fabricate a span.
 */
import { blockUntilReady, defaultDevice, init, nn, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import {
	createGemma3State,
	runGemma3Step,
	runLogitsProjection,
	type Gemma3,
	type Gemma3State,
} from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

const round = (x: number, n = 2) => +x.toFixed(n);
const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	return s.length ? s[s.length >> 1] : NaN;
};

function disposeState(s: Gemma3State) {
	for (const c of s.kvCaches) {
		c.k.dispose();
		c.v.dispose();
	}
	s.positionTensor.dispose();
}

async function sampleTimed(logits: np.Array, topK = 64, topP = 0.95) {
	const t0 = performance.now();
	const scaled = logits.div(1.0);
	const idx = np.flip(scaled.ref.argsort()).slice([0, topK]);
	await blockUntilReady(idx);
	const probsArr = nn.softmax(scaled.slice(idx.ref));
	const probs = (await probsArr.data()) as Float16Array;
	let sum = 0;
	let cutoff = 0;
	for (let i = 0; i < probs.length; i++) {
		sum += probs[i];
		if (sum >= topP) {
			cutoff = i;
			break;
		}
	}
	void cutoff;
	await idx.slice(0).data();
	return round(performance.now() - t0);
}

export async function probe() {
	const steps = (globalThis as { __steps?: number }).__steps ?? 6;
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model = loadModelF32(safetensors.parse(await res.arrayBuffer())) as Gemma3 & {
		tokenEmbedT?: np.Array;
	};

	const workloads = await (await fetch("/bench/workloads.json")).json();
	const wl = workloads.decode.find((w: { name: string }) => w.name === "decode-ctx1024");
	const context: number[] = wl.tokens.slice(0, 1024);

	let st = createGemma3State(model);
	const pre = runGemma3Step(tree.ref(model), st, np.array(context, { dtype: np.uint32 }));
	await blockUntilReady(pre.latent);
	pre.latent.dispose();
	st = pre.state;

	let token = context[context.length - 1];
	for (let i = 0; i < 3; i++) {
		const o = runGemma3Step(tree.ref(model), st, np.array([token], { dtype: np.uint32 }));
		st = o.state;
		const lg = runLogitsProjection(model, o.latent.ref.slice([-1]).flatten());
		await blockUntilReady(lg);
		lg.dispose();
		o.latent.dispose();
		token = 1000 + (i % 7);
	}

	const { profiler } = await import("@jax-js/jax");
	const rows: Record<string, unknown>[] = [];

	for (let i = 0; i < steps; i++) {
		for (let j = 0; j < 5; j++) await new Promise((r) => setTimeout(r, 50));
		const before = (performance.getEntriesByType("measure") as PerformanceMeasure[]).length;

		const c0 = performance.now();
		profiler.startTrace();
		const out = runGemma3Step(tree.ref(model), st, np.array([token], { dtype: np.uint32 }));
		st = out.state;
		const logits = runLogitsProjection(model, out.latent.ref.slice([-1]).flatten());
		await blockUntilReady(logits);
		const benchMs = performance.now() - c0;
		const samplingMs = await sampleTimed(logits.ref);
		const genMs = performance.now() - c0;
		logits.dispose();
		out.latent.dispose();
		token = 1000 + (i % 7);

		for (let j = 0; j < 14; j++) await new Promise((r) => setTimeout(r, 50));
		profiler.stopTrace();
		for (let j = 0; j < 6; j++) await new Promise((r) => setTimeout(r, 50));

		let ents = (performance.getEntriesByType("measure") as PerformanceMeasure[])
			.slice(before)
			.filter((e) => /^(Kernel|Routine)\[/.test(e.name))
			.map((e) => ({ name: e.name, start: e.startTime, end: e.startTime + e.duration }))
			.sort((a, b) => a.start - b.start);
		if (ents.length > 1) {
			const clusters: { start: number; end: number }[][] = [];
			let cur = [ents[0]];
			for (let k = 1; k < ents.length; k++) {
				if (ents[k].start - cur[cur.length - 1].end > 50) {
					clusters.push(cur);
					cur = [ents[k]];
				} else cur.push(ents[k]);
			}
			clusters.push(cur);
			clusters.sort((a, b) => b.length - a.length);
			ents = clusters[0];
		}

		const kernelMs = ents.reduce((a, e) => a + (e.end - e.start), 0);
		const spanMs = ents.length ? ents[ents.length - 1].end - ents[0].start : 0;
		// The projection is the only kernel over the whole vocabulary.
		const projMs = ents
			.filter((e) => e.name === "Kernel[262K]")
			.reduce((a, e) => a + (e.end - e.start), 0);
		const byName = new Map<string, { count: number; ms: number }>();
		for (const e of ents) {
			const a = byName.get(e.name) ?? { count: 0, ms: 0 };
			a.count++;
			a.ms += e.end - e.start;
			byName.set(e.name, a);
		}

		rows.push({
			benchMs: round(benchMs),
			genMs: round(genMs),
			samplingMs,
			kernels: ents.length,
			kernelMs: round(kernelMs),
			spanMs: round(spanMs),
			gapMs: round(spanMs - kernelMs),
			projMs: round(projMs),
			byName: [...byName.entries()].map(([name, v]) => ({ name, count: v.count, ms: round(v.ms) })),
		});
	}

	disposeState(st);

	const merged = new Map<string, { count: number; ms: number[] }>();
	for (const r of rows) {
		for (const k of r.byName as { name: string; count: number; ms: number }[]) {
			const a = merged.get(k.name) ?? { count: k.count, ms: [] };
			a.ms.push(k.ms);
			merged.set(k.name, a);
		}
	}

	const benchMed = median(rows.map((r) => r.benchMs as number));
	const kernelMed = median(rows.map((r) => r.kernelMs as number));
	const projMed = median(rows.map((r) => r.projMs as number));
	const gapMed = median(rows.map((r) => r.gapMs as number));

	return {
		ctx: 1024,
		steps,
		benches: rows.map((r) => r.benchMs),
		medians: {
			benchMs: benchMed,
			genMs: median(rows.map((r) => r.genMs as number)),
			samplingMs: median(rows.map((r) => r.samplingMs as number)),
			kernelMs: kernelMed,
			projectionKernelMs: projMed,
			otherKernelMs: round(kernelMed - projMed),
			gapMs: gapMed,
			hostResidualMs: round(benchMed - kernelMed - gapMed),
			kernels: median(rows.map((r) => r.kernels as number)),
		},
		kernelsByName: [...merged.entries()]
			.map(([name, v]) => ({ name, count: v.count, ms: median(v.ms) }))
			.sort((a, b) => b.ms - a.ms),
		tokPerSec: round(1000 / benchMed, 2),
	};
}
