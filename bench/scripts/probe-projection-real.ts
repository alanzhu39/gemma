/**
 * The decisive projection measurement: real buffers, both layouts.
 *
 * Establishes why every prior probe disagreed. In one process, GPU timestamps,
 * same weight, same single kernel:
 *
 *   np.dot(np.ones([640]),            W.T)   24.1 ms   27.9 GB/s   <- LITERAL
 *   np.dot(np.array([1,1,...]),       W.T)   67.9 ms    9.9 GB/s   <- real buffer
 *   np.dot(step latent,               W.T)   64.8 ms   10.4 GB/s   <- real buffer
 *
 * `np.ones` is not a buffer in jax-js: it is a `full` literal in the expression
 * graph, so a kernel that consumes it folds the input away and never reads it.
 * The 24 ms figure is the strided weight read alone; the 68 ms figure is what a
 * real step pays. That is 2.8x, and it is the reason this project has "measured"
 * the projection at both 27 ms and 73 ms.
 *
 * So this probe re-runs the layout comparison with real buffers only, which is the
 * only comparison that can justify a fix:
 *
 *   A  strided   x real   np.dot(x, W.transpose())   what runLinear does today
 *   B  contiguous x real  np.dot(x, Wt)              Wt = W.T.mul(1)
 *
 * Both use np.array() inputs (never ones), both verified to agree element-wise,
 * both traced for GPU kernel time, and both repeated with the order alternated.
 */
import { blockUntilReady, defaultDevice, init, jit, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { createGemma3State, runGemma3Step, runLinear, type Gemma3 } from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

const round = (x: number, n = 2) => +x.toFixed(n);

async function traceProj(body: () => Promise<void>) {
	const { profiler } = await import("@jax-js/jax");
	for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 50));
	const before = (performance.getEntriesByType("measure") as PerformanceMeasure[]).length;
	profiler.startTrace();
	await body();
	for (let i = 0; i < 14; i++) await new Promise((r) => setTimeout(r, 50));
	profiler.stopTrace();
	for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 50));
	const ents = (performance.getEntriesByType("measure") as PerformanceMeasure[])
		.slice(before)
		.filter((e) => /^(Kernel|Routine)\[/.test(e.name));
	return {
		projMs: round(ents.filter((e) => e.name === "Kernel[262K]").reduce((a, e) => a + e.duration, 0)),
		kernels: ents.length,
	};
}

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model = loadModelF32(safetensors.parse(await res.arrayBuffer())) as Gemma3 & {
		tokenEmbedT?: np.Array;
	};
	const W = model.tokenEmbed;
	const V = W.shape[0];
	const D = W.shape[1];
	const bytes = V * D * 4;
	const Wt = model.tokenEmbedT ?? (W.ref.transpose().mul(1) as np.Array);

	// A REAL latent input from a real step, round-tripped through the host so it is
	// unambiguously a buffer and not an expression.
	const workloads = await (await fetch("/bench/workloads.json")).json();
	const wl = workloads.decode.find((w: { name: string }) => w.name === "decode-ctx1024");
	const context: number[] = wl.tokens.slice(0, 1024);
	let st = createGemma3State(model);
	const pre = runGemma3Step(tree.ref(model), st, np.array(context, { dtype: np.uint32 }));
	await blockUntilReady(pre.latent);
	pre.latent.dispose();
	st = pre.state;
	for (let i = 0; i < 3; i++) {
		const o = runGemma3Step(tree.ref(model), st, np.array([1000 + i], { dtype: np.uint32 }));
		await blockUntilReady(o.latent);
		o.latent.dispose();
		st = o.state;
	}
	const out = runGemma3Step(tree.ref(model), st, np.array([1005], { dtype: np.uint32 }));
	await blockUntilReady(out.latent);
	const latentVals = Array.from((await out.latent.ref.data()) as Float32Array);
	const x = np.array(latentVals, { dtype: np.float32 });
	await blockUntilReady(x);

	// Verify the two layouts produce identical logits on this exact input.
	const a0 = np.dot(x.ref, W.ref.transpose());
	const b0 = np.dot(x.ref, Wt.ref);
	const [hA, hB] = (await Promise.all([a0.data(), b0.data()])) as [Float32Array, Float32Array];
	let worst = 0;
	let amaxA = 0;
	let amaxB = 0;
	let scale = 0;
	for (let i = 0; i < hA.length; i++) {
		if (hA[i] > hA[amaxA]) amaxA = i;
		if (hB[i] > hB[amaxB]) amaxB = i;
		worst = Math.max(worst, Math.abs(hA[i] - hB[i]));
		scale = Math.max(scale, Math.abs(hA[i]));
	}

	// Warm both.
	for (const a of [np.dot(x.ref, W.ref.transpose()), np.dot(x.ref, Wt.ref)]) {
		await blockUntilReady(a);
		a.dispose();
	}

	const strided: number[] = [];
	const contig: number[] = [];
	// Alternate order so any drift hits both equally.
	for (let i = 0; i < 3; i++) {
		const order = i % 2 === 0 ? ["s", "c"] : ["c", "s"];
		for (const k of order) {
			const r = await traceProj(async () => {
				const a =
					k === "s" ? np.dot(x.ref, W.ref.transpose()) : np.dot(x.ref, Wt.ref);
				await blockUntilReady(a);
				a.dispose();
			});
			(k === "s" ? strided : contig).push(r.projMs);
		}
	}

	const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[xs.length >> 1];
	const sMed = median(strided);
	const cMed = median(contig);

	x.dispose();
	out.latent.dispose();
	for (const c of out.state.kvCaches) {
		c.k.dispose();
		c.v.dispose();
	}
	out.state.positionTensor.dispose();
	if (!model.tokenEmbedT) Wt.dispose();

	const gbps = (ms: number) => round(bytes / 1e9 / (ms / 1000), 1);
	if (Math.max(gbps(sMed), gbps(cMed)) > 192) throw new Error("above peak");

	return {
		weights: { shape: [V, D], mib: round(bytes / 2 ** 20, 0) },
		inputIsRealBuffer: true,
		strided: { values: strided, median: sMed, gbps: gbps(sMed), note: "np.dot(x, W.T): W.T is a view" },
		contiguous: { values: contig, median: cMed, gbps: gbps(cMed), note: "np.dot(x, Wt): Wt contiguous" },
		ratio: round(sMed / Math.max(1e-9, cMed), 2),
		verify: {
			maxAbsDiff: +worst.toExponential(2),
			logitScale: round(scale),
			relative: +(worst / Math.max(1e-9, scale)).toExponential(2),
			argmaxMatch: amaxA === amaxB,
		},
	};
}
