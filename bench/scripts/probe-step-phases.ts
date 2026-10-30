/**
 * Where does a real decode step actually spend its 120 ms?
 *
 * Sampling turned out to cost only ~5.6 ms, so the obvious suspects are ruled
 * out. This times the real model step in phases, using the actual weights:
 *
 *   prefill (fills the KV cache) -> then for each decode step:
 *     - submit only (host time to issue the work)
 *     - blockUntilReady (wait for the GPU)
 *     - logits projection
 *     - readback
 *
 * The gap between "submit" and "wait" separates host-side cost from GPU time,
 * which is the split nobody has measured directly on hardware.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { createGemma3State, runGemma3Step, runLinear } from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model = loadModelF32(safetensors.parse(await res.arrayBuffer()));

	// Realistic context length, and it matters: 638 crosses the cache boundary.
	const ctx = new Array(640).fill(0).map((_, i) => 100 + (i % 200000));

	let st = createGemma3State(model);
	const t0 = performance.now();
	const pre = runGemma3Step(tree.ref(model), st, np.array(ctx, { dtype: np.uint32 }));
	await blockUntilReady(pre.latent);
	const prefillMs = performance.now() - t0;
	pre.latent.dispose();
	st = pre.state;

	const rows: Record<string, number>[] = [];
	for (let i = 0; i < 6; i++) {
		const tok = np.array([100 + i], { dtype: np.uint32 });

		const tA = performance.now();
		const out = runGemma3Step(tree.ref(model), st, tok);
		const tB = performance.now(); // host finished issuing
		await blockUntilReady(out.latent);
		const tC = performance.now(); // GPU finished the model

		// This second sync is required, not optional: the projection reads the
		// latent, so it cannot start until the model finishes. That means the
		// time below includes draining anything still queued, and is an upper
		// bound on the projection rather than its own cost. Isolate it by timing
		// the same projection twice in a row on already-resident data.
		const logits = runLinear(
			{ weight: tree.ref(model).tokenEmbed },
			out.latent.ref.slice([-1]).flatten(),
		);
		const tD = performance.now(); // host finished issuing projection
		await blockUntilReady(logits);
		const tE = performance.now(); // GPU finished projection
		await logits.data();
		const tF = performance.now(); // readback done

		// Isolated projection: same matvec, second time, nothing queued ahead.
		const lat2 = out.latent.ref.slice([-1]).flatten();
		const iso0 = performance.now();
		const isoLogits = runLinear({ weight: tree.ref(model).tokenEmbed }, lat2);
		const iso1 = performance.now();
		await blockUntilReady(isoLogits);
		const iso2 = performance.now();
		isoLogits.dispose();

		st = out.state;
		out.latent.dispose();

		rows.push({
			step: i,
			hostIssueModel: +(tB - tA).toFixed(1),
			gpuWaitModel: +(tC - tB).toFixed(1),
			hostIssueLogits: +(tD - tC).toFixed(1),
			gpuWaitLogits: +(tE - tD).toFixed(1),
			readback: +(tF - tE).toFixed(1),
			// Isolated projection: nothing queued ahead of it.
			isolatedLogitsIssue: +(iso1 - iso0).toFixed(1),
			isolatedLogitsWait: +(iso2 - iso1).toFixed(1),
			total: +(tF - tA).toFixed(1),
		});
	}
	return { prefillMs: +prefillMs.toFixed(1), rows };
}
