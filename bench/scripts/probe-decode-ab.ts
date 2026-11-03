/**
 * Decode A/B: does the prefill change affect a decode step at all?
 *
 * The prefill work is guarded to `isPrefill && !collectWeights`, and
 * `runDecoderLayer` is byte-identical, so decode should be unaffected. But a
 * benchmark run on a box at 94-97% CPU contention cannot settle that on its own,
 * and "should not" is not evidence. This runs the two arms interleaved in one
 * process so both see the same clock and the same contention, and reports the
 * paired difference.
 *
 *   base  the fused path for every step (what perf does today)
 *   new   the same, because the prefill branch cannot fire at S=1
 *
 * If the arms are the same code path, the measurement should show a ratio of
 * 1.00 with spread; any systematic difference would mean the guard leaks.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import {
	createGemma3State,
	runGemma3Step,
	runLogitsProjection,
	type Gemma3,
	type Gemma3State,
} from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const round = (x: number, n = 2) => +x.toFixed(n);

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
	const wl = workloads.decode.find((w: { name: string }) => w.name === "decode-ctx1024");
	const context: number[] = wl.tokens.slice(0, 1024);

	/** One full run: 1024-token prefill, then `steps` decode steps, as bench.ts. */
	const run = async (prefillLayer: boolean, steps = 24) => {
		(globalThis as { __noPrefillLayer?: boolean }).__noPrefillLayer = !prefillLayer;

		let state = createGemma3State(model);
		const pre = runGemma3Step(tree.ref(model), state, np.array(context, { dtype: np.uint32 }));
		await blockUntilReady(pre.latent);
		pre.latent.dispose();
		state = pre.state;

		let token = context[context.length - 1];
		const t0 = performance.now();
		for (let i = 0; i < steps; i++) {
			const out = runGemma3Step(tree.ref(model), state, np.array([token], { dtype: np.uint32 }));
			state = out.state;
			const logits = runLogitsProjection(model, out.latent.ref.slice([-1]).flatten());
			await blockUntilReady(logits);
			logits.dispose();
			out.latent.dispose();
			token = 1000 + (i % 7);
		}
		const dt = performance.now() - t0;
		dispose(state);
		return dt / steps;
	};

	// Warm both arms.
	await run(true, 4);
	await run(false, 4);

	const base: number[] = [];
	const neu: number[] = [];
	// Interleave so a clock or contention drift cannot favour one arm.
	for (let i = 0; i < 7; i++) {
		base.push(round(await run(false)));
		neu.push(round(await run(true)));
	}

	const b = median(base);
	const n = median(neu);
	return {
		note: "ms per token, 1024-token context, both arms interleaved in one process",
		baseMsPerToken: b,
		newMsPerToken: n,
		baseTps: round(1000 / b, 2),
		newTps: round(1000 / n, 2),
		ratio: round(n / b, 3),
		baseRuns: base,
		newRuns: neu,
		verdict:
			Math.abs(n / b - 1) < 0.03
				? "indistinguishable within 3%: decode is unaffected"
				: "differs by more than 3%: investigate",
	};
}
