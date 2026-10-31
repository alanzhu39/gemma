/**
 * Does the prefill branch ever execute during decode?
 *
 * A direct instrumentation rather than a timing comparison. The guard is
 * `isPrefill && !collectWeights && tiledPrefillApplies(S)`, and during decode
 * S is always 1, so the branch must never fire. Counting the calls is a proof;
 * a timing A/B on a box at 95% CPU contention is only evidence, and this session
 * already has one 45-66 ms spread on the *same* arm to prove it.
 *
 * Also reports min-of-runs for both arms, since under contention the minimum is
 * the sample least polluted by other processes.
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

	// Instrument: how many times does the tiled prefill layer run, and with what S?
	const calls: number[] = [];
	const g = globalThis as { __onPrefillLayer?: (s: number) => void };
	g.__onPrefillLayer = (s: number) => calls.push(s);

	const run = async (steps: number, prefillLayer: boolean) => {
		(globalThis as { __noPrefillLayer?: boolean }).__noPrefillLayer = !prefillLayer;
		calls.length = 0;
		let state = createGemma3State(model);
		const pre = runGemma3Step(tree.ref(model), state, np.array(context, { dtype: np.uint32 }));
		await blockUntilReady(pre.latent);
		pre.latent.dispose();
		state = pre.state;
		const prefillCalls = calls.length;
		const prefillS = [...new Set(calls)];

		let token = context[context.length - 1];
		calls.length = 0;
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
		const decodeCalls = calls.length;
		dispose(state);
		return { msPerToken: dt / steps, prefillCalls, prefillS, decodeCalls };
	};

	await run(4, true);
	await run(4, false);

	const base: number[] = [];
	const neu: number[] = [];
	let decodeCallsNew = -1;
	let prefillCallsNew = -1;
	for (let i = 0; i < 6; i++) {
		const b = await run(20, false);
		base.push(round(b.msPerToken));
		const n = await run(20, true);
		neu.push(round(n.msPerToken));
		decodeCallsNew = n.decodeCalls;
		prefillCallsNew = n.prefillCalls;
	}

	g.__onPrefillLayer = undefined;
	const minBase = Math.min(...base);
	const minNew = Math.min(...neu);
	return {
		tiledLayerCallsDuringPrefill_1024: prefillCallsNew,
		tiledLayerCallsDuringDecode: decodeCallsNew,
		prefillS: "[1024] expected; decode must be 0",
		baseMsPerToken: { runs: base, min: minBase, tpsAtMin: round(1000 / minBase, 2) },
		newMsPerToken: { runs: neu, min: minNew, tpsAtMin: round(1000 / minNew, 2) },
		ratioAtMin: round(minNew / minBase, 3),
		spreadNote: `same-arm spread: base ${round(Math.max(...base) / minBase, 2)}x, new ${round(Math.max(...neu) / minNew, 2)}x`,
	};
}
