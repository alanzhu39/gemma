/**
 * Does the restructured prefill layer compute the same thing as the fused one?
 *
 * Bisects the two changes that landed together: the layer being split into
 * non-jitted pieces (structure) and its matmuls moving to the tiled kernel
 * (arithmetic). Comparing all four combinations against the golden logits says
 * which one broke, instead of guessing.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { createGemma3State, runGemma3Step, runLogitsProjection } from "../../src/lib/gemma3/model";
import type { Gemma3 } from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

const TOKENS = [2, 651, 892, 4497, 603, 322, 1128, 571, 407, 4724, 235265];

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model: Gemma3 = loadModelF32(safetensors.parse(await res.arrayBuffer()));
	const golden = await (await fetch("/bench/goldens/f32-hardware.json")).json();

	const run = async (noTiled: boolean) => {
		(globalThis as { __noTiled?: boolean }).__noTiled = noTiled;
		const state = createGemma3State(model);
		const out = runGemma3Step(tree.ref(model), state, np.array(TOKENS, { dtype: np.uint32 }));
		const logits = runLogitsProjection(model, out.latent.ref.slice([-1]).flatten());
		const host = (await logits.ref.data()) as Float32Array;
		// Same 64 sampled indices the golden uses.
		const idx: number[] = golden.indices;
		const g: number[] = golden.logits;
		let scale = 0;
		for (const v of g) scale = Math.max(scale, Math.abs(v));
		let maxErr = 0;
		let maxAbs = 0;
		for (let i = 0; i < idx.length; i++) {
			const d = Math.abs(host[idx[i]] - g[i]);
			maxAbs = Math.max(maxAbs, d);
			maxErr = Math.max(maxErr, d / scale);
		}
		// Full-vocabulary argmax, and the top-2 margin so a near-tie is visible.
		let argmax = 0;
		for (let i = 1; i < host.length; i++) if (host[i] > host[argmax]) argmax = i;
		let top1 = -Infinity;
		let top2 = -Infinity;
		for (let i = 0; i < host.length; i++) {
			if (host[i] > top1) {
				top2 = top1;
				top1 = host[i];
			} else if (host[i] > top2) top2 = host[i];
		}
		logits.dispose();
		out.latent.dispose();
		for (const c of out.state.kvCaches) {
			c.k.dispose();
			c.v.dispose();
		}
		out.state.positionTensor.dispose();
		await blockUntilReady([]);
		return {
			matmul: noTiled ? "stock np.dot" : "tiled kernel",
			relErr: +maxErr.toExponential(3),
			maxAbs: +maxAbs.toExponential(3),
			argmax,
			top1: +top1.toFixed(4),
			top2: +top2.toFixed(4),
			margin: +(top1 - top2).toFixed(4),
			finite: [...host.subarray(0, 1000)].every((v) => Number.isFinite(v)),
		};
	};

	const tiled = await run(false);
	const stock = await run(true);
	return { goldenArgmax: golden.argmax, tiled, stock };
}
