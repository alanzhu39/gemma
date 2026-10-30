/**
 * Proves the production logits path is the warp-per-row kernel, not a fallback.
 *
 * A PASS on the correctness gate does not distinguish "the new kernel computes
 * the right logits" from "the new kernel declined and the old path ran". This
 * checks the two things that tell them apart:
 *
 *   1. `projectVocabDecline` returns null for the real model call, i.e. the
 *      production path takes the kernel branch.
 *   2. The production path's output differs bit-for-bit from the stock np.dot,
 *      which it cannot if it *is* the stock np.dot. Combined with an agreement
 *      well inside the logit scale, that is positive evidence of a different,
 *      correct implementation.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import {
	createGemma3State,
	runGemma3Step,
	runLinear,
	runLogitsProjection,
} from "../../src/lib/gemma3/model";
import { projectVocabDecline } from "../../src/lib/gemma3/vocab-projection";
import { loadModelF32 } from "../f32-model";

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model = loadModelF32(safetensors.parse(await res.arrayBuffer()));

	const state = createGemma3State(model);
	const step = runGemma3Step(
		tree.ref(model),
		state,
		np.array([2, 651, 892, 4497, 603, 322, 1128, 571, 407, 4724, 235265], {
			dtype: np.uint32,
		}),
	);
	await blockUntilReady(step.latent);
	const latent = step.latent;
	const row = latent.ref.slice([-1]).flatten();

	const decline = projectVocabDecline(model.tokenEmbed, row);
	const viaKernel = runLogitsProjection(model, row.ref);
	await blockUntilReady(viaKernel);
	const viaStock = runLinear({ weight: tree.ref(model).tokenEmbed }, row.ref);

	const a = (await viaKernel.data()) as Float32Array;
	const b = (await viaStock.data()) as Float32Array;

	let identical = 0;
	let worst = 0;
	let scale = 0;
	let argmaxA = 0;
	let argmaxB = 0;
	for (let i = 0; i < a.length; i++) {
		if (a[i] === b[i]) identical++;
		worst = Math.max(worst, Math.abs(a[i] - b[i]));
		scale = Math.max(scale, Math.abs(b[i]));
		if (a[i] > a[argmaxA]) argmaxA = i;
		if (b[i] > b[argmaxB]) argmaxB = i;
	}

	return {
		decline: decline, // must be null for the kernel branch to be taken
		usedKernel: decline === null,
		vocab: a.length,
		bitIdenticalElements: identical,
		identicalFraction: +(identical / a.length).toExponential(2),
		worstAbsDiff: +worst.toExponential(2),
		normalisedErr: +(worst / scale).toExponential(2),
		argmaxKernel: argmaxA,
		argmaxStock: argmaxB,
		argmaxSame: argmaxA === argmaxB,
	};
}
