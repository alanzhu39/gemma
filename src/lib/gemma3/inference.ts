import { createGemma3State, runGemma3Step, runLinear, type Gemma3, type Linear } from "./model";
import { PreTrainedTokenizer } from "@huggingface/transformers";
import { nn, numpy as np, tree } from "@jax-js/jax";
import type { AttentionWeights } from "../../routes/inference/context";

/**
 * Runs a single inference step and returns:
 * - The full token keys array (input + predicted token).
 * - The attention activations for this pass.
 */
export function generateOnce(
	model: Gemma3,
	tokenizer: PreTrainedTokenizer,
	text: string,
): [number[], AttentionWeights] {
	const tokens = tokenizer.encode(text);
	const tokensAr = np.array(tokens, { dtype: np.uint32 });

	const collectWeights = true;
	const state = createGemma3State(model, collectWeights);
	const latent = runGemma3Step(tree.ref(model), state, tokensAr.ref);

	const outProj: Linear = {
		weight: model.tokenEmbed.ref,
	};
	const logits = runLinear(outProj, latent.slice([-1]));
	const predictedToken = np.argmax(nn.softmax(logits, 1), 1);

	const collectedWeights = state.attentionWeights;
	// Must have collected weights
	const attentionWeights = collectedWeights!.map((layerWeights: np.Array) => layerWeights.js());

	return [tokensAr.js().concat(predictedToken.js()), attentionWeights];
}

// Run inference for model
export async function runInference(
	model: Gemma3,
	tokenizer: PreTrainedTokenizer,
	text: string,
	steps = 1,
) {
	// Tokenize input
	// TODO: custom tokenizer
	const tokens = tokenizer.encode(text);

	console.log("Tokenized text");
	console.log(`Num tokens: ${tokens.length}`);

	// Embed tokens
	const tokensAr = np.array(tokens, { dtype: np.uint32 });

	console.log("Embedded tokens");

	// Run step(s)
	const state = createGemma3State(model);
	const generatedTokens: number[] = [];
	let nextInput: np.Array = tokensAr;
	for (let i = 0; i < steps; i++) {
		const latent = runGemma3Step(tree.ref(model), state, nextInput);

		// Project back to token space
		const outProj: Linear = {
			weight: model.tokenEmbed.ref,
		};
		const logits = runLinear(outProj, latent.slice([-1]));

		// Decode tokens
		const predictedToken = np.argmax(nn.softmax(logits, 1), 1);
		nextInput = predictedToken.ref;
		generatedTokens.push(...predictedToken.ref.js());

		console.log(`Ran step ${i + 1}`);
	}
	nextInput.dispose();

	const decoded = tokenizer.decode(generatedTokens);

	console.log(`${text} -> ${decoded}`);
}
