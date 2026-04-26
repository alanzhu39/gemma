import {
	createGemma3State,
	MAX_CONTEXT_LEN,
	runGemma3Step,
	runLinear,
	type Gemma3,
	type Linear,
} from "./model";
import { PreTrainedTokenizer } from "@huggingface/transformers";
import { nn, numpy as np, tree } from "@jax-js/jax";
import type { AttentionWeights } from "../../routes/inference/context";

const STOP_TOKEN_IDS = new Set([1, 106]); // <eos>, <end_of_turn>

/**
 * Returns token ID sampled from logits.
 */
async function sampleLogits(
	logits: np.Array,
	temperature: number,
	topK: number,
	topP: number,
): Promise<number> {
	const scaledLogits = logits.div(temperature);
	const topKIndices = np.flip(scaledLogits.ref.argsort()).slice([0, topK]);

	const probs = (await nn.softmax(scaledLogits.slice(topKIndices.ref)).data()) as Float16Array;
	const cumSum = new Float16Array(probs.length);
	let pCutoff = 0;
	let currSum = 0;
	// Compute cumsum, find cutoff index
	for (let i = 0; i < probs.length; i++) {
		currSum += probs[i];
		cumSum[i] = currSum;
		if (currSum >= topP) {
			pCutoff = i;
			break;
		}
	}

	// Uniform random a number between 0 and cumsum
	const rand = Math.random() * currSum;

	// Binary search for lowest cumsum
	let left = 0;
	let right = pCutoff;
	let probsIndex = 0;
	while (left <= right) {
		const mid = Math.floor((left + right) / 2);
		if (cumSum[mid] > rand) {
			probsIndex = mid;
			right = mid - 1;
		} else {
			left = mid + 1;
		}
	}

	return (await topKIndices.slice(probsIndex).data())[0];
}

export async function* streamGenerate(
	model: Gemma3,
	tokenizer: PreTrainedTokenizer,
	messages: Array<{ role: string; content: string }>,
	{
		temperature = 1.0,
		topK = 64,
		topP = 0.95,
	}: {
		temperature?: number;
		topK?: number;
		topP?: number;
	} = {},
): AsyncGenerator<string> {
	const prompt = tokenizer.apply_chat_template(messages, {
		tokenize: false,
		add_generation_prompt: true,
	}) as string;

	const tokens = tokenizer.encode(prompt);
	const maxTokens = MAX_CONTEXT_LEN - tokens.length;
	const tokensAr = np.array(tokens, { dtype: np.uint32 });

	const state = createGemma3State(model);

	let nextInput: np.Array = tokensAr;
	for (let i = 0; i < maxTokens; i++) {
		const latent = runGemma3Step(tree.ref(model), state, nextInput);
		const logits = runLinear({ weight: model.tokenEmbed.ref }, latent.slice([-1]).flatten());

		const tokenId = await sampleLogits(logits.ref, temperature, topK, topP);

		if (STOP_TOKEN_IDS.has(tokenId)) break;

		yield tokenizer.decode([tokenId], { skip_special_tokens: true });

		nextInput = np.array([tokenId]);
	}
}

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
	const logits = runLinear(outProj, latent.slice([-1]).flatten());
	const predictedToken = np.argmax(nn.softmax(logits));

	const collectedWeights = state.attentionWeights;
	// Must have collected weights
	const attentionWeights = collectedWeights!.map((layerWeights: np.Array) => layerWeights.js());

	return [tokensAr.js().concat(predictedToken.js()), attentionWeights];
}

// Run inference for model
export function runInference(
	model: Gemma3,
	tokenizer: PreTrainedTokenizer,
	text: string,
	steps = 1,
): number[] {
	// Tokenize input
	// TODO: custom tokenizer
	const tokens = tokenizer.encode(text);

	// Embed tokens
	const tokensAr = np.array(tokens, { dtype: np.uint32 });

	// Run step(s)
	const state = createGemma3State(model);
	const generatedTokens: number[] = [];
	let nextInput: np.Array = tokensAr;
	for (let i = 0; i < steps; i++) {
		const latent = runGemma3Step(tree.ref(model), state, nextInput);

		// Project back to token space
		const logits = runLinear({ weight: model.tokenEmbed.ref }, latent.slice([-1]).flatten());

		// Decode tokens
		const predictedToken = np.argmax(nn.softmax(logits));
		nextInput = predictedToken.ref.slice(null);
		generatedTokens.push(predictedToken.ref.js());

		if (i % 10 === 9) {
			console.log(`Ran step ${i + 1}`);
		}
	}
	nextInput.dispose();

	return generatedTokens;
}
