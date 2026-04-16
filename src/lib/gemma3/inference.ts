import { runGemma3Step, runLinear, type Gemma3, type Linear } from "./model";
import { PreTrainedTokenizer } from "@huggingface/transformers";
import { nn, numpy as np, tree } from "@jax-js/jax";

// Run inference for model
export async function runInference(model: Gemma3, tokenizer: PreTrainedTokenizer, text: string) {
	// Tokenize input
	// TODO: custom tokenizer
	const tokens = tokenizer.encode(text);

	console.log("Tokenized text");

	// Embed tokens
	const tokensAr = np.array(tokens, { dtype: np.uint32 });

	console.log("Embedded tokens");

	// Run step(s)
	const latent = runGemma3Step(tree.ref(model), tokensAr);

	console.log("Ran step");

	// Project back to token space
	const outProj: Linear = {
		weight: model.tokenEmbed.ref,
	};
	const logits = runLinear(outProj, latent.slice([-1]));

	// Decode tokens
	const probs = nn.softmax(logits, 1);
	console.log(probs.ref.js());
	const predictedTokens = np.argmax(probs, 1);
	const decoded = tokenizer.decode(predictedTokens.js());

	console.log(`${text} -> ${decoded}`);
}
