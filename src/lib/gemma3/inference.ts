import { safetensors } from "@jax-js/loaders";
import { readFileSync } from "node:fs";
import { fromSafetensors, runGemma3Step, type Gemma3 } from "./model";
import { PreTrainedTokenizer } from "@huggingface/transformers";
import { numpy as np, tree } from "@jax-js/jax";

// Run inference for model
async function runInference() {
	// Load weights
	const buf = readFileSync("weights/model.f16.safetensors");
	const file = safetensors.parse(buf);
	const weights: Gemma3 = fromSafetensors(file);

	// Tokenize input
	// TODO: custom tokenizer
	const tokenizer = await PreTrainedTokenizer.from_pretrained("./tokenizer/");
	const text = "Plants create energy through a process known as";
	const tokens = tokenizer.encode(text);

	// Embed tokens
	const tokensAr = np.array(tokens, { dtype: np.uint32 });
	const embed = weights.tokenEmbed.ref.slice(tokensAr);

	// Run step(s)
	const latent = runGemma3Step(tree.ref(weights), embed);

	// Project back to token space
}

runInference();
