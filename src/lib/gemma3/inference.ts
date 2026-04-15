import { safetensors } from "@jax-js/loaders";
import { readFileSync } from "node:fs";
import { fromSafetensors, runGemma3Step, runLinear, type Gemma3, type Linear } from "./model";
import { PreTrainedTokenizer } from "@huggingface/transformers";
import { init, nn, numpy as np, tree } from "@jax-js/jax";

// Run inference for model
async function runInference() {
	console.log(`Available devices: ${await init()}`);

	// Load weights
	const buf = readFileSync("weights/model.f16.safetensors");
	const file = safetensors.parse(buf);
	const weights: Gemma3 = fromSafetensors(file);

	console.log("Loaded weights");

	// Tokenize input
	// TODO: custom tokenizer
	const tokenizer = await PreTrainedTokenizer.from_pretrained("./tokenizer/");
	const text = "Plants create energy through a process known as";
	const tokens = tokenizer.encode(text);

	console.log("Tokenized text");

	// Embed tokens
	const tokensAr = np.array(tokens, { dtype: np.uint32 });
	const embed = weights.tokenEmbed.ref.slice(tokensAr);

	console.log("Embedded tokens");

	// Run step(s)
	const latent = runGemma3Step(tree.ref(weights), embed);

	console.log("Ran step");

	// Project back to token space
	const outProj: Linear = {
		weight: weights.tokenEmbed.ref,
	};
	console.log(outProj.weight.shape, latent.shape);
	const logits = runLinear(outProj, latent);

	// Decode tokens
	const probs = nn.softmax(logits, 1);
	const predictedTokens = np.argmax(probs, 1);
	const decoded = tokenizer.decode(predictedTokens.js());

	console.log(`${text} -> ${decoded}`);
}

runInference();
