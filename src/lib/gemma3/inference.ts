import { safetensors } from "@jax-js/loaders";
import { readFileSync } from "node:fs";
import { fromSafetensors, type Gemma3 } from "./model";
import { PreTrainedTokenizer } from "@huggingface/transformers";

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
	// Slice weights.tokenEmbed

	// Decoder layers foward

	// Final RMS norm

	// Project back to token space
}

runInference();
