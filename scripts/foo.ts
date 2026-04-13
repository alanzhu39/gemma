import { safetensors, tokenizers, WeightMapper } from "@jax-js/loaders";
import { readFileSync } from "node:fs";
import { fromSafetensors } from "$lib/gemma3/model";

/* eslint-disable @typescript-eslint/no-unused-vars */

function test_fromFile() {
	const weightMapper = new WeightMapper({
		autoCamelCase: true,
	});

	const buf = readFileSync("weights/model.f16.safetensors");
	const file = safetensors.parse(buf); // => { tensors: { ... } };
	const mappedWeights = weightMapper.mapObject(file.tensors);
	const weights = safetensors.toNested(mappedWeights);

	console.log(Object.keys(weights.model.layers[0]));

	for (const key of Object.keys(weights.model.layers[0])) {
		console.log(Object.keys(weights.model.layers[0][key]));
	}
}

function test_fromSafetensors() {
	const buf = readFileSync("weights/model.f16.safetensors");
	const file = safetensors.parse(buf);
	const weights = fromSafetensors(file);

	console.log(weights.layers[0]);
}

function test_tokenizer() {
	const data = readFileSync("tokenizer/tokenizer.model");
	const tokenizer = tokenizers.Unigram.fromBinary(data);

	const text = "Hello, world!";
	console.log("Tokens: ", tokenizer.encode(text));
}

test_tokenizer();
