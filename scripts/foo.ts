import { safetensors, WeightMapper } from "@jax-js/loaders";
import { readFileSync } from "node:fs";
import { fromSafetensors } from "$lib/gemma3/model";

// eslint-disable-next-line @typescript-eslint/no-unused-vars
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

test_fromSafetensors();
