import { safetensors, tokenizers, WeightMapper } from "@jax-js/loaders";
import { DType, numpy as np } from "@jax-js/jax";
import { readFileSync } from "node:fs";
import { fromSafetensors, runAttention, type Gemma3Attention } from "$lib/gemma3/model";
import { PreTrainedTokenizer } from "@huggingface/transformers";

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

function test_jaxjs_tokenizer() {
	const data = readFileSync("tokenizer/tokenizer.model");
	const tokenizer = tokenizers.Unigram.fromBinary(data);

	console.log(tokenizer);

	const text = "Plants create energy through a process known as";
	console.log("Tokens: ", tokenizer.encode(text));
}

async function test_transformers_tokenizer() {
	const tokenizer = await PreTrainedTokenizer.from_pretrained("./tokenizer/");

	const text = "Plants create energy through a process known as";
	console.log("Tokens: ", tokenizer.encode(text));
}

function test_dot() {
	const N = 10;
	const q = np.ones([N, 4, 32]);
	const k = np.ones([N, 32]);

	const q_kT = np.dot(q, k.transpose());
	console.log(q_kT.shape);
}

function test_mask() {
	const a = np.tri(5, 5, 0, { dtype: DType.Bool });
	const b = np.tri(5, 5, -2, { dtype: DType.Bool });
	console.log(np.notEqual(a, b).js());
}

function test_rotate_half() {
	// console.log(rotateHalf(np.arange(6)).js());
}

function test_attention() {
	const weights: Gemma3Attention = {
		qProj: { weight: np.ones([1024, 640]) },
		kProj: { weight: np.ones([256, 640]) },
		vProj: { weight: np.ones([256, 640]) },
		oProj: { weight: np.ones([640, 1024]) },
		qNorm: { gamma: np.ones([256]) },
		kNorm: { gamma: np.ones([256]) },
	};
	const x = np.ones([999, 640]);
	console.log(runAttention(weights, x, true).shape);
}

test_attention();
