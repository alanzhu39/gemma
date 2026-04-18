import { safetensors, tokenizers, WeightMapper } from "@jax-js/loaders";
import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { readFileSync } from "node:fs";
import {
	emptyKVCache,
	fromSafetensors,
	MAX_CONTEXT_LEN,
	runAttention,
	type Gemma3,
	type Gemma3Attention,
} from "$lib/gemma3/model";
import { runInference } from "$lib/gemma3/inference";
import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";

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
	const offset = 1;
	const S = 2;
	const N = 4;
	const mask = np
		.arange(N)
		.sub((offset + S) % N)
		.add(N)
		.mod(N)
		.add(offset + S - N);
	// console.log(mask.js());
	const sequencePositions = np.arange(offset, offset + S).reshape([S, 1]);
	const slotPositions = np
		.arange(N)
		.sub((offset + S) % N)
		.add(N)
		.mod(N)
		.add(offset + S - N); // Cache slots mapped to their actual positions
	console.log(sequencePositions.ref.js());
	console.log(slotPositions.ref.js());
	console.log(
		slotPositions.ref.lessEqual(sequencePositions).mul(slotPositions.greaterEqual(0)).js(),
	);
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
	const isSlidingAttention = true;
	const kvCache = emptyKVCache(isSlidingAttention ? 512 : MAX_CONTEXT_LEN, 256);
	const x = np.ones([999, 640]);
	console.log(runAttention(weights, kvCache, x, 0, isSlidingAttention)[0].shape);
}

// Don't use, way too slow!
async function test_local_inference() {
	const devices = await init("cpu");
	defaultDevice("cpu");

	console.log("Running...");

	const data = readFileSync("weights/model.f16.safetensors");
	const file = safetensors.parse(data);
	const weights: Gemma3 = fromSafetensors(file);
	const text = "Plants create energy through a process known as";

	console.log("Loaded weights");

	const tokenizer = await AutoTokenizer.from_pretrained("./tokenizer/");

	await runInference(weights, tokenizer, text);

	tree.dispose(weights);
}

function test_modules() {
	const buf = readFileSync("weights/model.f16.safetensors");
	const file = safetensors.parse(buf); // => { tensors: { ... } };
	const weights = fromSafetensors(file);

	console.log(weights.layers[0].inputLayernorm.gamma.js());
}

function test_reshape() {
	const positions = np.arange(10).reshape([-1, 1]);
	// console.log(positions.js());

	// const x = np.arange(10).reshape([1, 10]);
	// const y = np.tile(x, [3, 1]);
	// console.log(y.js());

	const x = np.arange(10).sub(np.arange(5).reshape([5, 1]));
	console.log(x.js());
}

test_mask();
