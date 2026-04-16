import { safetensors, WeightMapper } from "@jax-js/loaders";
import { DType, nn, numpy as np } from "@jax-js/jax";

// Model weight and layer interfaces

// Common interfaces

export type Linear = {
	weight: np.Array; // [out, in]
	bias?: np.Array; // [out]
};

export function runLinear({ weight, bias }: Linear, x: np.Array): np.Array {
	x = np.dot(x, weight.transpose());
	if (bias) x = x.add(bias);
	return x;
}

export type RMSNorm = {
	gamma: np.Array;
};

export function runRMSNorm({ gamma }: RMSNorm, x: np.Array, eps: number = 1e-6): np.Array {
	// Gemma-style RMSNorm: out[i] = (input[i] / RMS(input)) * (weight[i] + 1)
	const dtype = x.dtype;
	x = x.astype(np.float32); // RMSNorm in high precision to avoid numerics issues.
	x = x
		.div(np.sqrt(np.square(x.ref).mean(-1, { keepdims: true }).add(eps)))
		.mul(gamma.astype(np.float32).add(1));
	return x.astype(dtype);
}

// Gemma3 interfaces

export type Gemma3 = {
	tokenEmbed: np.Array;
	layers: Gemma3DecoderLayer[];
	norm: RMSNorm;
};

export function runGemma3Step({ tokenEmbed, layers, norm }: Gemma3, tokensAr: np.Array): np.Array {
	// Token embedding weights unused here
	let x = runGemmaTextScaledWordEmbedding(tokenEmbed, tokensAr);

	for (let i = 0; i < layers.length; i++) {
		const isSlidingAttention = i != 5 && i != 11 && i != 17;
		x = runDecoderLayer(layers[i], x, isSlidingAttention);
	}

	return runRMSNorm(norm, x);
}

export function runGemmaTextScaledWordEmbedding(
	tokenEmbed: np.Array,
	tokensAr: np.Array,
	embedScale: number = 640 ** 0.5,
): np.Array {
	return tokenEmbed.slice(tokensAr).mul(embedScale);
}

export type Gemma3DecoderLayer = {
	inputLayernorm: RMSNorm;
	selfAttn: Gemma3Attention;
	postAttentionLayernorm: RMSNorm;
	preFeedforwardLayernorm: RMSNorm;
	mlp: Gemma3MLP;
	postFeedforwardLayernorm: RMSNorm;
};

export function runDecoderLayer(
	{
		inputLayernorm,
		selfAttn,
		postAttentionLayernorm,
		preFeedforwardLayernorm,
		mlp,
		postFeedforwardLayernorm,
	}: Gemma3DecoderLayer,
	x: np.Array,
	isSlidingAttention: boolean = false,
): np.Array {
	let residual = x.ref;
	x = runRMSNorm(inputLayernorm, x);
	x = runAttention(selfAttn, x, isSlidingAttention);
	x = runRMSNorm(postAttentionLayernorm, x);
	x = np.clip(x.add(residual), -65504.0, 65504.0);

	residual = x.ref;
	x = runRMSNorm(preFeedforwardLayernorm, x);
	x = runMLP(mlp, x);
	x = runRMSNorm(postFeedforwardLayernorm, x);
	x = np.clip(x.add(residual), -65504.0, 65504.0);

	return x;
}

export type Gemma3Attention = {
	qProj: Linear; // [1024, 640], no bias
	kProj: Linear; // [256, 640], no bias
	vProj: Linear; // [256, 640], no bias
	oProj: Linear; // [640, 1024], no bias
	qNorm: RMSNorm; // [256]
	kNorm: RMSNorm; // [256]
};

function precomputeRoPECache(
	S: number,
	headDim: number,
	base: number,
): {
	cos: np.Array;
	sin: np.Array;
} {
	const invFreq = np.divide(
		1,
		np.pow(base, np.arange(0, headDim, 2, { dtype: DType.Float32 }).div(headDim)),
	);
	const positions = np.arange(S);

	let angles = np.outer(positions, invFreq);
	angles = np.concatenate([angles.ref, angles], -1);

	const cos = np.cos(angles.ref);
	const sin = np.sin(angles);

	return {
		cos,
		sin,
	};
}

function rotateHalf(x: np.Array) {
	const half = x.shape.slice(-1)[0] / 2;
	const [x1, x2] = np.split(x, [half], -1);
	return np.concatenate([x2.neg(), x1], -1);
}

function applyRoPE(x: np.Array, cos: np.Array, sin: np.Array) {
	// x: (S, H, D) or (S, D); cos/sin: (S, D)
	// Broadcasting handles the head dim
	if (x.ndim == 3) {
		cos = np.expandDims(cos, 1); // (S, 1, D)
		sin = np.expandDims(sin, 1);
	}
	return x.ref.mul(cos).add(rotateHalf(x).mul(sin));
}

export function runAttention(
	{ qProj, kProj, vProj, oProj, kNorm, qNorm }: Gemma3Attention,
	x: np.Array, // [S, 640]
	isSlidingAttention: boolean = false,
): np.Array {
	// TODO: KV-cache
	const S = x.shape[0];
	const numHeads = 4;
	const headDim = 256;

	let q = runLinear(qProj, x.ref); // [S, 1024]
	q = runRMSNorm(qNorm, q.reshape([S, numHeads, headDim])); // [S, 4, 256]

	let k = runLinear(kProj, x.ref); // [S, 256]
	k = runRMSNorm(kNorm, k); // [S, 256]

	const base = isSlidingAttention ? 10000 : 1000000;
	const { cos, sin } = precomputeRoPECache(S, headDim, base);
	q = applyRoPE(q, cos.ref, sin.ref);
	k = applyRoPE(k, cos, sin);

	const v = runLinear(vProj, x); // [S, 256]

	// [S, 4, 256] * [256, S] -> [4, S, S]
	let scores = np.einsum("qhd,kd->hqk", q, k).mul(1 / 16); // 1 / sqrt(d_k = 256)
	let mask = np.tri(S, S, 0, { dtype: DType.Bool });
	if (isSlidingAttention) {
		mask = np.notEqual(mask, np.tri(S, S, -512, { dtype: DType.Bool }));
	}
	scores = np.where(mask, scores, -Infinity);

	// [4, S, S] * [S, 256] -> [S, 4, 256]
	const a = np.einsum("hqs,sv->qhv", nn.softmax(scores), v);

	return runLinear(oProj, a.reshape([S, numHeads * headDim]));
}

export type Gemma3MLP = {
	downProj: Linear; // no bias
	gateProj: Linear; // no bias
	upProj: Linear; // no bias
};

export function runMLP({ downProj, gateProj, upProj }: Gemma3MLP, x: np.Array): np.Array {
	return runLinear(downProj, nn.gelu(runLinear(gateProj, x.ref)).mul(runLinear(upProj, x)));
}

const weightMapper = new WeightMapper({
	suffix: {
		".embed_tokens.weight": ".tokenEmbed",
		"norm.weight": "norm.gamma",
	},
	autoCamelCase: true,
});

export function fromSafetensors(file: safetensors.File): Gemma3 {
	const mappedWeights = weightMapper.mapObject(file.tensors);
	const hydrated: Record<string, np.Array> = {};
	for (const [key, value] of Object.entries(mappedWeights)) {
		if (value.dtype === "F16") {
			hydrated[key] = np.array(new Float32Array(value.data as Float16Array<ArrayBuffer>), {
				dtype: np.float32,
				shape: value.shape,
			});
		} else {
			throw new Error(`Unexpected dtype ${value.dtype} for weight ${key}`);
		}
	}
	return safetensors.toNested(hydrated).model;
}
