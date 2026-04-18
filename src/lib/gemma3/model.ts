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

export type KVCache = {
	k: np.Array; // [max_seq_len, head_dim]
	v: np.Array; // [max_seq_len, head_dim]
	// Position might be > capacity, eg. for sliding attention layers we only need 512 slots of lookback.
	// Cache offset can be calculated as position % capacity.
	position: number;
};

export function emptyKVCache(maxSeqLen: number, headDim: number): KVCache {
	return {
		k: np.zeros([maxSeqLen, headDim], { dtype: DType.Float16 }),
		v: np.zeros([maxSeqLen, headDim], { dtype: DType.Float16 }),
		position: 0,
	};
}

// Gemma3 interfaces

export const MAX_CONTEXT_LEN = 32768;
const HEAD_DIM = 256;
const NUM_HEADS = 4;

export type Gemma3State = {
	kvCaches: KVCache[];
};

export function createGemma3State(model: Gemma3): Gemma3State {
	return {
		kvCaches: model.layers.map((_, i) =>
			emptyKVCache(isSlidingAttention(i) ? 512 : MAX_CONTEXT_LEN, 256),
		),
	};
}

export type Gemma3 = {
	tokenEmbed: np.Array;
	layers: Gemma3DecoderLayer[];
	norm: RMSNorm;
};

function isSlidingAttention(i: number): boolean {
	return i != 5 && i != 11 && i != 17;
}

export function runGemma3Step(
	{ tokenEmbed, layers, norm }: Gemma3,
	{ kvCaches }: Gemma3State,
	tokensAr: np.Array,
): np.Array {
	// Token embedding weights unused here
	let x = runGemmaTextScaledWordEmbedding(tokenEmbed, tokensAr);

	for (let i = 0; i < layers.length; i++) {
		[x, kvCaches[i]] = runDecoderLayer(layers[i], kvCaches[i], x, isSlidingAttention(i));
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
	kvCache: KVCache,
	x: np.Array,
	isSlidingAttention: boolean = false,
): [np.Array, KVCache] {
	let residual = x.ref;
	x = runRMSNorm(inputLayernorm, x);
	[x, kvCache] = runAttention(selfAttn, kvCache, x, isSlidingAttention);
	x = runRMSNorm(postAttentionLayernorm, x);
	x = np.clip(x.add(residual), -65504.0, 65504.0);

	residual = x.ref;
	x = runRMSNorm(preFeedforwardLayernorm, x);
	x = runMLP(mlp, x);
	x = runRMSNorm(postFeedforwardLayernorm, x);
	x = np.clip(x.add(residual), -65504.0, 65504.0);

	return [x, kvCache];
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
	offset: number,
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
	const positions = np.arange(offset);

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
	kvCache: KVCache,
	x: np.Array, // [S, 640]
	isSlidingAttention: boolean = false,
): [np.Array, KVCache] {
	const S = x.shape[0];
	const N = kvCache.k.shape[0];
	const offset = kvCache.position;

	let q = runRMSNorm(qNorm, runLinear(qProj, x.ref).reshape([S, NUM_HEADS, HEAD_DIM])); // [S, 4, 256]
	let k = runRMSNorm(kNorm, runLinear(kProj, x.ref)); // [S, 256]
	const v = runLinear(vProj, x); // [S, 256]

	// Apply RoPE
	const base = isSlidingAttention ? 10000 : 1000000;
	const { cos, sin } = precomputeRoPECache(offset + S, HEAD_DIM, base);
	q = applyRoPE(q, cos.ref.slice([offset, offset + S]), sin.ref.slice([offset, offset + S]));
	k = applyRoPE(k, cos.slice([offset, offset + S]), sin.slice([offset, offset + S]));

	// Update KV cache
	const positions = np.arange(N);
	const writeMask = positions.ref
		.greaterEqual(offset % N)
		.mul(positions.less((offset + S) % N))
		.reshape([-1, 1]);
	kvCache.k = np.where(
		writeMask.ref,
		np.tile(k, [Math.ceil(N / S), 1]).slice([0, N], []),
		kvCache.k,
	);
	kvCache.v = np.where(writeMask, np.tile(v, [Math.ceil(N / S), 1]).slice([0, N], []), kvCache.v);
	kvCache.position = offset + S;

	// [S, 4, 256] * [256, N] -> [4, S, N]
	let scores = np.einsum("SHD,ND->HSN", q, kvCache.k.ref).mul(1 / 16); // 1 / sqrt(headDim = 256)
	let mask = np.tri(S, N, offset, { dtype: DType.Bool });
	if (isSlidingAttention) {
		// Compute ring buffer mask
		const sequencePositions = np.arange(offset, offset + S).reshape([S, 1]);
		const slotPositions = np
			.arange(N)
			.sub((offset + S) % N)
			.add(N)
			.mod(N)
			.add(offset + S - N); // Cache slots mapped to their actual positions
		mask = slotPositions.ref.lessEqual(sequencePositions).mul(slotPositions.greaterEqual(0));
	}
	scores = np.where(mask, scores, -Infinity);

	// [4, S, N] * [N, 256] -> [S, 4, 256]
	const a = np.einsum("HSN,ND->SHD", nn.softmax(scores), kvCache.v.ref);

	return [runLinear(oProj, a.reshape([S, NUM_HEADS * HEAD_DIM])), kvCache];
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
