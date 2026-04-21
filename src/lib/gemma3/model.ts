import { safetensors, WeightMapper } from "@jax-js/loaders";
import { DType, jit, nn, numpy as np } from "@jax-js/jax";

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
};

export function emptyKVCache(maxSeqLen: number, headDim: number): KVCache {
	return {
		k: np.zeros([maxSeqLen, headDim], { dtype: DType.Float16 }),
		v: np.zeros([maxSeqLen, headDim], { dtype: DType.Float16 }),
	};
}

// Gemma3 interfaces

export const MAX_CONTEXT_LEN = 32768;
const HEAD_DIM = 256;
export const NUM_HEADS = 4;

export type Gemma3State = {
	kvCaches: KVCache[];
	position: number;
	collectWeights: boolean;
	attentionWeights?: np.Array[];
};

export function createGemma3State(model: Gemma3, collectWeights = false): Gemma3State {
	return {
		kvCaches: model.layers.map((_, i) =>
			emptyKVCache(isSlidingAttention(i) ? 512 : MAX_CONTEXT_LEN, 256),
		),
		position: 0,
		collectWeights,
		...(collectWeights ? { attentionWeights: [] } : {}),
	};
}

export type Gemma3 = {
	tokenEmbed: np.Array;
	layers: Gemma3DecoderLayer[];
	norm: RMSNorm;
};

export function isSlidingAttention(i: number): boolean {
	return i != 5 && i != 11 && i != 17;
}

export function runGemma3Step(
	{ tokenEmbed, layers, norm }: Gemma3,
	state: Gemma3State,
	tokensAr: np.Array,
): np.Array {
	// Token embedding weights unused here
	let x = runGemmaTextScaledWordEmbedding(tokenEmbed, tokensAr);

	for (let i = 0; i < layers.length; i++) {
		const out = runDecoderLayer(
			layers[i],
			state.kvCaches[i],
			x,
			state.position,
			isSlidingAttention(i),
			state.collectWeights,
		);
		x = out[0];
		state.kvCaches[i] = out[1];
		if (state.collectWeights) {
			state.attentionWeights!.push(out[2]!);
		}
	}

	const S = x.shape[0];
	state.position = state.position + S;

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

export const runDecoderLayer = jit(
	function runDecoderLayer(
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
		position: number,
		isSlidingAttention: boolean = false,
		collectWeights: boolean = false,
	): [np.Array, KVCache] | [np.Array, KVCache, np.Array] {
		let residual = x.ref;
		x = runRMSNorm(inputLayernorm, x);
		const out = runAttention(selfAttn, kvCache, x, position, isSlidingAttention, collectWeights);
		x = out[0];
		x = runRMSNorm(postAttentionLayernorm, x);
		x = np.clip(x.add(residual), -65504.0, 65504.0);

		residual = x.ref;
		x = runRMSNorm(preFeedforwardLayernorm, x);
		x = runMLP(mlp, x);
		x = runRMSNorm(postFeedforwardLayernorm, x);
		x = np.clip(x.add(residual), -65504.0, 65504.0);

		if (collectWeights) {
			// Must have scores collected from attention
			return [x, out[1], out[2]!];
		}

		return [x, out[1]];
	},
	{ staticArgnums: [3, 4, 5] },
);

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
	position: number, // Current position in the sequence
	isSlidingAttention: boolean = false,
	collectWeights: boolean = false,
): [np.Array, KVCache] | [np.Array, KVCache, np.Array] {
	const S = x.shape[0];
	const N = kvCache.k.shape[0];
	const offset = position;
	const isPrefill = offset === 0;

	let q = runRMSNorm(qNorm, runLinear(qProj, x.ref).reshape([S, NUM_HEADS, HEAD_DIM])); // [S, 4, 256]
	let k = runRMSNorm(kNorm, runLinear(kProj, x.ref)); // [S, 256]
	const v = runLinear(vProj, x); // [S, 256]

	// Apply RoPE
	const base = isSlidingAttention ? 10000 : 1000000;
	const { cos, sin } = precomputeRoPECache(offset + S, HEAD_DIM, base);
	q = applyRoPE(q, cos.ref.slice([offset, offset + S]), sin.ref.slice([offset, offset + S]));
	k = applyRoPE(k, cos.slice([offset, offset + S]), sin.slice([offset, offset + S]));

	// Update KV cache
	const slotPositions = isSlidingAttention
		? np
				.arange(N)
				.sub((offset + S) % N)
				.add(N)
				.mod(N)
				.add(offset + S - N) // Cache slots mapped to their actual positions
		: np.arange(N);
	const writeIndexes = slotPositions.ref.greaterEqual(offset).mul(slotPositions.ref.sub(offset));
	const writeMask = slotPositions.ref
		.greaterEqual(offset)
		.mul(slotPositions.ref.less(offset + S))
		.reshape([-1, 1]);
	kvCache.k = np.where(writeMask.ref, k.ref.slice(writeIndexes.ref), kvCache.k);
	kvCache.v = np.where(writeMask, v.ref.slice(writeIndexes), kvCache.v);

	let scores: np.Array;
	if (isPrefill) {
		// Run with computed values
		let mask = np.tri(S, S, 0, { dtype: DType.Bool });
		if (isSlidingAttention) {
			mask = mask.notEqual(np.tri(S, S, -512));
		}

		// [S, 4, 256] * [256, N] -> [4, S, N]
		scores = np.where(mask, np.einsum("SHD,ND->HSN", q, k).mul(1 / 16), -Infinity); // 1 / sqrt(headDim = 256)

		// [4, S, N] * [N, 256] -> [S, 4, 256]
		const a = np.einsum("HSN,ND->SHD", nn.softmax(scores.ref), v);

		x = runLinear(oProj, a.reshape([S, NUM_HEADS * HEAD_DIM]));
	} else {
		k.dispose();
		v.dispose();

		// Run with cached values
		let mask: np.Array;
		if (isSlidingAttention) {
			// Compute ring buffer mask
			const sequencePositions = np.arange(offset, offset + S).reshape([S, 1]);
			mask = slotPositions.ref.lessEqual(sequencePositions).mul(slotPositions.greaterEqual(0));
		} else {
			mask = np.tri(S, N, offset, { dtype: DType.Bool });
		}

		// [S, 4, 256] * [256, N] -> [4, S, N]
		scores = np.where(mask, np.einsum("SHD,ND->HSN", q, kvCache.k.ref).mul(1 / 16), -Infinity); // 1 / sqrt(headDim = 256)

		// [4, S, N] * [N, 256] -> [S, 4, 256]
		const a = np.einsum("HSN,ND->SHD", nn.softmax(scores.ref), kvCache.v.ref);

		x = runLinear(oProj, a.reshape([S, NUM_HEADS * HEAD_DIM]));
	}

	if (collectWeights) {
		return [x, kvCache, scores.slice([], [], [0, S])];
	}

	scores.dispose();

	return [x, kvCache];
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
