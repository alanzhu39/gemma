import { safetensors, WeightMapper } from "@jax-js/loaders";
import { DType, jit, nn, numpy as np } from "@jax-js/jax";

// Model weight and layer interfaces

// Common interfaces

export type Linear = {
	weight: np.Array; // [out, in]
	bias?: np.Array; // [out]
};

export const runLinear = jit(function runLinear({ weight, bias }: Linear, x: np.Array): np.Array {
	x = np.dot(x, weight.transpose());
	if (bias) x = x.add(bias);
	return x;
});

export type RMSNorm = {
	gamma: np.Array;
};

export const runRMSNorm = jit(
	function runRMSNorm({ gamma }: RMSNorm, x: np.Array, eps: number = 1e-6): np.Array {
		// Gemma-style RMSNorm: out[i] = (input[i] / RMS(input)) * (weight[i] + 1)
		const dtype = x.dtype;
		x = x.astype(np.float32); // RMSNorm in high precision to avoid numerics issues.
		x = x
			.div(np.sqrt(np.square(x.ref).mean(-1, { keepdims: true }).add(eps)))
			.mul(gamma.astype(np.float32).add(1));
		return x.astype(dtype);
	},
	{
		staticArgnums: [2],
	},
);

export type KVCache = {
	k: np.Array; // [max_seq_len, head_dim]
	v: np.Array; // [max_seq_len, head_dim]
};

export function emptyKVCache(maxSeqLen: number, headDim: number): KVCache {
	return {
		k: np.zeros([maxSeqLen, headDim], { dtype: DType.Float32 }),
		v: np.zeros([maxSeqLen, headDim], { dtype: DType.Float32 }),
	};
}

// Gemma3 interfaces

export const MAX_CONTEXT_LEN = 32768;
const HEAD_DIM = 256;
export const NUM_HEADS = 4;
export const NUM_LAYERS = 18;

export type Gemma3State = {
	kvCaches: KVCache[];
	position: number;
	collectWeights: boolean;
	attentionWeights?: np.Array[];
};

export function createGemma3State(model: Gemma3, collectWeights = false): Gemma3State {
	return {
		kvCaches: model.layers.map((_, i) => emptyKVCache(isSlidingAttention(i) ? 512 : 0, 256)),
		position: 0,
		collectWeights,
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

const CACHE_EXPANSION_SIZE = 128;

export function runGemma3Step(
	{ tokenEmbed, layers, norm }: Gemma3,
	{ kvCaches, position, collectWeights }: Gemma3State,
	tokensAr: np.Array,
): { latent: np.Array; state: Gemma3State; attentionWeights?: np.Array[] } {
	// Token embedding weights unused here
	let x = runGemmaTextScaledWordEmbedding(tokenEmbed, tokensAr).astype(DType.Float32);

	const isPrefill = position === 0;
	const S = x.shape[0];
	const newContextLen = position + S;
	const newCapacity =
		newContextLen > kvCaches[0].k.shape[0]
			? Math.ceil(newContextLen / CACHE_EXPANSION_SIZE) * CACHE_EXPANSION_SIZE
			: kvCaches[0].k.shape[0];

	const attentionWeights: np.Array[] = [];

	const { slidingWindowSlots, slidingWindowMask, globalSlots, globalMask, sequencePositions } =
		precomputeSlotsAndMasks(position, S, newCapacity, isPrefill);

	const { cos, sin } = precomputeRoPECache(position, S, HEAD_DIM, isPrefill);

	for (let i = 0; i < layers.length; i++) {
		// If kv cache is not large enough, expand it to next multiple of CACHE_EXPANSION_SIZE.
		if (!isSlidingAttention(i) && newCapacity > kvCaches[i].k.shape[0]) {
			kvCaches[i].k = np.pad(kvCaches[i].k, {
				0: [0, newCapacity - kvCaches[i].k.shape[0]],
			});
			kvCaches[i].v = np.pad(kvCaches[i].v, {
				0: [0, newCapacity - kvCaches[i].v.shape[0]],
			});
		}

		const slotPositions = isSlidingAttention(i) ? slidingWindowSlots : globalSlots;
		const mask = isSlidingAttention(i) ? slidingWindowMask : globalMask;

		const out = runDecoderLayer(
			layers[i],
			kvCaches[i],
			slotPositions.ref,
			x,
			position,
			mask.ref,
			cos.ref,
			sin.ref,
			isPrefill,
			collectWeights,
		);
		x = out[0];
		kvCaches[i] = out[1];
		if (collectWeights) {
			attentionWeights.push(out[2]!);
		}
	}

	position = newContextLen;

	slidingWindowSlots.dispose();
	globalSlots.dispose();
	sequencePositions.dispose();
	slidingWindowMask.dispose();
	globalMask.dispose();

	return {
		latent: runRMSNorm(norm, x),
		state: {
			position,
			kvCaches,
			collectWeights,
		},
		...(collectWeights ? { attentionWeights } : {}),
	};
}

export const runGemmaTextScaledWordEmbedding = jit(
	function runGemmaTextScaledWordEmbedding(
		tokenEmbed: np.Array,
		tokensAr: np.Array,
		embedScale: number = 640 ** 0.5,
	): np.Array {
		return tokenEmbed.slice(tokensAr).mul(embedScale);
	},
	{
		staticArgnums: [2],
	},
);

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
		slotPositions: np.Array,
		x: np.Array,
		position: np.Array,
		mask: np.Array,
		cos: np.Array,
		sin: np.Array,
		isPrefill: boolean,
		collectWeights: boolean = false,
	): [np.Array, KVCache] | [np.Array, KVCache, np.Array] {
		let residual = x.ref;
		x = runRMSNorm(inputLayernorm, x);
		const out = runAttention(
			selfAttn,
			kvCache,
			slotPositions,
			x,
			position,
			mask,
			cos,
			sin,
			isPrefill,
			collectWeights,
		);
		x = out[0];
		x = runRMSNorm(postAttentionLayernorm, x);
		x = x.add(residual);
		// x = np.clip(x, -65504.0, 65504.0);

		residual = x.ref;
		x = runRMSNorm(preFeedforwardLayernorm, x);
		x = runMLP(mlp, x);
		x = runRMSNorm(postFeedforwardLayernorm, x);
		x = x.add(residual);
		// x = np.clip(x, -65504.0, 65504.0);

		if (collectWeights) {
			// Must have scores collected from attention
			return [x, out[1], out[2]!];
		}

		return [x, out[1]];
	},
	{ staticArgnums: [8, 9] },
);

export type Gemma3Attention = {
	qProj: Linear; // [1024, 640], no bias
	kProj: Linear; // [256, 640], no bias
	vProj: Linear; // [256, 640], no bias
	oProj: Linear; // [640, 1024], no bias
	qNorm: RMSNorm; // [256]
	kNorm: RMSNorm; // [256]
};

const precomputeSlotsAndMasks = jit(
	function precomputeSlotsAndMasks(
		position: np.Array,
		S: number,
		cacheCapacity: number,
		isPrefill: boolean,
	): {
		slidingWindowSlots: np.Array;
		slidingWindowMask: np.Array;
		globalSlots: np.Array;
		globalMask: np.Array;
		sequencePositions: np.Array;
	} {
		const swaCapacity = 512;

		const sequencePositions = np.arange(S).add(position.ref).reshape([S, 1]);

		const slidingWindowSlots = np
			.arange(swaCapacity)
			.sub(position.ref.add(S).mod(swaCapacity))
			.add(swaCapacity)
			.mod(swaCapacity)
			.add(position.ref.add(S).sub(swaCapacity)); // Cache slots mapped to their actual positions
		const globalSlots = np.arange(cacheCapacity);

		const slidingWindowMask = isPrefill
			? np.tri(S, S, 0, { dtype: DType.Bool }).notEqual(np.tri(S, S, -512))
			: slidingWindowSlots.ref
					.lessEqual(sequencePositions.ref)
					.mul(slidingWindowSlots.ref.greaterEqual(0));
		const globalMask = isPrefill
			? np.tri(S, S, 0, { dtype: DType.Bool })
			: globalSlots.ref.lessEqual(sequencePositions.ref).mul(globalSlots.ref.greaterEqual(0));

		return {
			slidingWindowSlots,
			slidingWindowMask,
			globalSlots,
			globalMask,
			sequencePositions,
		};
	},
	{
		staticArgnums: [1, 2, 3],
	},
);

const precomputeRoPECache = jit(
	function precomputeRoPECache(
		offset: np.Array,
		seqLen: number,
		headDim: number,
		isSlidingAttention: boolean,
	): {
		cos: np.Array;
		sin: np.Array;
	} {
		const base = isSlidingAttention ? 10000 : 1000000;

		const invFreq = np.exp(
			np.arange(0, headDim, 2, { dtype: DType.Float32 }).mul(-Math.log(base) / headDim),
		);
		const positions = np.arange(seqLen).add(offset);

		let angles = np.outer(positions, invFreq);
		angles = np.concatenate([angles.ref, angles], -1);

		const cos = np.cos(angles.ref).astype(DType.Float32);
		const sin = np.sin(angles).astype(DType.Float32);

		return {
			cos,
			sin,
		};
	},
	{
		staticArgnums: [1, 2, 3],
	},
);

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
	slotPositions: np.Array,
	x: np.Array, // [S, 640]
	position: np.Array, // scalar, current position in the sequence
	mask: np.Array, // [S, S], causal and sliding window attention mask
	cos: np.Array,
	sin: np.Array,
	isPrefill: boolean,
	collectWeights: boolean = false,
): [np.Array, KVCache] | [np.Array, KVCache, np.Array] {
	const S = x.shape[0];

	let q = runRMSNorm(qNorm, runLinear(qProj, x.ref).reshape([S, NUM_HEADS, HEAD_DIM])); // [S, 4, 256]
	let k = runRMSNorm(kNorm, runLinear(kProj, x.ref)); // [S, 256]
	let v = runLinear(vProj, x); // [S, 256]

	// Apply RoPE
	q = applyRoPE(q, cos.ref, sin.ref);
	k = applyRoPE(k, cos, sin);

	// Update KV cache
	const writeIndexes = slotPositions.ref
		.greaterEqual(position.ref)
		.mul(slotPositions.ref.sub(position.ref));
	const writeMask = slotPositions.ref
		.greaterEqual(position.ref)
		.mul(slotPositions.ref.less(position.ref.add(S)))
		.reshape([-1, 1]);
	kvCache.k = np.where(writeMask.ref, k.ref.slice(writeIndexes.ref), kvCache.k);
	kvCache.v = np.where(writeMask, v.ref.slice(writeIndexes), kvCache.v);

	if (!isPrefill) {
		k.dispose();
		v.dispose();

		k = kvCache.k.ref;
		v = kvCache.v.ref;
	}

	// [S, 4, 256] * [256, N] -> [4, S, N]
	const weights = nn.softmax(
		np.where(
			mask,
			np.einsum("SHD,ND->HSN", q, k).mul(1 / 16), // 1 / sqrt(headDim = 256)
			-Infinity,
		),
	);

	// [4, S, N] * [N, 256] -> [S, 4, 256]
	const a = np.einsum("HSN,ND->SHD", weights.ref, v);

	x = runLinear(oProj, a.reshape([S, NUM_HEADS * HEAD_DIM]));

	position.dispose();

	if (collectWeights) {
		return [x, kvCache, weights.slice([], [], [0, S])];
	} else {
		weights.dispose();
		return [x, kvCache];
	}
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
			hydrated[key] = np.array(new Float16Array(value.data as Float16Array<ArrayBuffer>), {
				dtype: np.float16,
				shape: value.shape,
			});
		} else {
			throw new Error(`Unexpected dtype ${value.dtype} for weight ${key}`);
		}
	}
	return safetensors.toNested(hydrated).model;
}
