import { safetensors, WeightMapper } from "@jax-js/loaders";
import { numpy as np } from "@jax-js/jax";

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
	// Reference impl:
	// RMSNorm: x * gamma / sqrt(var + eps)
	const dtype = x.dtype;
	x = x.astype(np.float32); // RMSNorm in high precision to avoid numerics issues.
	const var_ = np.var_(x.ref, -1, { correction: 0, keepdims: true });
	x = x.mul(gamma).div(np.sqrt(var_.add(eps)));
	return x.astype(dtype);
}

// Gemma3 interfaces

// TODO: runners

export type Gemma3 = {
	tokenEmbed: np.Array;
	layers: Gemma3DecoderLayer[];
	norm: RMSNorm;
};

export type Gemma3DecoderLayer = {
	inputLayerNorm: np.Array;
	selfAttn: Gemma3Attention;
	postAttentionLayernorm: RMSNorm;
	preFeedforwardLayernorm: RMSNorm;
	mlp: Gemma3MLP;
	postFeedforwardLayernorm: RMSNorm;
};

export type Gemma3Attention = {
	qProj: Linear; // no bias
	kProj: Linear; // no bias
	vProj: Linear; // no bias
	oProj: Linear; // no bias
	kNorm: RMSNorm;
	qNorm: RMSNorm;
};

export type Gemma3MLP = {
	downProj: Linear; // no bias
	gateProj: Linear; // no bias
	upProj: Linear; // no bias
};

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
			hydrated[key] = np.array(value.data as Float16Array<ArrayBuffer>, {
				dtype: np.float16,
				shape: value.shape,
			});
		} else {
			throw new Error(`Unexpected dtype ${value.dtype} for weight ${key}`);
		}
	}
	return safetensors.toNested(hydrated).model;
}
