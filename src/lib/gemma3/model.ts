import { safetensors, WeightMapper } from "@jax-js/loaders";
import { nn, numpy as np } from "@jax-js/jax";

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
	// RMSNorm: x * gamma / sqrt(var + eps)
	const var_ = np.var_(x.ref, -1, { correction: 0, keepdims: true });
	return x.mul(gamma).div(np.sqrt(var_.add(eps)));
}

// Gemma3 interfaces

export type Gemma3 = {
	tokenEmbed: np.Array;
	layers: Gemma3DecoderLayer[];
	norm: RMSNorm;
};

export function runGemma3Step({ tokenEmbed, layers, norm }: Gemma3, x: np.Array): np.Array {
	// Token embedding weights unused here
	tokenEmbed.dispose();

	// TODO(opt): unroll?
	for (const layer of layers) {
		x = runDecoderLayer(layer, x);
	}

	return runRMSNorm(norm, x);
}

export type Gemma3DecoderLayer = {
	inputLayerNorm: RMSNorm;
	selfAttn: Gemma3Attention;
	postAttentionLayernorm: RMSNorm;
	preFeedforwardLayernorm: RMSNorm;
	mlp: Gemma3MLP;
	postFeedforwardLayernorm: RMSNorm;
};

function runDecoderLayer(
	{
		inputLayerNorm,
		selfAttn,
		postAttentionLayernorm,
		preFeedforwardLayernorm,
		mlp,
		postFeedforwardLayernorm,
	}: Gemma3DecoderLayer,
	x: np.Array,
): np.Array {
	let residual = x.ref;
	x = runRMSNorm(inputLayerNorm, x);
	x = runAttention(selfAttn, x);
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
	qProj: Linear; // no bias
	kProj: Linear; // no bias
	vProj: Linear; // no bias
	oProj: Linear; // no bias
	kNorm: RMSNorm;
	qNorm: RMSNorm;
};

function runAttention(
	{ qProj, kProj, vProj, oProj, kNorm, qNorm }: Gemma3Attention,
	x: np.Array,
): np.Array {
	// TODO: RoPE, GQA, KV-cache
	return np.zerosLike(x);
}

export type Gemma3MLP = {
	downProj: Linear; // no bias
	gateProj: Linear; // no bias
	upProj: Linear; // no bias
};

function runMLP({ downProj, gateProj, upProj }: Gemma3MLP, x: np.Array): np.Array {
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
