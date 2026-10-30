/**
 * FP32 loader for the published FP16 weights.
 *
 * Used when the WebGPU adapter lacks the `shader-f16` feature (CPU fallbacks
 * such as SwiftShader, CI containers). Weights are widened once at load time;
 * everything downstream is unchanged, so the engine code under test is the
 * same on both paths.
 */
import { safetensors, WeightMapper } from "@jax-js/loaders";
import { numpy as np } from "@jax-js/jax";
import type { Gemma3 } from "../src/lib/gemma3/model";

const weightMapper = new WeightMapper({
	suffix: {
		".embed_tokens.weight": ".tokenEmbed",
		"norm.weight": "norm.gamma",
	},
	autoCamelCase: true,
});

export function loadModelF32(file: safetensors.File): Gemma3 {
	const mapped = weightMapper.mapObject(file.tensors);
	const hydrated: Record<string, np.Array> = {};
	for (const [key, value] of Object.entries(mapped)) {
		const src = value.data as Float16Array;
		hydrated[key] = np.array(Float32Array.from(src), {
			dtype: np.float32,
			shape: value.shape,
		});
	}
	const model = safetensors.toNested(hydrated).model as Gemma3;
	// Match the production loader: the contiguous transposed copy that the stock
	// projection path reads. Without it the f32 harness would measure a different
	// code path from the one the browser build runs.
	model.tokenEmbedT = model.tokenEmbed.ref.transpose().mul(1) as np.Array;
	return model;
}
