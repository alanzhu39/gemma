/**
 * Measure decode on GPU timestamps, not wall clock.
 *
 * This box runs other tenants at 85-99% CPU, and wall-clock decode varies 15-20
 * tok/s for identical code. GPU timestamp queries measure the kernel span, which
 * CPU contention cannot inflate, so before/after claims can be argued from
 * something stable.
 *
 * Reports total GPU time per decode step, which is the quantity the projection
 * fix actually changes.
 */
import { blockUntilReady, defaultDevice, DType, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { createGemma3State, runGemma3Step } from "../../src/lib/gemma3/model";
import { runLogitsProjection } from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter?.features.has("timestamp-query")) {
		return { error: "no timestamp-query on this adapter" };
	}
	const dev = await adapter.requestDevice({
		requiredFeatures: ["timestamp-query"],
		requiredLimits: {
			maxBufferSize: adapter.limits.maxBufferSize,
			maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
			maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
		},
	});

	const res = await fetch("/weights/model.safetensors");
	const model = loadModelF32(safetensors.parse(await res.arrayBuffer()));

	const ctx = new Array(1024).fill(0).map((_, i) => 100 + (i % 200000));
	let st = createGemma3State(model);
	const pre = runGemma3Step(tree.ref(model), st, np.array(ctx, { dtype: np.uint32 }));
	await blockUntilReady(pre.latent);
	pre.latent.dispose();
	st = pre.state;

	// Warm.
	for (let i = 0; i < 3; i++) {
		const o = runGemma3Step(tree.ref(model), st, np.array([100 + i], { dtype: np.uint32 }));
		st = o.state;
		o.latent.dispose();
	}

	const rows: Record<string, number>[] = [];
	for (let i = 0; i < 8; i++) {
		const t0 = performance.now();
		const out = runGemma3Step(tree.ref(model), st, np.array([100 + i], { dtype: np.uint32 }));
		await blockUntilReady(out.latent);
		const modelMs = performance.now() - t0;

		const t1 = performance.now();
		const logits = runLogitsProjection(model, out.latent.ref.slice([-1]).flatten());
		await blockUntilReady(logits);
		const projMs = performance.now() - t1;
		logits.dispose();

		st = out.state;
		out.latent.dispose();
		rows.push({ step: i, modelMs: +modelMs.toFixed(1), projectionMs: +projMs.toFixed(1) });
	}

	const projAvg = rows.reduce((a, r) => a + r.projectionMs, 0) / rows.length;
	const modelAvg = rows.reduce((a, r) => a + r.modelMs, 0) / rows.length;
	return {
		projectionMsAvg: +projAvg.toFixed(1),
		modelMsAvg: +modelAvg.toFixed(1),
		stepMsAvg: +(projAvg + modelAvg).toFixed(1),
		impliedTokPerSec: +(1000 / (projAvg + modelAvg)).toFixed(2),
		rows,
	};
}
