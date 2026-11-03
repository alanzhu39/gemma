/**
 * Decode-path correctness check (complements bench/check.ts, which only
 * exercises a single short prefill).
 *
 * Prefills `context`, then teacher-forces `decode` one token at a time through
 * the S=1 decode path, recording sampled logits + argmax at every step. Finally
 * it re-runs the whole sequence as one prefill and compares the last position
 * against the last decode step, so decode is checked against an independent
 * code path even without a stored reference.
 *
 * Choose lengths that cross the interesting boundaries: position 512 (sliding
 * window wrap) and multiples of the KV-cache expansion size (global growth).
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import {
	createGemma3State,
	fromSafetensors,
	runGemma3Step,
	runLogitsProjection,
} from "../src/lib/gemma3/model";
import type { Gemma3, Gemma3State } from "../src/lib/gemma3/model";
import { loadModelF32 } from "./f32-model";

export type DecodeStep = {
	logits: number[];
	argmax: number;
	ms: number;
	/** WebGPU API calls made by the step itself (excludes the logits readback). */
	counts?: Record<string, number>;
};

function gpuCounts(): Record<string, number> | undefined {
	const c = (globalThis as { __gpuCounts?: Record<string, number> }).__gpuCounts;
	return c ? { ...c } : undefined;
}

function diffCounts(a?: Record<string, number>, b?: Record<string, number>) {
	if (!a || !b) return undefined;
	return Object.fromEntries(Object.keys(a).map((k) => [k, b[k] - a[k]]));
}

function sampleIndices(vocab: number): number[] {
	const out: number[] = [];
	for (let i = 0; i < 64; i++) out.push(Math.floor((i * (vocab - 1)) / 63));
	return out;
}

async function lastLogits(model: Gemma3, latent: np.Array) {
	const logits = runLogitsProjection(model, latent.slice([-1]).flatten());
	const all = (await logits.data()) as Float16Array | Float32Array;
	let argmax = 0;
	for (let i = 1; i < all.length; i++) if (all[i] > all[argmax]) argmax = i;
	return { logits: sampleIndices(all.length).map((i) => Number(all[i])), argmax };
}

function disposeState(state: Gemma3State) {
	for (const c of state.kvCaches) {
		c.k.dispose();
		c.v.dispose();
	}
	state.positionTensor.dispose();
}

export async function runDecodeCheck(context: number[], decode: number[]) {
	const devices = await init("webgpu");
	if (!devices.includes("webgpu")) throw new Error("WebGPU required");
	defaultDevice("webgpu");
	const adapter = await navigator.gpu.requestAdapter();
	const precision = adapter!.features.has("shader-f16" as GPUFeatureName) ? "f16" : "f32";

	const res = await fetch("/weights/model.safetensors");
	const file = safetensors.parse(await res.arrayBuffer());
	const model = precision === "f16" ? fromSafetensors(file) : loadModelF32(file);

	let state = createGemma3State(model);
	const pre = runGemma3Step(tree.ref(model), state, np.array(context, { dtype: np.uint32 }));
	state = pre.state;
	const steps: DecodeStep[] = [];
	const first = await lastLogits(model, pre.latent);
	steps.push({ ...first, ms: 0 });

	for (const tok of decode) {
		const c0 = gpuCounts();
		const t0 = performance.now();
		const out = runGemma3Step(tree.ref(model), state, np.array([tok], { dtype: np.uint32 }));
		state = out.state;
		await blockUntilReady(out.latent);
		const ms = performance.now() - t0;
		const counts = diffCounts(c0, gpuCounts());
		steps.push({ ...(await lastLogits(model, out.latent)), ms, counts });
	}
	disposeState(state);

	// Independent path: one prefill over the full sequence.
	const full = createGemma3State(model);
	const all = runGemma3Step(
		tree.ref(model),
		full,
		np.array([...context, ...decode], { dtype: np.uint32 }),
	);
	const prefillLast = await lastLogits(model, all.latent);
	disposeState(all.state);

	return { precision, context: context.length, decode, steps, prefillLast };
}
