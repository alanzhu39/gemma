/**
 * Long-run decode stability, in-page half. See bench/stability.mjs for the
 * assertions and the rationale.
 *
 * Teacher-forces a few hundred decode steps and records the WebGPU traffic of
 * each one, plus a final check that the decode path still agrees with a single
 * prefill of the same tokens.
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

type Counts = Record<string, number>;

export type StepRecord = {
	dispatch: number;
	submit: number;
	createBuffer: number;
	createBufferBytes: number;
	writeBuffer: number;
	bindGroup: number;
	pipe: number;
	pipeAsync: number;
	shader: number;
	mapAsync: number;
	/** Latent norm at this position, so a divergence shows up as a trend. */
	norm: number;
	/** Global KV capacity in rows during this step, to identify repad steps. */
	globalCapacity: number;
	/** The token id fed to this step. */
	token: number;
};

function counts(): Counts {
	return { ...(globalThis as unknown as { __gpuCounts: Counts }).__gpuCounts };
}

function diff(a: Counts, b: Counts): Counts {
	return Object.fromEntries(Object.keys(a).map((k) => [k, b[k] - a[k]]));
}

function disposeState(state: Gemma3State) {
	for (const c of state.kvCaches) {
		c.k.dispose();
		c.v.dispose();
	}
	state.positionTensor.dispose();
}

async function lastLogits(model: Gemma3, latent: np.Array) {
	const logits = runLogitsProjection(model, latent.ref.slice([-1]).flatten());
	const all = (await logits.data()) as Float16Array | Float32Array;
	let argmax = 0;
	let second = -Infinity;
	for (let i = 0; i < all.length; i++) {
		if (all[i] > all[argmax]) {
			second = all[argmax];
			argmax = i;
		} else if (i !== argmax && all[i] > second) {
			second = all[i];
		}
	}
	return { argmax, top1: Number(all[argmax]), margin: Number(all[argmax] - second) };
}

export async function runStability(context: number[], decode: number[]) {
	const devices = await init("webgpu");
	if (!devices.includes("webgpu")) throw new Error("WebGPU required");
	defaultDevice("webgpu");
	const adapter = await navigator.gpu.requestAdapter();
	const precision = adapter!.features.has("shader-f16" as GPUFeatureName) ? "f16" : "f32";
	const file = safetensors.parse(await (await fetch("/weights/model.safetensors")).arrayBuffer());
	const model = precision === "f16" ? fromSafetensors(file) : loadModelF32(file);

	let state = createGemma3State(model);
	const t0 = performance.now();
	const pre = runGemma3Step(tree.ref(model), state, np.array(context, { dtype: np.uint32 }));
	await blockUntilReady(pre.latent);
	const prefillMs = performance.now() - t0;
	state = pre.state;
	// The prefill latent has served its purpose; the decode run supplies its own.
	pre.latent.dispose();

	const steps: StepRecord[] = [];
	let finite = true;
	let argmaxCount = 1;
	const t1 = performance.now();
	for (const tok of decode) {
		const before = counts();
		const out = runGemma3Step(tree.ref(model), state, np.array([tok], { dtype: np.uint32 }));
		state = out.state;
		await blockUntilReady(out.latent);
		const d = diff(before, counts());
		const l = await lastLogits(model, out.latent);
		if (!Number.isFinite(l.top1) || !Number.isFinite(l.margin)) finite = false;
		if (Number.isFinite(l.argmax)) argmaxCount++;
		out.latent.dispose();
		steps.push({
			globalCapacity: state.kvCaches[5].k.shape[0],
			token: tok,
			dispatch: d.dispatch,
			submit: d.submit,
			createBuffer: d.createBuffer,
			createBufferBytes: d.createBufferBytes,
			writeBuffer: d.writeBuffer,
			bindGroup: d.bindGroup,
			pipe: d.pipe,
			pipeAsync: d.pipeAsync,
			shader: d.shader,
			mapAsync: d.mapAsync,
			norm: l.top1,
		});
	}
	const decodeMs = performance.now() - t1;
	const caps = {
		sliding: state.kvCaches[0].k.shape[0],
		global: state.kvCaches[5].k.shape[0],
	};
	const position = state.position;
	disposeState(state);

	// Independent path: one prefill over the whole sequence, compared at the
	// last position, so decode is checked against a different code path after
	// hundreds of steps rather than against itself.
	const full = createGemma3State(model);
	const all = runGemma3Step(
		tree.ref(model),
		full,
		np.array([...context, ...decode], { dtype: np.uint32 }),
	);
	const ref = await lastLogits(model, all.latent);
	all.latent.dispose();
	const lastStep = steps[steps.length - 1];
	// The same comparison bench/decode-check uses: relative distance between the
	// two independent paths' top-1 logits.
	const selfErr = Math.abs(lastStep.norm - ref.top1) / Math.max(1e-3, Math.abs(ref.top1));
	disposeState(all.state);

	return {
		precision,
		steps,
		caps,
		position,
		prefillMs,
		decodeMs,
		finite,
		argmaxCount,
		selfErr,
		refArgmax: ref.argmax,
	};
}
