/**
 * Where the prefill path spends its host time, and whether submits or kernels
 * are the cost.
 *
 * At S=128 the tiled-matmul path is *slower* than the fused layer (136 ms vs
 * 105 ms) even though its GPU work fell from 87.8 ms to 11.1 ms. That means the
 * wall time is host issue, not compute, and the cause has to be counted rather
 * than assumed: a jitted call submits one command buffer per executable, and an
 * un-jitted sequence of small ops submits one per op.
 *
 * This counts, per arm and per S: kernels, submits, and the two host API costs
 * that scale with submits.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { createGemma3State, runGemma3Step } from "../../src/lib/gemma3/model";
import type { Gemma3, Gemma3State } from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

const round = (x: number, n = 2) => +x.toFixed(n);

function dispose(state: Gemma3State) {
	for (const c of state.kvCaches) {
		c.k.dispose();
		c.v.dispose();
	}
	state.positionTensor.dispose();
}

type Arm = "fused" | "split" | "kernel";
function select(arm: Arm) {
	const g = globalThis as { __noTiledMatmul?: boolean; __noPrefillLayer?: boolean };
	g.__noPrefillLayer = arm === "fused";
	g.__noTiledMatmul = arm !== "kernel";
}

/** Count GPU API calls made during one step. */
function startApiCounter(queue: GPUQueue) {
	const c = { submit: 0, bindGroup: 0, encoder: 0, buffer: 0 };
	const proto = GPUDevice.prototype as unknown as Record<string, unknown>;
	const saved: Record<string, unknown> = {};

	const wrap = (obj: Record<string, unknown>, key: string, protoObj?: Record<string, unknown>) => {
		const orig = obj[key] as (...a: unknown[]) => unknown;
		if (typeof orig !== "function") return;
		saved[key] = orig;
		obj[key] = function (...a: unknown[]) {
			c[key as keyof typeof c] = (c[key as keyof typeof c] ?? 0) + 1;
			return orig.apply(this, a);
		};
		if (protoObj) saved[`p:${key}`] = true;
	};

	wrap(queue as unknown as Record<string, unknown>, "submit");
	wrap(proto, "createBindGroup");
	wrap(proto, "createCommandEncoder");
	wrap(proto, "createBuffer");

	return {
		stop() {
			for (const k of ["submit", "createBindGroup", "createCommandEncoder", "createBuffer"]) {
				if (saved[k]) (queue as unknown as Record<string, unknown>)[k] = saved[k];
			}
		},
		counts: c,
	};
}

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model: Gemma3 = loadModelF32(safetensors.parse(await res.arrayBuffer()));
	const dev = (await import("@jax-js/jax")).getWebGPUDevice();

	const workloads = await (await fetch("/bench/workloads.json")).json();
	const longest: number[] = workloads.prefill
		.find((w: { name: string }) => w.name === "prefill-1024")
		.tokens.slice(0, 1024);

	const once = async (tokens: number[]) => {
		const state = createGemma3State(model);
		const t0 = performance.now();
		const out = runGemma3Step(tree.ref(model), state, np.array(tokens, { dtype: np.uint32 }));
		await blockUntilReady(out.latent);
		await out.latent.ref.slice([-1]).data();
		const dt = performance.now() - t0;
		out.latent.dispose();
		dispose(out.state);
		return dt;
	};

	select("fused");
	for (let i = 0; i < 3; i++) await once(longest);

	const rows: Record<string, unknown>[] = [];
	for (const S of [128, 1024]) {
		const tokens = longest.slice(0, S);
		const row: Record<string, unknown> = { S };
		for (const arm of ["fused", "split", "kernel"] as Arm[]) {
			select(arm);
			for (let i = 0; i < 3; i++) await once(tokens);

			const counter = startApiCounter(dev.queue);
			const state = createGemma3State(model);
			const out = runGemma3Step(tree.ref(model), state, np.array(tokens, { dtype: np.uint32 }));
			await blockUntilReady(out.latent);
			const c = counter.counts;
			counter.stop();
			// jax-js caches command encoders per executable, so counting is only
			// meaningful for submits and bind groups on a warm path.
			out.latent.dispose();
			dispose(out.state);

			row[arm] = { ...c };
		}
		rows.push(row);
	}

	select("kernel");
	return {
		note: "counts are per step, on a warm pipeline cache",
		rows,
	};
}
