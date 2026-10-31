/**
 * Once the GPU work collapses, what is left is host issue.
 *
 * After the projections, QK and PV moved to the tiled kernel, a prefill-1024
 * pass spends 74.8 ms of GPU time but 405 ms of wall time. That gap is the
 * actual optimization target now, so this splits it:
 *
 *   stepIssue   time for `runGemma3Step` to return (lowering + enqueue only)
 *   wait        time for the GPU to finish after that
 *   readback    the one-element read the harness does to force completion
 *
 * and counts the submits the step makes, since a submit is the unit of host cost
 * that the un-fused layer multiplied.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { createGemma3State, runGemma3Step } from "../../src/lib/gemma3/model";
import type { Gemma3, Gemma3State } from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const round = (x: number, n = 2) => +x.toFixed(n);

function dispose(s: Gemma3State) {
	for (const c of s.kvCaches) {
		c.k.dispose();
		c.v.dispose();
	}
	s.positionTensor.dispose();
}

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model: Gemma3 = loadModelF32(safetensors.parse(await res.arrayBuffer()));
	const dev = (await import("@jax-js/jax")).getWebGPUDevice();

	const workloads = await (await fetch("/bench/workloads.json")).json();
	const tokens: number[] = workloads.prefill
		.find((w: { name: string }) => w.name === "prefill-1024")
		.tokens.slice(0, 1024);

	// Count submits per step, and how much host time each costs.
	let submits = 0;
	const origSubmit = dev.queue.submit.bind(dev.queue);
	(dev.queue as unknown as { submit: (b: GPUCommandBuffer[]) => void }).submit = (b) => {
		submits++;
		origSubmit(b);
	};

	const run = async () => {
		const state = createGemma3State(model);
		submits = 0;
		const t0 = performance.now();
		const out = runGemma3Step(tree.ref(model), state, np.array(tokens, { dtype: np.uint32 }));
		const t1 = performance.now(); // issue complete
		await blockUntilReady(out.latent);
		const t2 = performance.now(); // GPU done
		await out.latent.ref.slice([-1]).data();
		const t3 = performance.now(); // readback done
		const n = submits;
		out.latent.dispose();
		dispose(out.state);
		return {
			issueMs: round(t1 - t0),
			waitMs: round(t2 - t1),
			readbackMs: round(t3 - t2),
			totalMs: round(t3 - t0),
			submits: n,
		};
	};

	for (let i = 0; i < 3; i++) await run();
	const rows = [];
	for (let i = 0; i < 9; i++) rows.push(await run());

	(dev.queue as unknown as { submit: (b: GPUCommandBuffer[]) => void }).submit = origSubmit;

	const col = (k: "issueMs" | "waitMs" | "readbackMs" | "totalMs") =>
		median(rows.map((r) => r[k]));
	return {
		ctxLen: tokens.length,
		median: {
			issueMs: round(col("issueMs")),
			waitMs: round(col("waitMs")),
			readbackMs: round(col("readbackMs")),
			totalMs: round(col("totalMs")),
			submits: median(rows.map((r) => r.submits)),
			tps: round((tokens.length * 1000) / col("totalMs"), 1),
		},
		runs: rows,
	};
}
