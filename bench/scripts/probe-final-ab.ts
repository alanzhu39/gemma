/**
 * The final A/B: cost of each candidate, in the model, interleaved, verified.
 *
 * Four arms, all running the real decode step at ctx 1024 (f32, GTX 1060) and all
 * beginning from their own identical prefill so every arm projects the same
 * position and can be compared element-wise:
 *
 *   A  as-shipped before the fix   runLinear(tokenEmbed)   one submit per kernel
 *   B  + submit batching           same projection, submits batched, destroys deferred
 *   C  + contiguous projection     runLinear on a [640,V] copy, one submit per kernel
 *   D  + both
 *
 * Why each is a candidate, from measurements in this tree:
 *  - The projection reads W=[262144,640] through a transpose *view*, so successive
 *    k are 2560 B apart: 4 B used of every 128 B line. The same bytes contiguous
 *    run at 165 GB/s vs 10 GB/s (probe-projection-real, one process, GPU
 *    timestamps, verified equal).
 *  - jax-js submits one command buffer per kernel: 469 per step. A synthetic
 *    control of 468 dependent kernels costs 45.4 ms that way and 6.0 ms in one
 *    command buffer (probe-decode-breakdown section G), and the real step's GPU
 *    span is 86 ms for 26 ms of kernels (probe-submit-batching).
 *
 * Arms are interleaved per step so the machine's drifting load hits all four
 * equally. Every arm's logits are compared to arm A; a faster arm that computes
 * something different is a bug and is reported as one.
 */
import { blockUntilReady, defaultDevice, getWebGPUDevice, init, jit, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import {
	createGemma3State,
	runGemma3Step,
	runLinear,
	type Gemma3,
	type Gemma3State,
} from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

const round = (x: number, n = 2) => +x.toFixed(n);
const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	return s.length ? s[s.length >> 1] : NaN;
};
const q = (xs: number[], p: number) => {
	const s = [...xs].sort((a, b) => a - b);
	return s.length ? s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))] : NaN;
};

function disposeState(s: Gemma3State) {
	for (const c of s.kvCaches) {
		c.k.dispose();
		c.v.dispose();
	}
	s.positionTensor.dispose();
}

/** Same jitted dot as runLinear, but for an already-[dim,vocab] weight. */
const runLinearContig = jit((w: np.Array, x: np.Array) => np.dot(x, w));

/**
 * Batch submits and defer destroys on one device's queue.
 *
 * `pipelineSubmit` in the WebGPU backend calls queue.submit immediately and the
 * buffer is destroyed as soon as its refcount hits zero, which is before a
 * deferred submit would run - so both have to move together or the kernel reads
 * a destroyed buffer and silently computes zero (established by
 * bench/scripts/submit-hazard.ts).
 */
function installSubmitBatching(device: GPUDevice) {
	const queue = device.queue;
	const bufProto = GPUBuffer.prototype as unknown as Record<string, unknown>;
	const origSubmit = queue.submit.bind(queue);
	const origMapAsync = GPUBuffer.prototype.mapAsync;
	const origDestroy = GPUBuffer.prototype.destroy;
	const origOnDone = queue.onSubmittedWorkDone.bind(queue);

	let pending: GPUCommandBuffer[] = [];
	let dying: GPUBuffer[] = [];
	let batches = 0;
	let maxBatch = 0;

	const flush = () => {
		if (pending.length) {
			const b = pending;
			pending = [];
			batches++;
			maxBatch = Math.max(maxBatch, b.length);
			origSubmit(b);
		}
		if (dying.length) {
			const d = dying;
			dying = [];
			for (const x of d) origDestroy.call(x);
		}
	};

	(queue as unknown as { submit: (b: GPUCommandBuffer[]) => void }).submit = (buffers) => {
		pending.push(...buffers);
	};
	bufProto.destroy = function (this: GPUBuffer) {
		dying.push(this);
	};
	bufProto.mapAsync = function (this: GPUBuffer, ...args: unknown[]) {
		flush();
		return (origMapAsync as (...a: unknown[]) => Promise<void>).apply(this, args);
	};
	(queue as unknown as { onSubmittedWorkDone: () => Promise<void> }).onSubmittedWorkDone = () => {
		flush();
		return origOnDone();
	};

	return {
		flush,
		stats: () => ({ batches, maxBatch }),
		uninstall: () => {
			flush();
			(queue as unknown as { submit: (b: GPUCommandBuffer[]) => void }).submit = origSubmit;
			bufProto.destroy = origDestroy;
			bufProto.mapAsync = origMapAsync;
			(queue as unknown as { onSubmittedWorkDone: () => Promise<void> }).onSubmittedWorkDone = origOnDone;
		},
	};
}

export async function probe() {
	const trials = (globalThis as { __trials?: number }).__trials ?? 3;
	const stepsPerTrial = (globalThis as { __steps?: number }).__steps ?? 2;

	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model = loadModelF32(safetensors.parse(await res.arrayBuffer())) as Gemma3 & {
		tokenEmbedT?: np.Array;
	};
	const Wt = model.tokenEmbedT ?? (model.tokenEmbed.ref.transpose().mul(1) as np.Array);

	const workloads = await (await fetch("/bench/workloads.json")).json();
	const wl = workloads.decode.find((w: { name: string }) => w.name === "decode-ctx1024");
	const context: number[] = wl.tokens.slice(0, 1024);

	const prefillInto = async () => {
		let s = createGemma3State(model);
		const pre = runGemma3Step(tree.ref(model), s, np.array(context, { dtype: np.uint32 }));
		await blockUntilReady(pre.latent);
		pre.latent.dispose();
		return pre.state;
	};

	const project = (model: Gemma3, st: Gemma3State, token: number, contiguous: boolean) => {
		const out = runGemma3Step(tree.ref(model), st, np.array([token], { dtype: np.uint32 }));
		const x = out.latent.ref.slice([-1]).flatten();
		const logits = contiguous
			? runLinearContig(Wt.ref, x)
			: runLinear({ weight: tree.ref(model).tokenEmbed }, x);
		out.latent.dispose();
		return { logits, state: out.state };
	};

	type Arm = "A" | "B" | "C" | "D";
	const arms: Arm[] = ["A", "B", "C", "D"];
	const times: Record<Arm, number[]> = { A: [], B: [], C: [], D: [] };
	const refLogits: Record<string, Float32Array | null> = { A: null, B: null, C: null, D: null };
	let scale = 0;
	let patchStats = { batches: 0, maxBatch: 0 };

	// Warm both projection pipelines before measuring. Each arm needs its own
	// freshly prefilled state: runGemma3Step consumes the state it is given.
	for (const contiguous of [false, true]) {
		let s = await prefillInto();
		for (let i = 0; i < 2; i++) {
			const r = project(model, s, 1000 + i, contiguous);
			await blockUntilReady(r.logits);
			r.logits.dispose();
			s = r.state;
		}
		disposeState(s);
	}

	for (let trial = 0; trial < trials; trial++) {
		for (const arm of arms) {
			const contiguous = arm === "C" || arm === "D";
			const batched = arm === "B" || arm === "D";
			let st = await prefillInto();
			const p = batched ? installSubmitBatching(getWebGPUDevice()) : null;
			try {
				for (let i = 0; i < stepsPerTrial; i++) {
					const t0 = performance.now();
					const r = project(model, st, 1000 + (i % 7), contiguous);
					await blockUntilReady(r.logits);
					times[arm].push(round(performance.now() - t0));
					const h = (await r.logits.ref.data()) as Float32Array;
					if (i === 0) {
						if (!refLogits[arm]) refLogits[arm] = h;
						for (let k = 0; k < h.length; k++) scale = Math.max(scale, Math.abs(h[k]));
					}
					r.logits.dispose();
					st = r.state;
				}
			} finally {
				if (p) {
					p.flush();
					const s = p.stats();
					patchStats = {
						batches: patchStats.batches + s.batches,
						maxBatch: Math.max(patchStats.maxBatch, s.maxBatch),
					};
					p.uninstall();
				}
				disposeState(st);
			}
		}
	}

	// Verify every arm against A: same position, so identical logits expected.
	const base = refLogits.A;
	const verify: Record<string, unknown> = {};
	for (const arm of arms) {
		const cur = refLogits[arm];
		if (!base || !cur) {
			verify[arm] = "missing";
			continue;
		}
		let worst = 0;
		let amaxA = 0;
		let amaxB = 0;
		for (let k = 0; k < base.length; k++) {
			if (base[k] > base[amaxA]) amaxA = k;
			if (cur[k] > cur[amaxB]) amaxB = k;
			worst = Math.max(worst, Math.abs(base[k] - cur[k]));
		}
		verify[arm] = {
			maxAbsDiff: +worst.toExponential(2),
			relative: +(worst / Math.max(1e-9, scale)).toExponential(2),
			argmaxMatch: amaxA === amaxB,
		};
	}

	const summary = Object.fromEntries(
		arms.map((a) => [
			a,
			{
				values: times[a],
				median: median(times[a]),
				min: Math.min(...times[a]),
				p90: q(times[a], 0.9),
			},
		]),
	) as Record<Arm, { values: number[]; median: number; min: number; p90: number }>;

	if (!model.tokenEmbedT) Wt.dispose();

	return {
		ctx: 1024,
		trials,
		stepsPerTrial,
		arms: {
			A: "as-shipped (strided projection, one submit per kernel)",
			B: "+ submit batching only",
			C: "+ contiguous projection only",
			D: "+ both",
		},
		summary,
		savedVsA: {
			B: round(summary.A.median - summary.B.median),
			C: round(summary.A.median - summary.C.median),
			D: round(summary.A.median - summary.D.median),
		},
		verify,
		patchStats,
	};
}
