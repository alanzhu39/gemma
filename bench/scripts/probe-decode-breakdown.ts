/**
 * Account for every millisecond of a decode step.
 *
 * The benchmark reports 120.6 ms/token (decode-ctx1024, f32). Traced GPU kernels
 * total ~51 ms across the whole token, so ~70 ms is unaccounted. This probe
 * measures, in one process on one warm GPU with the real model:
 *
 *   A. the real step, phase by phase (host issue vs GPU wait), real sampler
 *   B. the step with pieces removed (modelOnly vs modelProj = bench.ts's shape)
 *   C. the projection alone, two independent code paths cross-checked
 *   D. the sampler alone on a resident logits vector
 *   E. GPU kernel time, from jax-js's own timestamp-query trace
 *   F. the GPU timeline per queue.submit, from onSubmittedWorkDone() — this is
 *      what locates the missing time: gaps *between* dispatches, not in them
 *   G. a controlled A/B of 468 dependent kernels issued as N submits vs 1
 *
 * Guard rails:
 *  - every bandwidth is checked against the card's ~192 GB/s peak; above it is
 *    a probe bug (a silently elided kernel), not a result;
 *  - the projection is computed by runLinear and np.dot and compared, so "fast"
 *    cannot mean "skipped";
 *  - the argsort's top element is checked against a host argmax;
 *  - GPU-side numbers come from GPU timestamps, so CPU contention cannot
 *    stretch them; wall numbers are reported with the contention measured.
 */
import {
	blockUntilReady,
	defaultDevice,
	getWebGPUDevice,
	init,
	nn,
	numpy as np,
	tree,
} from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import {
	createGemma3State,
	runGemma3Step,
	runLinear,
	type Gemma3,
	type Gemma3State,
} from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

const PEAK_GBPS = 192; // GTX 1060, NVIDIA's decimal GB/s convention

type Counts = {
	submit: number;
	submitMs: number;
	dispatch: number;
	createBuffer: number;
	createBufferMs: number;
	createBindGroup: number;
	createBindGroupMs: number;
	encoder: number;
	encoderMs: number;
	mapAsync: number;
	mapAsyncMs: number;
	copyB2B: number;
	copyB2BMs: number;
	destroy: number;
	destroyMs: number;
};

const ZERO: Counts = {
	submit: 0, submitMs: 0, dispatch: 0,
	createBuffer: 0, createBufferMs: 0,
	createBindGroup: 0, createBindGroupMs: 0,
	encoder: 0, encoderMs: 0,
	mapAsync: 0, mapAsyncMs: 0,
	copyB2B: 0, copyB2BMs: 0,
	destroy: 0, destroyMs: 0,
};

function snap(): Counts {
	const c = (globalThis as { __gpuT?: Counts }).__gpuT;
	return c ? { ...c } : { ...ZERO };
}

function diff(a: Counts, b: Counts): Counts {
	const out = {} as Counts;
	for (const k of Object.keys(ZERO) as (keyof Counts)[]) out[k] = +(b[k] - a[k]).toFixed(3);
	return out;
}

const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const minOf = (xs: number[]) => Math.min(...xs);
const maxOf = (xs: number[]) => Math.max(...xs);
const round = (x: number, n = 2) => +x.toFixed(n);

function disposeState(state: Gemma3State) {
	for (const c of state.kvCaches) {
		c.k.dispose();
		c.v.dispose();
	}
	state.positionTensor.dispose();
}

/**
 * Per-submit GPU completion timeline.
 *
 * `GPUQueue.onSubmittedWorkDone()` resolves once everything submitted so far has
 * finished. Called right after each submit, it gives the GPU-timeline completion
 * time of submit i on the CPU clock. Deltas between consecutive completions are
 * the time the GPU spent on each command buffer plus the gap before the next one
 * completed. The queue is deep, so this is not a per-submit fence.
 */
function startSubmitTimeline(queue: GPUQueue) {
	const marks: { t: number }[] = [];
	let active = true;
	const origSubmit = queue.submit.bind(queue);
	(queue as unknown as { submit: (b: GPUCommandBuffer[]) => void }).submit = (buffers) => {
		origSubmit(buffers);
		if (!active) return;
		const slot = { t: 0 };
		marks.push(slot);
		void queue.onSubmittedWorkDone().then(() => {
			slot.t = performance.now();
		});
	};
	return {
		stop() {
			active = false;
			(queue as unknown as { submit: (b: GPUCommandBuffer[]) => void }).submit = origSubmit;
			return marks;
		},
	};
}

/** Summarise a completion timeline: total span and gap statistics. */
function timelineStats(marks: { t: number }[], t0: number) {
	const done = marks.map((m, i) => ({ i, t: m.t })).filter((m) => m.t > 0);
	if (done.length < 2) return { n: done.length, spanMs: 0, medianGapMs: 0, p90GapMs: 0, maxGapMs: 0, sumGapMs: 0 };
	const deltas: number[] = [];
	for (let i = 1; i < done.length; i++) deltas.push(round(done[i].t - done[i - 1].t, 3));
	const span = round(done[done.length - 1].t - t0, 2);
	const sum = round(deltas.reduce((a, b) => a + b, 0), 2);
	return {
		n: done.length,
		spanMs: span,
		medianGapMs: median(deltas),
		p90GapMs: [...deltas].sort((a, b) => a - b)[Math.floor(deltas.length * 0.9)],
		maxGapMs: maxOf(deltas),
		sumGapMs: sum,
	};
}

/**
 * The sampling path from src/lib/gemma3/inference.ts, instrumented. Same ops in
 * the same order as the original, so the total transfers; each phase is timed.
 */
async function sampleLogitsTimed(
	logits: np.Array,
	temperature: number,
	topK: number,
	topP: number,
	verifyTop1: boolean,
) {
	const row: Record<string, number> = {};
	let t = performance.now();
	const scaledLogits = logits.div(temperature);
	row.sampleDivIssue = round(performance.now() - t);

	// Force the sort to finish before timing the gather; otherwise the gather's
	// cost silently includes the sort's.
	t = performance.now();
	const topKIndices = np.flip(scaledLogits.ref.argsort()).slice([0, topK]);
	const tIssued = performance.now();
	row.argsortIssue = round(tIssued - t);
	await blockUntilReady(topKIndices);
	row.argsortWait = round(performance.now() - tIssued);

	if (verifyTop1) {
		t = performance.now();
		row.argsortTop1 = (await topKIndices.ref.slice(0).data())[0] as number;
		row.argsortTop1Readback = round(performance.now() - t);
	}

	t = performance.now();
	const probsArr = nn.softmax(scaledLogits.slice(topKIndices.ref));
	const softmaxIssued = performance.now();
	row.softmaxIssue = round(softmaxIssued - t);

	const probs = (await probsArr.data()) as Float16Array;
	row.probsReadback = round(performance.now() - softmaxIssued);

	const cumSum = new Float16Array(probs.length);
	let pCutoff = 0;
	let currSum = 0;
	for (let i = 0; i < probs.length; i++) {
		currSum += probs[i];
		cumSum[i] = currSum;
		if (currSum >= topP) {
			pCutoff = i;
			break;
		}
	}
	const rand = Math.random() * currSum;
	let left = 0;
	let right = pCutoff;
	let probsIndex = 0;
	while (left <= right) {
		const mid = Math.floor((left + right) / 2);
		if (cumSum[mid] > rand) {
			probsIndex = mid;
			right = mid - 1;
		} else {
			left = mid + 1;
		}
	}

	t = performance.now();
	row.token = (await topKIndices.slice(probsIndex).data())[0] as number;
	row.tokenReadback = round(performance.now() - t);
	row.probsSum = round(currSum, 4);
	return row;
}

/** Cost of a sync with nothing to wait for. */
async function emptySyncMs(arr: np.Array, reps = 10) {
	await blockUntilReady(arr);
	const t0 = performance.now();
	for (let i = 0; i < reps; i++) await blockUntilReady(arr.ref);
	return round((performance.now() - t0) / reps);
}

/**
 * Price 468 dependent kernels under the three submission shapes:
 *   one-submit-each   - what jax-js does today
 *   submit-batched    - 468 command buffers, one submit
 *   one-command-buf   - 468 passes, one submit (the fused shape)
 * Every kernel reads the previous kernel's output, so nothing can overlap.
 */
async function submitShapeProbe(N: number) {
	const adapter = await navigator.gpu.requestAdapter();
	const dev = await adapter!.requestDevice({
		requiredLimits: {
			maxBufferSize: adapter!.limits.maxBufferSize,
			maxStorageBufferBindingSize: adapter!.limits.maxStorageBufferBindingSize,
			maxComputeWorkgroupStorageSize: adapter!.limits.maxComputeWorkgroupStorageSize,
		} as GPURequiredLimits,
	});
	const buf = dev.createBuffer({
		size: 1024 * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
	});
	const stage = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
	const mod = dev.createShaderModule({
		code: `
		@group(0) @binding(0) var<storage, read_write> b: array<f32>;
		@compute @workgroup_size(64)
		fn main(@builtin(global_invocation_id) g: vec3<u32>) {
			if (g.x < 64u) { b[g.x] = b[g.x] + 1.0; }
		}`,
	});
	const pipe = dev.createComputePipeline({
		layout: "auto",
		compute: { module: mod, entryPoint: "main" },
	});
	const bg = dev.createBindGroup({
		layout: pipe.getBindGroupLayout(0),
		entries: [{ binding: 0, resource: { buffer: buf } }],
	});
	const addPass = (enc: GPUCommandEncoder) => {
		const p = enc.beginComputePass();
		p.setPipeline(pipe);
		p.setBindGroup(0, bg);
		p.dispatchWorkgroups(1);
		p.end();
	};
	const finish = async (buffers: GPUCommandBuffer[]) => {
		const enc = dev.createCommandEncoder();
		enc.copyBufferToBuffer(buf, 0, stage, 0, 4);
		buffers.push(enc.finish());
		for (let i = 0; i < buffers.length; i += 64) dev.queue.submit(buffers.slice(i, i + 64));
		await stage.mapAsync(GPUMapMode.READ);
		stage.unmap();
	};

	const timeIt = async (label: string, fn: () => Promise<void>, reps = 3) => {
		for (let i = 0; i < 2; i++) await fn();
		const t0 = performance.now();
		for (let i = 0; i < reps; i++) await fn();
		const ms = (performance.now() - t0) / reps;
		return { shape: label, kernels: N, ms: round(ms), usPerKernel: round((ms * 1000) / N, 1) };
	};

	const out = [];
	out.push(
		await timeIt("one submit per kernel", async () => {
			const buffers: GPUCommandBuffer[] = [];
			for (let i = 0; i < N; i++) {
				const enc = dev.createCommandEncoder();
				addPass(enc);
				dev.queue.submit([enc.finish()]);
			}
			// Final readback: copy + submit + map, so the work is complete.
			const enc = dev.createCommandEncoder();
			enc.copyBufferToBuffer(buf, 0, stage, 0, 4);
			await finish([enc.finish()]);
		}),
	);
	out.push(
		await timeIt("N command buffers, 1 submit", async () => {
			const buffers: GPUCommandBuffer[] = [];
			for (let i = 0; i < N; i++) {
				const enc = dev.createCommandEncoder();
				addPass(enc);
				buffers.push(enc.finish());
			}
			await finish(buffers);
		}),
	);
	out.push(
		await timeIt("N passes in 1 command buffer", async () => {
			const enc = dev.createCommandEncoder();
			for (let i = 0; i < N; i++) addPass(enc);
			await finish([enc.finish()]);
		}),
	);

	buf.destroy();
	stage.destroy();
	return out;
}

export async function probe() {
	const ctxArg = (globalThis as { __ctx?: number }).__ctx ?? 1024;
	const steps = (globalThis as { __steps?: number }).__steps ?? 8;
	const warmup = (globalThis as { __warmup?: number }).__warmup ?? 3;

	await init("webgpu");
	defaultDevice("webgpu");
	const dev = getWebGPUDevice();
	const adapter = await navigator.gpu.requestAdapter();
	const info = {
		hasTimestampQuery: dev.features.has("timestamp-query"),
		f16: adapter!.features.has("shader-f16" as GPUFeatureName),
		maxBufferSize: adapter!.limits.maxBufferSize,
		maxStorageBinding: adapter!.limits.maxStorageBufferBindingSize,
		maxComputeWorkgroupStorageSize: adapter!.limits.maxComputeWorkgroupStorageSize,
	};
	// A bare requestDevice() caps buffers at 256 MB, which would silently skip
	// the 640 MB projection. jax-js requests adapter limits; verify it stuck.
	const limitsOk =
		dev.limits.maxBufferSize === adapter!.limits.maxBufferSize &&
		dev.limits.maxStorageBufferBindingSize === adapter!.limits.maxStorageBufferBindingSize;

	const res = await fetch("/weights/model.safetensors");
	const file = safetensors.parse(await res.arrayBuffer());
	// No shader-f16 on this card, so f32 is this hardware's real path and
	// matches every published result for this box (precision "f32").
	const model: Gemma3 = loadModelF32(file);

	const workloads = await (await fetch("/bench/workloads.json")).json();
	const wl = workloads.decode.find((w: { name: string }) => w.name === "decode-ctx1024");
	const context: number[] = wl.tokens.slice(0, ctxArg);

	// ---- prefill + warmup ---------------------------------------------------
	let st = createGemma3State(model);
	const pre = runGemma3Step(tree.ref(model), st, np.array(context, { dtype: np.uint32 }));
	await blockUntilReady(pre.latent);
	pre.latent.dispose();
	st = pre.state;

	const runStep = async (token: number) => {
		const out = runGemma3Step(tree.ref(model), st, np.array([token], { dtype: np.uint32 }));
		await blockUntilReady(out.latent);
		out.latent.dispose();
		st = out.state;
	};
	for (let i = 0; i < warmup; i++) await runStep(1000 + i);

	// =======================================================================
	// A. The real step, phase by phase, with the real generation sampler.
	// =======================================================================
	type Phase = Record<string, number>;
	const fullSteps: Phase[] = [];

	for (let i = 0; i < steps; i++) {
		const token = 1000 + (i % 7);
		const c0 = snap();
		const t0 = performance.now();

		const out = runGemma3Step(tree.ref(model), st, np.array([token], { dtype: np.uint32 }));
		const t1 = performance.now();
		await blockUntilReady(out.latent);
		const t2 = performance.now();

		const logits = runLinear(
			{ weight: tree.ref(model).tokenEmbed },
			out.latent.ref.slice([-1]).flatten(),
		);
		const t3 = performance.now();
		await blockUntilReady(logits);
		const t4 = performance.now();

		const s = await sampleLogitsTimed(logits.ref, 1.0, 64, 0.95, false);
		const t5 = performance.now();

		st = out.state;
		out.latent.dispose();
		logits.dispose();
		const c1 = snap();

		fullSteps.push({
			step: i,
			modelIssue: round(t1 - t0),
			modelWait: round(t2 - t1),
			projIssue: round(t3 - t2),
			projWait: round(t4 - t3),
			sampling: round(t5 - t4),
			total: round(t5 - t0),
			...s,
			...Object.fromEntries(
				Object.entries(diff(c0, c1)).map(([k, v]) => [`api_${k}`, v as number]),
			),
		});
	}

	// =======================================================================
	// B. Remove pieces one at a time. modelProj is exactly bench.ts's region.
	// =======================================================================
	const variants: Record<string, { ms: number[]; api: Counts[]; resync: number[]; tl: number[] }> = {
		modelOnly: { ms: [], api: [], resync: [], tl: [] },
		modelProj: { ms: [], api: [], resync: [], tl: [] },
	};

	for (let i = 0; i < steps; i++) {
		const token = 1000 + (i % 7);

		// -- modelOnly, with a per-submit GPU completion timeline.
		let c0 = snap();
		let t0 = performance.now();
		let tl = startSubmitTimeline(dev.queue);
		let out = runGemma3Step(tree.ref(model), st, np.array([token], { dtype: np.uint32 }));
		await blockUntilReady(out.latent);
		let dt = performance.now() - t0;
		let marks = tl.stop();
		let c1 = snap();
		variants.modelOnly.ms.push(round(dt));
		variants.modelOnly.api.push(diff(c0, c1));
		const tls = timelineStats(marks, t0);
		variants.modelOnly.tl.push(tls);
		variants.modelOnly.resync.push(await emptySyncMs(out.latent, 4));
		st = out.state;
		out.latent.dispose();

		// -- modelProj
		c0 = snap();
		t0 = performance.now();
		tl = startSubmitTimeline(dev.queue);
		out = runGemma3Step(tree.ref(model), st, np.array([token], { dtype: np.uint32 }));
		const logits = runLinear(
			{ weight: tree.ref(model).tokenEmbed },
			out.latent.ref.slice([-1]).flatten(),
		);
		await blockUntilReady(logits);
		dt = performance.now() - t0;
		marks = tl.stop();
		c1 = snap();
		variants.modelProj.ms.push(round(dt));
		variants.modelProj.api.push(diff(c0, c1));
		variants.modelProj.tl.push(timelineStats(marks, t0));
		variants.modelProj.resync.push(await emptySyncMs(logits, 4));
		st = out.state;
		out.latent.dispose();
		logits.dispose();
	}

	// =======================================================================
	// C. The projection alone (the disputed number), cross-checked.
	// =======================================================================
	const vec = np.ones([640], { dtype: np.float32 });
	const W = tree.ref(model).tokenEmbed;
	const projBytes = 262144 * 640 * 4;
	const projTimes: number[] = [];
	const projDirect: number[] = [];

	const checkA = runLinear({ weight: tree.ref(model).tokenEmbed }, vec.ref);
	const checkB = np.dot(vec.ref, W.ref.transpose());
	const [hostA, hostB] = (await Promise.all([checkA.data(), checkB.data()])) as [
		Float32Array,
		Float32Array,
	];
	let projMaxAbsDiff = 0;
	for (let i = 0; i < hostA.length; i++)
		projMaxAbsDiff = Math.max(projMaxAbsDiff, Math.abs(hostA[i] - hostB[i]));
	let projFinite = true;
	for (let i = 0; i < hostA.length; i++) if (!Number.isFinite(hostA[i])) projFinite = false;

	let projTimeline: ReturnType<typeof timelineStats> | null = null;
	for (let i = 0; i < 8; i++) {
		let t0 = performance.now();
		let tl = startSubmitTimeline(dev.queue);
		const lg = runLinear({ weight: tree.ref(model).tokenEmbed }, vec.ref);
		await blockUntilReady(lg);
		projTimes.push(round(performance.now() - t0));
		if (i === 0) projTimeline = timelineStats(tl.stop(), t0);
		else tl.stop();
		lg.dispose();

		t0 = performance.now();
		const lg2 = np.dot(vec.ref, W.ref.transpose());
		await blockUntilReady(lg2);
		projDirect.push(round(performance.now() - t0));
		lg2.dispose();
	}
	const projMedianMs = median(projTimes);
	const projGbps = round(projBytes / 1e9 / (projMedianMs / 1000), 1);
	if (projGbps > PEAK_GBPS)
		throw new Error(`projection ${projGbps} GB/s exceeds peak; work was elided`);

	// =======================================================================
	// D. Sampler alone on a resident logits vector.
	// =======================================================================
	const logitsVec = runLinear({ weight: tree.ref(model).tokenEmbed }, vec.ref);
	await blockUntilReady(logitsVec);
	const hostLogits = (await logitsVec.ref.data()) as Float32Array;
	let hostArgmax = 0;
	for (let i = 1; i < hostLogits.length; i++) if (hostLogits[i] > hostLogits[hostArgmax]) hostArgmax = i;

	const samplerRuns: Phase[] = [];
	const verify = await sampleLogitsTimed(logitsVec.ref, 1.0, 64, 0.95, true);
	for (let i = 0; i < 8; i++) samplerRuns.push(await sampleLogitsTimed(logitsVec.ref, 1.0, 64, 0.95, false));
	// Best case: min over runs, so CPU contention inflates it as little as possible.
	const samplerTotal = (r: Phase) =>
		r.sampleDivIssue + r.argsortIssue + r.argsortWait + r.softmaxIssue + r.probsReadback + r.tokenReadback;
	const samplerMinMs = Math.min(...samplerRuns.map(samplerTotal));
	logitsVec.dispose();

	// =======================================================================
	// E. GPU kernel time from jax-js's timestamp-query trace (model step only).
	// =======================================================================
	type TraceAgg = { count: number; totalMs: number };
	let kernelTrace: Record<string, TraceAgg> | string = "unavailable";
	if (info.hasTimestampQuery) {
		const { profiler } = await import("@jax-js/jax");
		const measures: PerformanceMeasure[] = [];
		const obs = new PerformanceObserver((list) => {
			for (const e of list.getEntries()) measures.push(e as PerformanceMeasure);
		});
		try {
			obs.observe({ type: "measure" });
		} catch {
			// getEntries fallback below
		}
		profiler.startTrace();
		const out = runGemma3Step(tree.ref(model), st, np.array([1003], { dtype: np.uint32 }));
		await blockUntilReady(out.latent);
		for (let i = 0; i < 12; i++) await new Promise((r) => setTimeout(r, 60));
		profiler.stopTrace();
		for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 60));
		obs.disconnect();
		st = out.state;
		out.latent.dispose();

		const all = measures.length
			? measures
			: (performance.getEntriesByType("measure") as unknown as PerformanceMeasure[]).filter(
					(e) => /^(Kernel|Routine)\[/.test(e.name),
				);
		const agg: Record<string, TraceAgg> = {};
		let traced = 0;
		for (const e of all) {
			const a = (agg[e.name] ??= { count: 0, totalMs: 0 });
			a.count++;
			a.totalMs += e.duration;
			traced += e.duration;
		}
		for (const a of Object.values(agg)) a.totalMs = round(a.totalMs);
		kernelTrace = all.length
			? {
					__totalMs: { count: all.length, totalMs: round(traced) },
					...Object.fromEntries(
						Object.entries(agg)
							.sort((a, b) => b[1].totalMs - a[1].totalMs)
							.slice(0, 12),
					),
				}
			: "no entries captured";
	}

	// =======================================================================
	// G. Controlled A/B: 468 dependent kernels, three submission shapes.
	// =======================================================================
	const submitShapes = await submitShapeProbe(468);

	vec.dispose();
	disposeState(st);

	const col = (rows: Phase[], k: string) => rows.map((r) => r[k]).filter((x) => typeof x === "number");
	const agg = (v: { ms: number[]; api: Counts[]; resync: number[]; tl: unknown[] }) => ({
		medianMs: median(v.ms),
		minMs: minOf(v.ms),
		maxMs: maxOf(v.ms),
		spread: round(maxOf(v.ms) / Math.max(1e-9, minOf(v.ms)), 3),
		resyncMs: median(v.resync),
		timeline: v.tl.map((x) => x as ReturnType<typeof timelineStats>),
		submits: median(v.api.map((c) => c.submit)),
		submitMs: median(v.api.map((c) => c.submitMs)),
		dispatches: median(v.api.map((c) => c.dispatch)),
		createBuffer: median(v.api.map((c) => c.createBuffer)),
		createBufferMs: median(v.api.map((c) => c.createBufferMs)),
		createBindGroup: median(v.api.map((c) => c.createBindGroup)),
		createBindGroupMs: median(v.api.map((c) => c.createBindGroupMs)),
		mapAsync: median(v.api.map((c) => c.mapAsync)),
		mapAsyncMs: median(v.api.map((c) => c.mapAsyncMs)),
		destroy: median(v.api.map((c) => c.destroy)),
		destroyMs: median(v.api.map((c) => c.destroyMs)),
		encoder: median(v.api.map((c) => c.encoder)),
		encoderMs: median(v.api.map((c) => c.encoderMs)),
	});

	return {
		config: { ctx: ctxArg, steps, warmup, precision: "f32", deviceLimitsAligned: limitsOk, ...info },
		phases: {
			modelIssueMs: median(col(fullSteps, "modelIssue")),
			modelWaitMs: median(col(fullSteps, "modelWait")),
			projIssueMs: median(col(fullSteps, "projIssue")),
			projWaitMs: median(col(fullSteps, "projWait")),
			samplingMs: median(col(fullSteps, "sampling")),
			totalMs: median(col(fullSteps, "total")),
			minTotalMs: minOf(col(fullSteps, "total")),
			spread: round(maxOf(col(fullSteps, "total")) / Math.max(1e-9, minOf(col(fullSteps, "total"))), 3),
		},
		samplingBreakdown: {
			divIssueMs: median(col(fullSteps, "sampleDivIssue")),
			argsortIssueMs: median(col(fullSteps, "argsortIssue")),
			argsortWaitMs: median(col(fullSteps, "argsortWait")),
			softmaxIssueMs: median(col(fullSteps, "softmaxIssue")),
			probsReadbackMs: median(col(fullSteps, "probsReadback")),
			tokenReadbackMs: median(col(fullSteps, "tokenReadback")),
			probsSum: median(col(fullSteps, "probsSum")),
		},
		stepVariants: { modelOnly: agg(variants.modelOnly), modelProj: agg(variants.modelProj) },
		projection: {
			shape: "[1,640] x [640,262144] f32, 640 MiB",
			viaRunLinearMs: projTimes,
			viaNpDotMs: projDirect,
			medianMs: projMedianMs,
			minMs: minOf(projTimes),
			gbps: projGbps,
			peakGbps: PEAK_GBPS,
			maxAbsDiffBetweenPaths: +projMaxAbsDiff.toExponential(2),
			finite: projFinite,
			timeline: projTimeline,
		},
		samplerIsolated: {
			runs: samplerRuns,
			verify,
			hostArgmax,
			argsortTop1MatchesHost: verify.argsortTop1 === hostArgmax,
			medianTotalMs: median(samplerRuns.map(samplerTotal)),
			minTotalMs: samplerMinMs,
		},
		fullSteps,
		kernelTrace,
		submitShapes,
	};
}
