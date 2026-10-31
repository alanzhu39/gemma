/**
 * What do jax-js's per-dispatch buffer allocations cost, and could a pool
 * explain the projection's ~9 GB/s?
 *
 * jax-js has no buffer pool: every `dispatch` is preceded by a `malloc` (one
 * `device.createBuffer`) and followed by `decRef` -> `buffer.destroy()`
 * (webgpu-*.js:906-913). A decode step therefore creates and destroys several
 * hundred buffers, which is host work the engine performs in the middle of the
 * step. If a `createBuffer`/`destroy` pair is expensive, that host work is a
 * candidate explanation for why one 640 MiB matvec only reaches ~9 GB/s on a
 * card that streams at ~100 GB/s.
 *
 * Two independent measurements, because they answer different questions:
 *
 *  Phase A (in-model, real weights, real decode step)
 *    - time inside `GPUDevice.createBuffer` and `GPUBuffer.destroy` directly,
 *      with `performance.now()`, per decode step and per phase (model vs
 *      projection), plus the size histogram that phase B replays.
 *    - the same step's host-issue / GPU-wait split, so the allocation number
 *      can be stated as a fraction of a step rather than on its own.
 *    - the isolated jax-js projection (same matvec, nothing queued ahead) as a
 *      same-session baseline for the 73 ms figure.
 *    - an *independent* numeric check: five logits recomputed on the CPU from
 *      the read-back latent and read-back weight rows. A wrapped path that
 *      silently produced zeros would pass a self-comparison; it cannot pass
 *      this.
 *
 *  Phase B (raw WebGPU, synthetic 640 MiB weight, same access pattern)
 *    - the projection's own generated WGSL is captured from `createShaderModule`
 *      during the projection phase, so "same access pattern" is checkable
 *      rather than asserted.
 *    - fresh allocate+destroy per call (as shipped, including destroy right
 *      after submit, before completion) vs pooled (allocation hoisted out).
 *    - pooled *with* the step's real allocation churn replayed in front of it,
 *      which is the only variant that can show churn hurting the big read.
 *    - two reference points: the same read with vec4 loads, and a pure
 *      streaming reduce, to separate "access pattern" from everything else.
 *    - every variant is verified against a CPU reference before its time is
 *      reported, and anything above the card's ~192 GB/s peak is reported as a
 *      probe bug rather than as a result.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { createGemma3State, runGemma3Step, runLinear, type Gemma3 } from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

// ---------------------------------------------------------------------------
// Instrumentation
// ---------------------------------------------------------------------------

type Phase = "model" | "projection" | "readback";

type PhaseStats = {
	createMs: number;
	createN: number;
	createBytes: number;
	createMaxMs: number;
	createTimesMs: number[];
	destroyMs: number;
	destroyN: number;
	destroyMaxMs: number;
	bindGroupMs: number;
	bindGroupN: number;
};

const emptyPhase = (): PhaseStats => ({
	createMs: 0,
	createN: 0,
	createBytes: 0,
	createMaxMs: 0,
	createTimesMs: [],
	destroyMs: 0,
	destroyN: 0,
	destroyMaxMs: 0,
	bindGroupMs: 0,
	bindGroupN: 0,
});

/**
 * Wrap the allocation calls on the prototypes.
 *
 * Wrapping only records wall time and forwards the result unchanged, so the
 * only way it can perturb the model is the cost of `performance.now()` itself.
 * That is checked, not assumed: see the independent CPU dot-product check in
 * `phaseA`, and the `bench:check` gate run in the report.
 */
function instrument() {
	const phases = new Map<Phase, PhaseStats>();
	let current: Phase | null = null;
	const sizes: number[] = [];
	const shaders: { code: string }[] = [];

	const deviceProto = GPUDevice.prototype as unknown as Record<string, unknown>;
	const bufferProto = GPUBuffer.prototype as unknown as Record<string, unknown>;

	const origCreateBuffer = deviceProto.createBuffer as (this: GPUDevice, d: GPUBufferDescriptor) => GPUBuffer;
	const origCreateBindGroup = deviceProto.createBindGroup as (
		this: GPUDevice,
		d: GPUBindGroupDescriptor,
	) => GPUBindGroup;
	const origCreateShaderModule = deviceProto.createShaderModule as (
		this: GPUDevice,
		d: GPUShaderModuleDescriptor,
	) => GPUShaderModule;
	const origDestroy = bufferProto.destroy as (this: GPUBuffer) => void;

	const stats = (): PhaseStats | null => (current ? (phases.get(current) ?? null) : null);

	deviceProto.createBuffer = function (desc: GPUBufferDescriptor) {
		const s = stats();
		if (!s) return origCreateBuffer.call(this, desc);
		const t0 = performance.now();
		const buf = origCreateBuffer.call(this, desc);
		const dt = performance.now() - t0;
		s.createMs += dt;
		s.createN += 1;
		s.createBytes += desc.size;
		if (dt > s.createMaxMs) s.createMaxMs = dt;
		s.createTimesMs.push(dt);
		sizes.push(desc.size);
		return buf;
	};

	deviceProto.createBindGroup = function (desc: GPUBindGroupDescriptor) {
		const s = stats();
		if (!s) return origCreateBindGroup.call(this, desc);
		const t0 = performance.now();
		const bg = origCreateBindGroup.call(this, desc);
		const dt = performance.now() - t0;
		s.bindGroupMs += dt;
		s.bindGroupN += 1;
		return bg;
	};

	deviceProto.createShaderModule = function (desc: GPUShaderModuleDescriptor) {
		const mod = origCreateShaderModule.call(this, desc);
		// Record every shader the page ever creates. jax-js caches pipelines by
		// source hash, so the projection's shader appears once, during warmup;
		// gating on a phase would record nothing at all.
		shaders.push({ code: typeof desc.code === "string" ? desc.code : "" });
		return mod;
	};

	bufferProto.destroy = function () {
		const s = stats();
		if (!s) return origDestroy.call(this);
		const t0 = performance.now();
		origDestroy.call(this);
		const dt = performance.now() - t0;
		s.destroyMs += dt;
		s.destroyN += 1;
		if (dt > s.destroyMaxMs) s.destroyMaxMs = dt;
	};

	return {
		phases,
		sizes,
		shaders,
		begin(phase: Phase) {
			current = phase;
			if (!phases.has(phase)) phases.set(phase, emptyPhase());
		},
		end() {
			current = null;
		},
		reset() {
			for (const k of phases.keys()) phases.set(k, emptyPhase());
			sizes.length = 0;
		},
	};
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	if (!s.length) return NaN;
	const m = Math.floor(s.length / 2);
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

function summarizePhase(p: PhaseStats) {
	const sorted = [...p.createTimesMs].sort((a, b) => a - b);
	const q = (f: number) => (sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(f * sorted.length))] : NaN);
	return {
		createN: p.createN,
		createBytes: p.createBytes,
		createMs: +p.createMs.toFixed(2),
		/** Mean cost of one createBuffer, in microseconds. The headline number. */
		createMeanUs: +(1000 * (p.createMs / Math.max(1, p.createN))).toFixed(2),
		createMedianUs: +(1000 * (sorted.length ? sorted[sorted.length >> 1] : 0)).toFixed(2),
		createP90Us: +(1000 * q(0.9)).toFixed(2),
		createMaxUs: +(1000 * p.createMaxMs).toFixed(2),
		destroyN: p.destroyN,
		destroyMs: +p.destroyMs.toFixed(2),
		destroyMeanUs: +(1000 * (p.destroyMs / Math.max(1, p.destroyN))).toFixed(2),
		destroyMaxUs: +(1000 * p.destroyMaxMs).toFixed(2),
		bindGroupN: p.bindGroupN,
		bindGroupMs: +p.bindGroupMs.toFixed(2),
	};
}

// ---------------------------------------------------------------------------
// Phase A: the real decode step
// ---------------------------------------------------------------------------

/** One decode step exactly as bench/bench.ts times it, returning its logits. */
async function decodeStep(
	model: Gemma3,
	st: ReturnType<typeof createGemma3State>,
	token: number,
	inst: ReturnType<typeof instrument>,
) {
	const input = np.array([token], { dtype: np.uint32 });

	inst.begin("model");
	const tA = performance.now();
	const out = runGemma3Step(tree.ref(model), st, input);
	await blockUntilReady(out.latent);
	const tB = performance.now();
	inst.end();

	// The projection reads the latent, so it cannot overlap with the model.
	// This is the in-model projection cost.
	inst.begin("projection");
	const issue0 = performance.now();
	const logits = runLinear(
		{ weight: tree.ref(model).tokenEmbed },
		out.latent.ref.slice([-1]).flatten(),
	);
	const issue1 = performance.now();
	await blockUntilReady(logits);
	const issue2 = performance.now();
	inst.end();

	inst.begin("readback");
	const t0 = performance.now();
	const data = await logits.data();
	const readbackMs = performance.now() - t0;
	inst.end();

	const total = performance.now() - tA;
	out.latent.dispose();
	return {
		logits: data,
		stepMs: total,
		modelIssueMs: tB - tA,
		projectionWaitMs: issue2 - issue1,
		projectionIssueMs: issue1 - issue0,
		readbackMs,
		state: out.state,
	};
}

async function phaseA(inst: ReturnType<typeof instrument>, ctxLen: number, steps: number) {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const gemma = loadModelF32(safetensors.parse(await res.arrayBuffer()));

	const context = Array.from({ length: ctxLen }, (_, i) => 100 + (i % 200000));
	const prefill = async () => {
		const s0 = createGemma3State(gemma);
		const pre = runGemma3Step(tree.ref(gemma), s0, np.array(context, { dtype: np.uint32 }));
		await blockUntilReady(pre.latent);
		pre.latent.dispose();
		return pre.state;
	};

	// Warm pipeline compilation and the KV cache before anything is timed.
	let st = await prefill();
	const warm = await decodeStep(gemma, st, 100, inst);
	st = warm.state;
	inst.reset();
	inst.shaders.length = 0;

	const rows: Record<string, number>[] = [];
	let firstLogits: number[] | null = null;

	try {
		for (let i = 0; i < steps; i++) {
			inst.reset();
			if (i === 0) inst.shaders.length = 0; // capture the projection WGSL once
			const r = await decodeStep(gemma, st, 100 + i, inst);
			st = r.state;

			const modelPhase = summarizePhase(inst.phases.get("model") ?? emptyPhase());
			const projPhase = summarizePhase(inst.phases.get("projection") ?? emptyPhase());

			rows.push({
				step: i,
				stepMs: +r.stepMs.toFixed(2),
				modelIssueMs: +r.modelIssueMs.toFixed(2),
				projectionIssueMs: +r.projectionIssueMs.toFixed(2),
				projectionWaitMs: +r.projectionWaitMs.toFixed(2),
				readbackMs: +r.readbackMs.toFixed(2),
				allocModelMs: modelPhase.createMs,
				destroyModelMs: modelPhase.destroyMs,
				buffersModel: modelPhase.createN,
				allocProjMs: projPhase.createMs,
				buffersProj: projPhase.createN,
				createMeanModelUs: modelPhase.createMeanUs,
				bindGroupMs: +(modelPhase.bindGroupMs + projPhase.bindGroupMs).toFixed(2),
				allocShareOfStep: +(((modelPhase.createMs + projPhase.createMs) / r.stepMs) * 100).toFixed(1),
			});
			if (i === 0) firstLogits = Array.from(r.logits);
		}

		// --- independent numeric check -------------------------------------
		// Recompute five logits on the CPU from the read-back latent and
		// read-back weight rows. Nothing in the wrapped path can help this be
		// right by accident: a silent zero shows up immediately.
		const st3 = await prefill();
		const out3 = runGemma3Step(tree.ref(gemma), st3, np.array([100], { dtype: np.uint32 }));
		await blockUntilReady(out3.latent);
		const latentRow = out3.latent.ref.slice([-1]).flatten();
		const latent = Array.from(await latentRow.ref.data());
		const logitsArr = await (async () => {
			const lg = runLinear({ weight: tree.ref(gemma).tokenEmbed }, latentRow);
			await blockUntilReady(lg);
			return lg;
		})();
		void logitsArr;

		// One readback of the full logit vector. `.slice` consumes, so indexing
		// it five times would free it after the first.
		const gpuLogits = Array.from(await logitsArr.data());
		const probeIndices = [0, 1, 7, 137, 65535];
		const cpuDots: { index: number; gpu: number; cpu: number; absDiff: number }[] = [];
		for (const index of probeIndices) {
			const row = Array.from(
				await tree.ref(gemma).tokenEmbed.slice([index, index + 1]).flatten().data(),
			);
			let acc = 0;
			for (let k = 0; k < row.length; k++) acc += latent[k] * row[k];
			const gpu = gpuLogits[index];
			cpuDots.push({ index, gpu: +gpu.toFixed(4), cpu: +acc.toFixed(4), absDiff: +Math.abs(gpu - acc).toFixed(4) });
		}
		const worstAbsDiff = Math.max(...cpuDots.map((d) => d.absDiff));
		// fp32 with a 640-term sum: a few ulps of a magnitude-10 result.
		const numericOk = worstAbsDiff < 0.05;

		out3.latent.dispose();

		let argmax = 0;
		for (let i = 1; i < firstLogits!.length; i++) if (firstLogits![i] > firstLogits![argmax]) argmax = i;
		let finite = 0;
		for (const v of firstLogits!) if (Number.isFinite(v)) finite++;

		const allocTotal =
			median(rows.map((r) => r.allocModelMs)) + median(rows.map((r) => r.allocProjMs));

		return {
			ctxLen,
			steps,
			rows,
			medianStepMs: +median(rows.map((r) => r.stepMs)).toFixed(2),
			medianAllocMsPerStep: +allocTotal.toFixed(2),
			medianAllocModelMs: +median(rows.map((r) => r.allocModelMs)).toFixed(2),
			medianAllocProjMs: +median(rows.map((r) => r.allocProjMs)).toFixed(2),
			medianBuffersPerStep: median(rows.map((r) => r.buffersModel + r.buffersProj)),
			medianCreateUs: +median(rows.map((r) => r.createMeanModelUs)).toFixed(2),
			medianBindGroupMs: +median(rows.map((r) => r.bindGroupMs)).toFixed(2),
			medianProjectionWaitMs: +median(rows.map((r) => r.projectionWaitMs)).toFixed(2),
			// Size histogram of the last timed step, replayed by phase B.
			sizeHistogram: [...inst.sizes],
			projectionShaders: [...inst.shaders],
			numericCheck: { cpuDots, worstAbsDiff, ok: numericOk },
			firstStepArgmax: argmax,
			firstStepTopLogit: +firstLogits![argmax].toFixed(3),
			firstStepFiniteFraction: +(finite / firstLogits!.length).toFixed(3),
		};
	} finally {
		inst.phases.clear();
	}
}

// ---------------------------------------------------------------------------
// Phase B: the projection in isolation, fresh vs pooled
// ---------------------------------------------------------------------------

const ROWS = 262144; // vocabulary; [262144, 640] f32 = 640 MiB
const COLS = 640;
const THREADS = 65536; // 256 workgroups x 256 threads, as jax-js tunes it
const VEC4 = COLS / 4;
/** Workgroups for the coalesced reference; ROWS divides it evenly. */
const COALESCED_GROUPS = 4096;

/**
 * jax-js's actual projection kernel, verbatim in shape.
 *
 * Captured from `createShaderModule` on this tree
 * (bench/results/...artifacts.json, kernel 2630e65c): 65536 threads, each
 * owning FOUR consecutive rows, walking all 640 columns as four scalar loads
 * sixteen bytes apart. Consecutive lanes are 4 rows (10 KiB) apart, so a warp
 * touches 32 sectors 10 KiB apart for every 16 B it uses. Reproducing this
 * exactly is the point: a row-per-thread probe would answer a different
 * question.
 */
const jaxProjection = `
@group(0) @binding(0) var<storage, read> in0 : array<f32>;
@group(0) @binding(1) var<storage, read> in1 : array<f32>;
@group(0) @binding(2) var<storage, read_write> result : array<f32>;
const COLS: u32 = ${COLS}u;
const THREADS: u32 = ${THREADS}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
	let gidx: u32 = id.x;
	if (gidx >= THREADS) { return; }
	var acc0: f32 = f32(0);
	var acc1: f32 = f32(0);
	var acc2: f32 = f32(0);
	var acc3: f32 = f32(0);
	// Row block: eight threads share 32 consecutive rows, four each.
	let rowBase: u32 = (gidx / 8u) * 32u + (gidx % 8u) * 4u;
	for (var ridx: u32 = 0u; ridx < ${COLS / 4}u; ridx++) {
		let c: u32 = ridx * 4u;
		let a0: f32 = in0[c];
		let a1: f32 = in0[c + 1u];
		let a2: f32 = in0[c + 2u];
		let a3: f32 = in0[c + 3u];
		let base: u32 = rowBase * COLS + c;
		acc0 += a0 * in1[base] + a1 * in1[base + 1u] + a2 * in1[base + 2u] + a3 * in1[base + 3u];
		acc1 += a0 * in1[base + COLS] + a1 * in1[base + COLS + 1u] + a2 * in1[base + COLS + 2u] + a3 * in1[base + COLS + 3u];
		acc2 += a0 * in1[base + 2u * COLS] + a1 * in1[base + 2u * COLS + 1u] + a2 * in1[base + 2u * COLS + 2u] + a3 * in1[base + 2u * COLS + 3u];
		acc3 += a0 * in1[base + 3u * COLS] + a1 * in1[base + 3u * COLS + 1u] + a2 * in1[base + 3u * COLS + 2u] + a3 * in1[base + 3u * COLS + 3u];
	}
	result[rowBase] = acc0;
	result[rowBase + 1u] = acc1;
	result[rowBase + 2u] = acc2;
	result[rowBase + 3u] = acc3;
}`;

/**
 * Identical row ownership and work, vec4 loads instead of four scalars.
 * Isolates load width: same bytes per thread, same stride between lanes, same
 * thread count. If this alone is fast, the shipped kernel's problem is that it
 * issues 32-bit loads it does not have to.
 */
const jaxProjectionVec4 = `
@group(0) @binding(0) var<storage, read> in0 : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> in1 : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> result : array<f32>;
const C4: u32 = ${VEC4}u;
const THREADS: u32 = ${THREADS}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
	let gidx: u32 = id.x;
	if (gidx >= THREADS) { return; }
	let rowBase: u32 = (gidx / 8u) * 32u + (gidx % 8u) * 4u;
	var acc0: vec4<f32> = vec4<f32>(0.0);
	var acc1: vec4<f32> = vec4<f32>(0.0);
	var acc2: vec4<f32> = vec4<f32>(0.0);
	var acc3: vec4<f32> = vec4<f32>(0.0);
	for (var ridx: u32 = 0u; ridx < C4; ridx++) {
		let a: vec4<f32> = in0[ridx];
		let base: u32 = rowBase * C4 + ridx;
		acc0 = acc0 + a * in1[base];
		acc1 = acc1 + a * in1[base + C4];
		acc2 = acc2 + a * in1[base + 2u * C4];
		acc3 = acc3 + a * in1[base + 3u * C4];
	}
	result[rowBase] = acc0.x + acc0.y + acc0.z + acc0.w;
	result[rowBase + 1u] = acc1.x + acc1.y + acc1.z + acc1.w;
	result[rowBase + 2u] = acc2.x + acc2.y + acc2.z + acc2.w;
	result[rowBase + 3u] = acc3.x + acc3.y + acc3.z + acc3.w;
}`;

/**
 * One 32-thread workgroup per row, coalesced: the 32 addresses of one load
 * instruction are consecutive within the row, instead of 32 rows 10 KiB apart.
 * 160 vec4 per row over 32 lanes is exactly 5 iterations.
 *
 * Deliberately no subgroups intrinsic and no early return: the loop count is a
 * compile-time constant so control flow is uniform, which is what both
 * `subgroupAdd` and `workgroupBarrier` require. An earlier version branched on
 * `row >= arrayLength(&result)` and subgroupAdd refused to compile; the probe
 * caught it because every kernel is verified before it is timed.
 */
const coalescedPerRow = `
@group(0) @binding(0) var<storage, read> in0 : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> in1 : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> result : array<f32>;
const C4: u32 = ${VEC4}u;
const GROUPS: u32 = ${COALESCED_GROUPS}u;
const ROW_STRIDE: u32 = ${COALESCED_GROUPS}u;
var<workgroup> part: array<f32, 32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
	// Fixed trip count, computed from constants: uniform for every invocation.
	var row: u32 = wid.x;
	for (var iter: u32 = 0u; iter < ${ROWS / COALESCED_GROUPS}u; iter = iter + 1u) {
		let base: u32 = row * C4;
		var acc: f32 = 0.0;
		for (var i: u32 = 0u; i < C4; i = i + 32u) {
			let k: u32 = i + li;
			let a: vec4<f32> = in0[k];
			let v: vec4<f32> = in1[base + k];
			acc = acc + v.x * a.x + v.y * a.y + v.z * a.z + v.w * a.w;
		}
		part[li] = acc;
		workgroupBarrier();
		var step: u32 = 16u;
		loop {
			if (step == 0u) { break; }
			if (li < step) { part[li] = part[li] + part[li + step]; }
			workgroupBarrier();
			step = step / 2u;
		}
		if (li == 0u) { result[row] = part[0]; }
		workgroupBarrier();
		row = row + ROW_STRIDE;
	}
}`;

/** Pure streaming read of the same bytes: this session's ceiling. */
const streamRead = `
@group(0) @binding(0) var<storage, read> in1 : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
const TOTAL4: u32 = ${(ROWS * COLS) / 4}u;
const GROUPS: u32 = 4096u;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
	let lane: u32 = wid.x * 256u + li;
	var acc: vec4<f32> = vec4<f32>(0.0);
	var i: u32 = lane;
	loop {
		if (i >= TOTAL4) { break; }
		acc = acc + in1[i];
		i = i + GROUPS * 256u;
	}
	if (li == 0u) { out[wid.x] = acc.x + acc.y + acc.z + acc.w; }
}`;

async function phaseB(sizeHistogram: number[]) {
	const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
	if (!adapter) throw new Error("no adapter");
	const device = await adapter.requestDevice({
		// Without these, a 640 MiB buffer silently fails to create.
		requiredLimits: {
			maxBufferSize: adapter.limits.maxBufferSize,
			maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
			maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
			maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
			maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
			maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
			maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
			maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
			maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
		},
	});

	const errors: string[] = [];
	device.addEventListener("uncapturederror", (e) =>
		errors.push(String((e as GPUUncapturedErrorEvent).error.message)),
	);

	const W_BYTES = ROWS * COLS * 4;
	const limits = {
		maxBufferSize: device.limits.maxBufferSize,
		maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
		adapterMaxBufferSize: adapter.limits.maxBufferSize,
		adapterMaxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
	};
	if (W_BYTES > device.limits.maxStorageBufferBindingSize) {
		throw new Error(
			`weight buffer ${W_BYTES} exceeds maxStorageBufferBindingSize ${device.limits.maxStorageBufferBindingSize}`,
		);
	}

	// Synthetic weights: 4096 distinct rows tiled to 262144. Same bytes, same
	// stride, same alignment, and deterministic so the CPU reference is cheap.
	const data = new Float32Array(ROWS * COLS);
	const block = new Float32Array(4096 * COLS);
	for (let i = 0; i < block.length; i++) block[i] = (((i * 2654435761) >>> 13) % 2001) / 1000 - 1;
	data.set(block, 0);
	for (let rows = 4096; rows < ROWS; rows *= 2) data.copyWithin(rows * COLS, 0, rows * COLS);
	const x = new Float32Array(COLS);
	for (let i = 0; i < COLS; i++) x[i] = (((i * 40503) % 101) - 50) / 50;

	const mk = (size: number, usage: GPUBufferUsageFlags) => device.createBuffer({ size, usage });
	const wBuf = mk(W_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
	device.queue.writeBuffer(wBuf, 0, data);
	const xBuf = mk(x.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
	device.queue.writeBuffer(xBuf, 0, x);
	const stage = mk(4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

	const Y_BYTES = ROWS * 4;
	const yFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;

	const makePipe = (code: string) => {
		const mod = device.createShaderModule({ code });
		const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
		return { pipe, layout: pipe.getBindGroupLayout(0) };
	};

	const shipped = makePipe(jaxProjection);
	const shippedVec4 = makePipe(jaxProjectionVec4);
	const coalesced = makePipe(coalescedPerRow);
	const streamPipe = makePipe(streamRead);

	/** One dispatch + submit + 4-byte readback, exactly the shipped ordering. */
	const dispatch = async (opts: {
		pipe: GPUComputePipeline;
		bg: GPUBindGroup;
		y: GPUBuffer;
		groups: number;
	}) => {
		const enc = device.createCommandEncoder();
		const pass = enc.beginComputePass();
		pass.setPipeline(opts.pipe);
		pass.setBindGroup(0, opts.bg);
		pass.dispatchWorkgroups(opts.groups);
		pass.end();
		enc.copyBufferToBuffer(opts.y, 0, stage, 0, 4);
		device.queue.submit([enc.finish()]);
		await stage.mapAsync(GPUMapMode.READ);
		stage.unmap();
	};

	/** jax-js's shape: fresh output buffer + bind group per call, destroyed
	 *  immediately after submit, before the GPU has finished. */
	const dispatchFresh = async (p: { pipe: GPUComputePipeline; layout: GPUBindGroupLayout }) => {
		const y = mk(Y_BYTES, yFlags);
		const bg = device.createBindGroup({
			layout: p.layout,
			entries: [
				{ binding: 0, resource: { buffer: xBuf } }, // in0 = latent x
				{ binding: 1, resource: { buffer: wBuf } }, // in1 = weights
				{ binding: 2, resource: { buffer: y } },
			],
		});
		const enc = device.createCommandEncoder();
		const pass = enc.beginComputePass();
		pass.setPipeline(p.pipe);
		pass.setBindGroup(0, bg);
		pass.dispatchWorkgroups(256);
		pass.end();
		enc.copyBufferToBuffer(y, 0, stage, 0, 4);
		device.queue.submit([enc.finish()]);
		y.destroy(); // ship order: submit then destroy, no wait
		await stage.mapAsync(GPUMapMode.READ);
		stage.unmap();
	};

	/** Verify a projection-shaped kernel against a scalar CPU reference. */
	const verifyProjection = async (p: { pipe: GPUComputePipeline; layout: GPUBindGroupLayout }, groups = 256) => {
		const y = mk(Y_BYTES, yFlags);
		const big = mk(Y_BYTES, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
		const bg = device.createBindGroup({
			layout: p.layout,
			entries: [
				{ binding: 0, resource: { buffer: xBuf } },
				{ binding: 1, resource: { buffer: wBuf } },
				{ binding: 2, resource: { buffer: y } },
			],
		});
		const enc = device.createCommandEncoder();
		const pass = enc.beginComputePass();
		pass.setPipeline(p.pipe);
		pass.setBindGroup(0, bg);
		pass.dispatchWorkgroups(groups);
		pass.end();
		enc.copyBufferToBuffer(y, 0, big, 0, Y_BYTES);
		device.queue.submit([enc.finish()]);
		await big.mapAsync(GPUMapMode.READ);
		const got = new Float32Array(big.getMappedRange()).slice();
		big.unmap();

		let worst = 0;
		for (const row of [0, 1, 3, 4, 4095, 4096, 100003, 262143]) {
			let ref = 0;
			const base = row * COLS;
			for (let k = 0; k < COLS; k++) ref += data[base + k] * x[k];
			worst = Math.max(worst, Math.abs(ref - got[row]));
		}
		y.destroy();
		big.destroy();
		return {
			worstAbsErr: +worst.toExponential(2),
			row0: +got[0].toFixed(4),
			// A silently-skipped kernel leaves zeros. Count them, do not eyeball.
			nonzeroRows: got.reduce((n, v) => n + (v !== 0 ? 1 : 0), 0),
		};
	};

	/** Verify the pure streaming read: sum of every f32 lane, compared to CPU. */
	const verifyStream = async () => {
		const groups = 4096;
		const out = mk(groups * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
		const stageBuf = mk(groups * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
		const bg = device.createBindGroup({
			layout: streamPipe.layout,
			entries: [
				{ binding: 0, resource: { buffer: wBuf } },
				{ binding: 1, resource: { buffer: out } },
			],
		});
		const enc = device.createCommandEncoder();
		const pass = enc.beginComputePass();
		pass.setPipeline(streamPipe.pipe);
		pass.setBindGroup(0, bg);
		pass.dispatchWorkgroups(groups);
		pass.end();
		enc.copyBufferToBuffer(out, 0, stageBuf, 0, groups * 4);
		device.queue.submit([enc.finish()]);
		await stageBuf.mapAsync(GPUMapMode.READ);
		// Copy before unmapping: the mapped-view buffer is detached afterwards.
		const got = new Float32Array(stageBuf.getMappedRange()).slice();
		stageBuf.unmap();
		const sum = got.reduce((a, b) => a + b, 0);

		let ref = 0;
		// fp32 accumulation order differs; a sampled reference is enough to
		// prove the read happened, which is all this variant claims.
		for (let i = 0; i < data.length; i += 997) ref += data[i];
		out.destroy();
		stageBuf.destroy();
		return {
			gpuSumOverAll: +sum.toFixed(2),
			cpuSumOfSample: +ref.toFixed(2),
			sampledLanes: Math.ceil(data.length / 997),
			// Every group must have read something; a skipped dispatch leaves 0s.
			nonzero: got.reduce((n, v) => n + (v !== 0 ? 1 : 0), 0),
		};
	};

	/**
	 * Time several variants INTERLEAVED, round-robin, discarding the first round.
	 *
	 * Block-sequential timing of A then B was tried first and was misleading: a
	 * repeat of variant A measured 3 ms slower than its own first block, which is
	 * bigger than the effect being looked for. The GPU clock and the host's
	 * thermal state drift over a run, so any A-vs-B difference measured in
	 * separate blocks is confounded with that drift. Round-robin makes drift hit
	 * every variant equally; the per-round spread is reported so the reader can
	 * see the noise rather than take a single median on faith.
	 */
	const timeInterleaved = async (variants: { label: string; fn: () => Promise<unknown> }[], rounds = 7) => {
		const times = new Map<string, number[]>(variants.map((v) => [v.label, []]));
		// One discarded warm round per variant, so no variant pays a first-call cost.
		for (const v of variants) await v.fn();
		for (let r = 0; r < rounds; r++) {
			for (const v of variants) {
				const t0 = performance.now();
				await v.fn();
				times.get(v.label)!.push(performance.now() - t0);
			}
		}
		return variants.map((v) => {
			const xs = times.get(v.label)!;
			return {
				label: v.label,
				medianMs: +median(xs).toFixed(2),
				minMs: +Math.min(...xs).toFixed(2),
				maxMs: +Math.max(...xs).toFixed(2),
				samples: xs.map((x) => +x.toFixed(1)),
			};
		});
	};

	const timed = async (fn: () => Promise<unknown>, warm = 2, reps = 5) => {
		for (let i = 0; i < warm; i++) await fn();
		const times: number[] = [];
		for (let i = 0; i < reps; i++) {
			const t0 = performance.now();
			await fn();
			times.push(performance.now() - t0);
		}
		return {
			medianMs: +median(times).toFixed(2),
			minMs: +Math.min(...times).toFixed(2),
			maxMs: +Math.max(...times).toFixed(2),
		};
	};

	const gbps = (ms: number) => +(W_BYTES / 1e9 / (ms / 1000)).toFixed(1);
	const results: Record<string, unknown>[] = [];

	// --- verify every kernel before timing anything -----------------------
	const verifyShipped = await verifyProjection(shipped);
	const verifyVec4 = await verifyProjection(shippedVec4);
	const verifyCoalesced = await verifyProjection(coalesced, COALESCED_GROUPS);
	const verifyStreamRun = await verifyStream();
	const projectionChecks = [verifyShipped, verifyVec4, verifyCoalesced];
	const verified =
		projectionChecks.every((v) => v.worstAbsErr < 1e-2 && v.nonzeroRows > ROWS * 0.99) &&
		verifyStreamRun.nonzero === 4096;
	if (!verified) {
		throw new Error(
			`a probe kernel is incorrect, refusing to time any of them: ${JSON.stringify({
				verifyShipped,
				verifyVec4,
				verifyCoalesced,
				verifyStreamRun,
				errors,
			})}`,
		);
	}

	// Pooled output buffer + bind group, shared by the pooled variants.
	const yPool = mk(Y_BYTES, yFlags);
	const mkBg = (layout: GPUBindGroupLayout, y: GPUBuffer) =>
		device.createBindGroup({
			layout,
			entries: [
				{ binding: 0, resource: { buffer: xBuf } },
				{ binding: 1, resource: { buffer: wBuf } },
				{ binding: 2, resource: { buffer: y } },
			],
		});
	const bgShipped = mkBg(shipped.layout, yPool);

	// --- 1. pooled (control): allocation fully hoisted ---------------------
	const pooled = await timed(() =>
		dispatch({ pipe: shipped.pipe, bg: bgShipped, y: yPool, groups: 256 }),
	);
	results.push({ case: "shipped shape, pooled buffers", ...pooled, gbps: gbps(pooled.medianMs) });

	// --- 2. fresh vs pooled, interleaved -----------------------------------
	// The headline comparison: does allocating the 1 MiB output buffer per call
	// cost anything next to a 640 MiB read? Same kernel, same weights, same
	// readback; only the allocation differs.
	{
		const inter = await timeInterleaved(
			[
				{
					label: "pooled",
					fn: () => dispatch({ pipe: shipped.pipe, bg: bgShipped, y: yPool, groups: 256 }),
				},
				{ label: "fresh alloc + destroy after submit", fn: () => dispatchFresh(shipped) },
			],
			7,
		);
		for (const r of inter) {
			results.push({ case: `shipped shape, ${r.label} [interleaved]`, ...r, gbps: gbps(r.medianMs) });
		}
	}

	// --- 3. pooled + the step's real allocation churn around it ------------
	const makeChurn = () => {
		const made: GPUBuffer[] = [];
		for (const size of sizeHistogram) {
			made.push(
				mk(Math.max(4, size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
			);
		}
		return made;
	};
	const churnOnly = await timed(() => {
		const made = makeChurn();
		for (const b of made) b.destroy();
	});
	results.push({
		case: `allocation churn only (${sizeHistogram.length} create + destroy)`,
		...churnOnly,
		gbps: 0,
	});
	const pooledChurn = await timed(async () => {
		const made = makeChurn();
		await dispatch({ pipe: shipped.pipe, bg: bgShipped, y: yPool, groups: 256 });
		for (const b of made) b.destroy();
	});
	results.push({
		case: `shipped shape, pooled + ${sizeHistogram.length} buffers churned around it`,
		...pooledChurn,
		gbps: gbps(pooledChurn.medianMs),
	});

	// --- 3b. churn, but handed back BEFORE the dispatch --------------------
	// Distinguishes "allocating hurts the following kernel" from "havING 47 MB
	// of extra live buffers hurts it". The kernel is identical in both.
	const churnThenRelease = await timed(async () => {
		const made = makeChurn();
		for (const b of made) b.destroy();
		await dispatch({ pipe: shipped.pipe, bg: bgShipped, y: yPool, groups: 256 });
	});
	results.push({
		case: `shipped shape, pooled, ${sizeHistogram.length} buffers churned + released first`,
		...churnThenRelease,
		gbps: gbps(churnThenRelease.medianMs),
	});

	// --- 3c. is the churn penalty bytes, or buffer count? -----------------
	// The two candidate mechanisms have different fixes, so they need separating.
	// Same live bytes in ONE buffer vs the same live bytes across 471 buffers
	// tells us whether the cost is memory pressure or per-buffer bookkeeping.
	// Also a tiny-size 471-buffer case: per-buffer cost with no bytes at all.
	{
		const totalChurnBytes = sizeHistogram.reduce((a, b) => a + b, 0);
		const manyTiny = () =>
			Array.from({ length: sizeHistogram.length }, () =>
				mk(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
			);
		const release = (bs: GPUBuffer[]) => bs.forEach((b) => b.destroy());

		const inter = await timeInterleaved(
			[
				{ label: "pooled, no churn", fn: () => dispatch({ pipe: shipped.pipe, bg: bgShipped, y: yPool, groups: 256 }) },
				{
					// Same bytes as the churn case, but in ONE allocation: separates
					// "live memory pressure" from "per-buffer bookkeeping".
					label: `pooled + 1 buffer of ${(totalChurnBytes / 2 ** 20).toFixed(0)} MiB live`,
					fn: async () => {
						const big = mk(
							Math.max(4, totalChurnBytes),
							GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
						);
						await dispatch({ pipe: shipped.pipe, bg: bgShipped, y: yPool, groups: 256 });
						big.destroy();
					},
				},
				{
					label: `pooled + ${sizeHistogram.length} tiny (4 B) buffers live`,
					fn: async () => {
						const made = manyTiny();
						await dispatch({ pipe: shipped.pipe, bg: bgShipped, y: yPool, groups: 256 });
						release(made);
					},
				},
				{
					label: `pooled + ${sizeHistogram.length} real-size buffers live`,
					fn: async () => {
						const made = makeChurn();
						await dispatch({ pipe: shipped.pipe, bg: bgShipped, y: yPool, groups: 256 });
						release(made);
					},
				},
			],
			5,
		);
		for (const r of inter) {
			results.push({ case: `${r.label} [interleaved]`, ...r, gbps: gbps(r.medianMs) });
		}
	}

	// --- 3d. THE decisive pool question ------------------------------------
	// A pool does not make buffers disappear; it keeps them alive and reuses
	// them, so per step it removes the create/destroy churn but leaves the same
	// buffers resident. That is a different state from "no churn and nothing
	// resident". Timing all three separates the two:
	//
	//   resident + churn     = what jax-js does today (create/free every step)
	//   resident, no churn   = what a pool would give you
	//   not resident         = the idealised floor
	//
	// If pool-state is close to today, a pool cannot be the win regardless of
	// what the create/destroy calls themselves cost.
	{
		const resident = sizeHistogram.map((sz) =>
			mk(Math.max(4, sz), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
		);
		const inter = await timeInterleaved(
			[
				{
					label: "today: 471 resident + churn every call",
					fn: async () => {
						const made = sizeHistogram.map((sz) =>
							mk(Math.max(4, sz), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
						);
						await dispatch({ pipe: shipped.pipe, bg: bgShipped, y: yPool, groups: 256 });
						made.forEach((b) => b.destroy());
					},
				},
				{
					label: "with a pool: 471 resident, no churn",
					fn: () => dispatch({ pipe: shipped.pipe, bg: bgShipped, y: yPool, groups: 256 }),
				},
			],
			7,
		);
		for (const r of inter) {
			results.push({ case: `${r.label} [interleaved]`, ...r, gbps: gbps(r.medianMs) });
		}
		resident.forEach((b) => b.destroy());
	}

	// --- 4. same rows, vec4 loads: isolates load width ---------------------
	const vec4 = await timed(() =>
		dispatch({ pipe: shippedVec4.pipe, bg: mkBg(shippedVec4.layout, yPool), y: yPool, groups: 256 }),
	);
	results.push({ case: "same rows/threads, vec4 loads, pooled", ...vec4, gbps: gbps(vec4.medianMs) });

	// --- 5. warp-per-row: isolates coalescing ------------------------------
	const coalescedBg = mkBg(coalesced.layout, yPool);
	const coalescedRun = await timed(() =>
		dispatch({ pipe: coalesced.pipe, bg: coalescedBg, y: yPool, groups: COALESCED_GROUPS }),
	);
	results.push({
		case: "coalesced: one 32-lane row/group, pooled",
		...coalescedRun,
		gbps: gbps(coalescedRun.medianMs),
	});

	// --- 6. pure streaming read: this session's ceiling --------------------
	const streamGroups = 4096;
	const outBuf = mk(streamGroups * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
	const streamBg = device.createBindGroup({
		layout: streamPipe.layout,
		entries: [
			{ binding: 0, resource: { buffer: wBuf } },
			{ binding: 1, resource: { buffer: outBuf } },
		],
	});
	const stream = await timed(() => dispatch({ pipe: streamPipe.pipe, bg: streamBg, y: outBuf, groups: streamGroups }));
	results.push({ case: "pure streaming read of the same 640 MiB", ...stream, gbps: gbps(stream.medianMs) });

	const overPeak = results.filter((r) => typeof r.gbps === "number" && (r.gbps as number) > 192);

	yPool.destroy();
	outBuf.destroy();
	wBuf.destroy();
	xBuf.destroy();
	stage.destroy();

	return {
		deviceLimits: limits,
		weightMiB: +(W_BYTES / 2 ** 20).toFixed(0),
		verify: { shipped: verifyShipped, shippedVec4: verifyVec4, coalesced: verifyCoalesced, stream: verifyStreamRun, verified },
		churnBuffers: sizeHistogram.length,
		results,
		overPeak,
		gpuErrors: errors.slice(0, 5),
	};
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

export async function run(opts: { ctxLen?: number; steps?: number } = {}) {
	const inst = instrument();
	const ctxLen = opts.ctxLen ?? 1024;
	const steps = opts.steps ?? 6;

	const a = await phaseA(inst, ctxLen, steps);
	const b = await phaseB(a.sizeHistogram);

	return {
		phaseA: {
			...a,
			projectionShaders: a.projectionShaders.map((s) => ({
				chars: s.code.length,
				head: s.code.split("\n").slice(0, 6).join("\n"),
				hasArrayLength: s.code.includes("arrayLength"),
				workgroupSize: /@workgroup_size\(([^)]*)\)/.exec(s.code)?.[1] ?? null,
				readsWeightLike: /array<f32>|array<vec4<f32>>/.test(s.code),
			})),
		},
		phaseB: b,
		summary: {
			medianStepMs: a.medianStepMs,
			medianAllocMsPerStep: a.medianAllocMsPerStep,
			allocShareOfStepPct: +((a.medianAllocMsPerStep / a.medianStepMs) * 100).toFixed(1),
			medianProjectionWaitMs: a.medianProjectionWaitMs,
			projectionShareOfStepPct: +((a.medianProjectionWaitMs / a.medianStepMs) * 100).toFixed(1),
			projectionGbpsInModel: a.projectionGbps,
			numericCheckOk: a.numericCheck.ok,
		},
	};
}
