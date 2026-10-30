/**
 * The vocabulary projection: [640] x [640, V] -> [V]. This is the logits step,
 * and it is the single largest GPU region in decode: it streams the whole tied
 * embedding matrix (640 MiB at V=262144, f32) to produce one vector of numbers,
 * once per token. Nothing about it is arithmetic-bound; it is a bandwidth test.
 *
 * jax-js lowers this to a row-per-thread matvec and reaches 21.9 GB/s isolated
 * (8.6 GB/s in-model) against a measured read ceiling of 101-140 GB/s on this
 * card. The deficit is entirely access pattern: one thread owns one output row,
 * so a warp touches 32 rows that are 2560 B apart and each lane load pulls a
 * whole 128 B cache line to use 4 B of it.
 *
 * This module runs a warp-per-row kernel instead: the 32 lanes of a subgroup
 * share one output row, each striding k by 32, so a warp's loads cover 128
 * contiguous bytes per iteration and the row is walked sequentially. Lanes are
 * combined with `subgroupShuffleXor`.
 *
 * Two things make that possible only from here rather than from array ops:
 *   - `subgroups` must be a *device* feature, which means asking for it at
 *     `requestDevice` time. patches/jax-js-subgroups.patch adds it to jax-js's
 *     own request so the weights (which live on that device) can be read by it.
 *   - jax-js has no primitive that expresses "32 threads reduce one output".
 *
 * Correctness is not assumed. `projectVocab` validates the shape and dtype,
 * refuses anything it does not understand, and falls back to the stock
 * `runLinear` path. The correctness gate (`npm run bench:check`) runs the real
 * model through this code, so a kernel that computes the wrong thing fails
 * there rather than silently changing logits.
 */

import { DType, getWebGPUDevice, numpy as np } from "@jax-js/jax";

/** Warps per workgroup; each warp owns one output row. */
const WARPS_PER_WORKGROUP = 8;
const WORKGROUP_SIZE = WARPS_PER_WORKGROUP * 32;

/** WebGPU's hard limit on workgroups per dispatch dimension. */
const MAX_WORKGROUPS = 65535;

type Dtype = "f32" | "f16";

function shaderSource(cols: number, rows: number, dtype: Dtype): string {
	const enable = dtype === "f16" ? `enable subgroups;\nenable f16;` : `enable subgroups;`;
	const ty = dtype === "f16" ? "f16" : "f32";
	const castOut = dtype === "f16" ? "f16(acc)" : "acc";

	return `${enable}

@group(0) @binding(0) var<storage, read> w: array<${ty}>;
@group(0) @binding(1) var<storage, read> x: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> y: array<${ty}>;

const COLS: u32 = ${cols}u;
const ROWS: u32 = ${rows}u;
const WARPS: u32 = ${WARPS_PER_WORKGROUP}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
	@builtin(workgroup_id) wid: vec3<u32>,
	@builtin(local_invocation_index) li: u32,
) {
	let warp = li / 32u;
	let lane = li % 32u;
	let row = wid.x * WARPS + warp;

	// One row per warp, and the grid is sized so every row is covered by exactly
	// one warp. There is deliberately no grid-stride loop: subgroupShuffleXor
	// is only valid in *uniform* control flow, and a break on row >= ROWS
	// depends on the warp index, so the compiler rejects the shuffle inside it
	// ("must only be called from subgroup uniform control flow"). Sizing the
	// grid instead keeps the whole body uniform, and the vocabulary is small
	// enough that ceil(262144/8) = 32768 workgroups is half the 65535 limit.
	//
	// The guard is a plain return after the reduce would also be wrong: a lane
	// that exits early no longer participates in its group's shuffle. Since the
	// grid covers rows exactly, the out-of-range case does not arise; the check
	// is kept for a shape whose row count is not a multiple of 8, where the extra
	// warps read past the end and must not write.
	let inRange = row < ROWS;
	// Clamp rather than return: every lane of every warp then reaches the
	// shuffle together, and out-of-range warps compute a scratch row harmlessly.
	let safeRow = select(0u, row, inRange);

	let base = safeRow * COLS;
	// f32 accumulation regardless of storage type: summing 640 products in f16
	// would lose most of the mantissa on the way to the final value.
	var acc: f32 = 0.0;
	// Lane-strided: iteration i reads w[base + lane + 32*i], so the 32 lanes of a
	// warp cover 128 contiguous bytes and the next iteration reads the 128 bytes
	// after that. Every load is a full cache line.
	for (var k: u32 = lane; k < COLS; k = k + 32u) {
		acc = acc + f32(w[base + k]) * f32(x[k]);
	}

	// Butterfly reduce inside the subgroup. All 32 lanes hold the same row, so
	// control flow is uniform within the subgroup and the shuffle is well
	// defined. No workgroupBarrier is used or needed: a barrier inside
	// non-uniform control flow is UB and skips the kernel silently, which is how
	// this shape once "measured" 550 GB/s.
	acc = acc + subgroupShuffleXor(acc, 16u);
	acc = acc + subgroupShuffleXor(acc, 8u);
	acc = acc + subgroupShuffleXor(acc, 4u);
	acc = acc + subgroupShuffleXor(acc, 2u);
	acc = acc + subgroupShuffleXor(acc, 1u);

	if (lane == 0u && inRange) { y[row] = ${castOut}; }
}
`;
}

type Compiled = {
	pipeline: GPUComputePipeline;
	layout: GPUBindGroupLayout;
	workgroups: number;
};

/**
 * Compiled pipelines, keyed by everything baked into the shader source. Built
 * once: a shader compile in the decode loop is the exact regression the
 * device-scalar position fix existed to remove.
 */
const pipelines = new WeakMap<GPUDevice, Map<string, Compiled>>();

function compile(device: GPUDevice, cols: number, rows: number, dtype: Dtype): Compiled | null {
	if (!device.features.has("subgroups" as GPUFeatureName)) return null;
	if (dtype === "f16" && !device.features.has("shader-f16" as GPUFeatureName)) return null;

	let byKey = pipelines.get(device);
	if (!byKey) {
		byKey = new Map();
		pipelines.set(device, byKey);
	}
	const key = `${cols}x${rows}x${dtype}`;
	const hit = byKey.get(key);
	if (hit) return hit;

	const workgroups = Math.min(MAX_WORKGROUPS, Math.ceil(rows / WARPS_PER_WORKGROUP));
	const module = device.createShaderModule({
		code: shaderSource(cols, rows, dtype),
	});
	const pipeline = device.createComputePipeline({
		layout: "auto",
		compute: { module, entryPoint: "main" },
	});
	// A compile error must not become a silent no-op that writes nothing: zeros
	// are a plausible-looking logits vector and would propagate as a wrong token
	// with no error anywhere. Surface it, and let the caller fall back.
	device.pushErrorScope("validation");
	const compiled: Compiled = { pipeline, layout: pipeline.getBindGroupLayout(0), workgroups };
	void device.popErrorScope().then((err) => {
		if (err) console.error(`[vocab-projection] shader compile failed: ${err.message}`);
	});

	byKey.set(key, compiled);
	return compiled;
}

function deviceOrNull(): GPUDevice | null {
	try {
		return getWebGPUDevice() ?? null;
	} catch {
		return null;
	}
}

/**
 * Why the kernel declined a call, for callers that want to log it. `null` when
 * the kernel would have handled it. Declining is legal, but a *silent* decline
 * on the hot path would look like a working optimization while quietly running
 * the slow path, so callers can ask and say so once.
 */
export function projectVocabDecline(weight: np.Array, x: np.Array): string | null {
	if (weight.shape.length !== 2 || x.shape.length !== 1) return "not a matrix x vector";
	if (weight.shape[1] !== x.shape[0]) return "inner dimensions do not match";
	const dtype = dtypeOf(weight);
	if (!dtype) return `unsupported weight dtype ${weight.dtype}`;
	if (dtypeOf(x) !== dtype) return `activation dtype ${x.dtype} differs from weight ${weight.dtype}`;
	const device = deviceOrNull();
	if (!device) return "not on the WebGPU backend";
	if (!device.features.has("subgroups" as GPUFeatureName)) {
		return "device lacks the subgroups feature (see patches/jax-js-subgroups.patch)";
	}
	if (dtype === "f16" && !device.features.has("shader-f16" as GPUFeatureName)) {
		return "f16 weights but no shader-f16 feature";
	}
	return null;
}

function dtypeOf(weight: np.Array): Dtype | null {
	if (weight.dtype === DType.Float32) return "f32";
	if (weight.dtype === DType.Float16) return "f16";
	return null;
}

/**
 * Logits for one activation vector, or `null` if this shape/dtype/device is not
 * one the kernel handles. A `null` return is a request to use the stock path,
 * not an error: the f16/f32 split, the backend, and the vocabulary size all
 * vary, and the fallback is always correct.
 *
 * Ownership: consumes both `weight` and `x`, matching `runLinear`.
 */
export function projectVocab(weight: np.Array, x: np.Array): np.Array | null {
	const weightShape = weight.shape;
	const xShape = x.shape;
	if (weightShape.length !== 2 || xShape.length !== 1) return null;

	const [rows, cols] = weightShape;
	if (cols !== xShape[0]) return null;

	const dtype = dtypeOf(weight);
	if (!dtype || dtypeOf(x) !== dtype) return null;

	const device = deviceOrNull();
	if (!device) return null;

	const compiled = compile(device, cols, rows, dtype);
	if (!compiled) return null;

	// `gpuBufferSync` realizes the array and hands back the buffer that now backs
	// it, materialising a contiguous row-major copy if the view was not already
	// one. That is what makes reading it as a flat array here valid, and it is
	// why the size check below is a real guard rather than a formality.
	const wBuf = weight.ref.gpuBufferSync();
	const xBuf = x.ref.gpuBufferSync();
	const wBytes = rows * cols * (dtype === "f16" ? 2 : 4);
	if (wBuf.size < wBytes || xBuf.size < cols * (dtype === "f16" ? 2 : 4)) {
		weight.dispose();
		x.dispose();
		return null;
	}

	// A zeroed destination owned by jax-js, so the result is an ordinary array
	// that the rest of the engine can consume without a second conversion. The
	// 512 KiB-1 MiB fill is 0.1% of the 640 MiB the kernel reads.
	const y = np.zeros([rows], { dtype: dtype === "f16" ? DType.Float16 : DType.Float32 });
	const yBuf = y.ref.gpuBufferSync();

	const bindGroup = device.createBindGroup({
		layout: compiled.layout,
		entries: [
			{ binding: 0, resource: { buffer: wBuf } },
			{ binding: 1, resource: { buffer: xBuf } },
			{ binding: 2, resource: { buffer: yBuf } },
		],
	});
	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline(compiled.pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(compiled.workgroups);
	pass.end();
	// Submit before releasing the inputs, so every buffer is still alive when the
	// command buffer that references it is handed to the queue. The reverse order
	// destroys a buffer under a pending command buffer, which produces zeros with
	// no error at all (bench/scripts/submit-hazard.ts).
	device.queue.submit([encoder.finish()]);

	weight.dispose();
	x.dispose();

	return y;
}
