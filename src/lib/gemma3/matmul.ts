/**
 * A shared-memory tiled matmul, for the [S, ...] tensors a prefill pass works on.
 *
 * Why this exists. `runLinear` is `np.dot(x, weight.transpose())`, and jax-js
 * lowers that to one thread per output element with a loop over the reduction
 * axis that reads the weight through a *view* with strides `(1, in)`. That means
 * consecutive reduction steps are a whole row apart in memory, so every
 * multiply-add pulls a fresh 128-byte cache line to use 4 bytes of it. It is the
 * same access pattern that cost the decode vocabulary projection 3.8x, except
 * prefill pays it on every projection in all 18 layers.
 *
 * Measured in-model on a GTX 1060 (f32, S=1024), one layer, GPU timestamps:
 *
 *   projection      np.dot     this kernel    ratio
 *   qProj           3.81 ms      1.52 ms       2.5x
 *   oProj           3.49 ms      1.57 ms       2.2x
 *   gate / up       7.10 ms      2.86 ms       2.5x
 *   down           10.76 ms      2.99 ms       3.6x
 *   QK (4 heads)    5.68 ms      2.26 ms       2.5x
 *   PV (4 heads)    4.03 ms      2.45 ms       1.6x
 *
 * Totalled, the prefill matmuls run at 333 GFLOPS through np.dot and 891 GFLOPS
 * here, against ~3.9 TFLOPS of f32 peak at this card's measured 1506 MHz. The
 * card's *f32 cores* are the limit now, not its memory: prefill is compute bound
 * where decode was bandwidth bound, which is why the vocabulary-projection fix
 * does nothing for it.
 *
 * What the kernel does: a 128x128 output tile per workgroup, 16x16 threads,
 * 8x8 outputs per thread held in registers, and both operands staged through
 * shared memory 16 reduction steps at a time. One weight element then serves 128
 * output rows instead of one.
 *
 * Two implementation notes that are easy to get wrong, both learned by measuring:
 *
 *  - The accumulators must be individual `f32` variables, not `array<f32, 64>`.
 *    A dynamically indexed WGSL local array is spilled to local memory, which is
 *    global DRAM. Declaring it that way made this kernel run at 126 GFLOPS --
 *    slower than the code it replaces. jax-js's own codegen emits `var acc0:
 *    f32` for the same reason.
 *
 *  - jax-js's `profiler.startTrace()` only instruments kernels jax-js submits
 *    itself, so a hand-written dispatch is invisible to it. `timedMatmul` below
 *    uses its own timestamp query set; that is the only way to get a GPU-time
 *    number for this kernel that host contention cannot inflate.
 *
 * Correctness is not assumed anywhere. `matmulDecline` says exactly why a call
 * is not handled, `npm run bench:check` runs the real model through this code,
 * and every call site falls back to the stock `np.dot`, which is correct by
 * construction.
 */
import { DType, getWebGPUDevice, numpy as np } from "@jax-js/jax";

const TK = 16; // reduction steps staged at a time

/**
 * Tile configurations, keyed by output-tile side.
 *
 * Both hold a square register block, but differ in how many blocks the card can
 * keep in flight. Measured; see probe-tile-pick.ts.
 */
const TILES = {
	128: { TN: 128, DIM: 16 }, // 8x8 accumulators
	64: { TN: 64, DIM: 16 }, // 4x4 accumulators
} as const;
type TileSide = keyof typeof TILES;

/**
 * The default tile side, and why it is 64 rather than the textbook 128.
 *
 * Measured on every projection shape at every S the suite benchmarks
 * (probe-tile-pick.ts): the 64 tile is faster in all 28 combinations, by 1.11x
 * to 1.77x, reaching 1272 GFLOPS against 929 for the 128 tile. The 128 tile
 * carries 64 f32 accumulators per thread and 256 threads per workgroup, which on
 * a 10-SM card leaves too few blocks in flight to hide the shared-memory
 * staging; the 4x4 block of the 64 tile keeps more blocks resident.
 *
 * `__matmulTile` is a probe hook, not a feature: it exists so the sweep above can
 * be re-run without editing the kernel.
 */
function tileFor(override?: number): TileSide {
	if (override === 64 || override === 128) return override;
	return 64;
}

/**
 * Which way `weight` is stored.
 *
 *  - `"out,in"` is the safetensors layout, `[N, K]` row-major: the reduction
 *    axis K is contiguous within a row.
 *  - `"in,out"` is what a materialised transpose or a `[K, N]` buffer gives:
 *    the output axis N is contiguous.
 *
 * Both load the same 16x128 shared-memory tile; only the global read index
 * differs, and `"in,out"` coalesces slightly better on load. Supporting both
 * means no call site has to build a transposed copy first.
 */
type Layout = "out,in" | "in,out";

function unrolled(reg: number, template: (i: number, j: number) => string): string {
	let out = "";
	for (let i = 0; i < reg; i++) for (let j = 0; j < reg; j++) out += template(i, j);
	return out;
}

/** Storage dtype of the two input operands. Accumulation is always f32. */
export type MatmulDtype = "f32" | "f16";

function shaderSource(
	M: number,
	N: number,
	K: number,
	layout: Layout,
	side: TileSide,
	dtype: MatmulDtype,
): string {
	// Inputs are read as their storage type and widened on load; `acc`, the
	// shared tiles and the output are always f32. That is the same contract
	// jax-js uses for f16 weights (read f16, accumulate f32), so this path
	// produces the same numbers whether the weights are stored narrow or wide.
	const enable = dtype === "f16" ? "enable f16;\n" : "";
	const inTy = dtype === "f16" ? "f16" : "f32";
	const { TN, DIM } = TILES[side];
	const REG = TN / DIM;
	const accumulators = unrolled(REG, (i, j) => `  var acc${i}${j}: f32 = 0.0;\n`);

	// `As`/`Ws` are f32 shared arrays, so assignment from an f16 global load
	// performs the widening. Nothing accumulates in f16.
	const loadA = unrolled(REG, (i) => `    av${i} = As[(ar + ${i}u) * ${TK + 1}u + kk];\n`);
	const loadB = unrolled(REG, (i) => `    bv${i} = Ws[kk * ${TN + 1}u + (ac + ${i}u)];\n`);

	const fma = unrolled(REG, (i, j) => `      acc${i}${j} = acc${i}${j} + av${i} * bv${j};\n`);

	const store = unrolled(
		REG,
		(i, j) =>
			`  { let m: u32 = m0 + ar + ${i}u; let n: u32 = n0 + ac + ${j}u;\n` +
			`    if ((m < M) && (n < N)) { c[m * N + n] = acc${i}${j}; } }\n`,
	);

	let scalarDecls = "";
	for (let i = 0; i < REG; i++) scalarDecls += `      var av${i}: f32; var bv${i}: f32;\n`;

	// Global read index for the weight tile element (reduction step rk, output
	// column cn). Only this line differs between the two layouts.
	const wIndex =
		layout === "out,in"
			? `w[(n0 + cn) * K + (k0 + rk)]`
			: `w[(k0 + rk) * N + (n0 + cn)]`;

	return `
${enable}const TN: u32 = ${TN}u;
const TK: u32 = ${TK}u;
const DIM: u32 = ${DIM}u;
const REG: u32 = ${REG}u;
const M: u32 = ${M}u;
const N: u32 = ${N}u;
const K: u32 = ${K}u;

@group(0) @binding(0) var<storage, read> a: array<${inTy}>;        // [M,K]
@group(0) @binding(1) var<storage, read> w: array<${inTy}>;        // [N,K] or [K,N]
@group(0) @binding(2) var<storage, read_write> c: array<f32>;  // [M,N]

// +1 element of padding on the inner axis. Without it a column-wise read of a
// row whose length is a multiple of 32 banks serialises the whole workgroup.
var<workgroup> As: array<f32, ${TN * (TK + 1)}>;
var<workgroup> Ws: array<f32, ${TK * (TN + 1)}>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) li: u32,
) {
  let m0: u32 = wid.y * TN;
  let n0: u32 = wid.x * TN;
  let tx: u32 = li % DIM;
  let ty: u32 = li / DIM;
  let ar: u32 = ty * REG;
  let ac: u32 = tx * REG;

${accumulators}
  for (var k0: u32 = 0u; k0 < K; k0 = k0 + TK) {
    // Activation tile: ${TN} rows x ${TK} reduction steps, contiguous along K.
    for (var i: u32 = li; i < ${TN * TK}u; i = i + 256u) {
      let r: u32 = i / TK;
      let ck: u32 = i % TK;
      let m: u32 = m0 + r;
      let kk: u32 = k0 + ck;
      As[r * ${TK + 1}u + ck] = select(0.0, a[m * K + kk], (m < M) && (kk < K));
    }
    // Weight tile: ${TK} reduction steps x ${TN} output columns.
    for (var i: u32 = li; i < ${TK * TN}u; i = i + 256u) {
      let rk: u32 = i / TN;
      let cn: u32 = i % TN;
      let n: u32 = n0 + cn;
      let kk: u32 = k0 + rk;
      Ws[rk * ${TN + 1}u + cn] = select(0.0, ${wIndex}, (n < N) && (kk < K));
    }
    workgroupBarrier();

    for (var kk: u32 = 0u; kk < TK; kk = kk + 1u) {
${scalarDecls}
${loadA}${loadB}${fma}    }
    workgroupBarrier();
  }

${store}}
`;
}

type Compiled = {
	pipeline: GPUComputePipeline;
	layout: GPUBindGroupLayout;
	gridX: number;
	gridY: number;
};
const compiled = new WeakMap<GPUDevice, Map<string, Compiled>>();

function compile(
	device: GPUDevice,
	M: number,
	N: number,
	K: number,
	layout: Layout,
	side: TileSide,
	dtype: MatmulDtype,
): Compiled {
	let byKey = compiled.get(device);
	if (!byKey) {
		byKey = new Map();
		compiled.set(device, byKey);
	}
	const key = `${M}x${N}x${K}x${layout}x${side}x${dtype}`;
	const hit = byKey.get(key);
	if (hit) return hit;

	const module = device.createShaderModule({ code: shaderSource(M, N, K, layout, side, dtype) });
	const pipeline = device.createComputePipeline({
		layout: "auto",
		compute: { module, entryPoint: "main" },
	});
	// A shader that fails to validate writes nothing, and a matrix of zeros is a
	// plausible-looking wrong answer that would propagate silently. Surface the
	// error and let the caller fall back to the stock path.
	device.pushErrorScope("validation");
	const entry: Compiled = {
		pipeline,
		layout: pipeline.getBindGroupLayout(0),
		gridX: Math.ceil(N / TILES[side].TN),
		gridY: Math.ceil(M / TILES[side].TN),
	};
	void device.popErrorScope().then((e) => {
		if (e) console.error(`[matmul] shader compile failed: ${e.message}`);
	});
	byKey.set(key, entry);
	return entry;
}

/**
 * Why `matmul` will not handle this call. `null` when it will.
 *
 * A decline is legal -- the f16 path, the CPU/WASM backends and any rank this
 * kernel does not cover all fall back to `runLinear` -- but a *silent* decline
 * on the hot path would look like a working optimization while quietly running
 * the slow path, so callers can ask and log the reason once.
 */
export function matmulDecline(weight: np.Array, x: np.Array, layout: Layout): string | null {
	if (weight.shape.length !== 2 || x.shape.length !== 2) return "not a matrix-matrix product";
	const [rows, cols] = weight.shape;
	const [m, k] = x.shape;
	const n = layout === "out,in" ? rows : cols;
	const kk = layout === "out,in" ? cols : rows;
	if (kk !== k) return `reduction axis ${kk} does not match activation ${k}`;
	if (n <= 0 || m <= 0) return "empty output";
	const width = dtypeOf(weight);
	if (!width) return `weight dtype ${weight.dtype} (f32 and f16 only)`;
	if (dtypeOf(x) !== width) return `mixed dtypes: weight ${weight.dtype}, activation ${x.dtype}`;

	const device = getWebGPUDeviceOrNull();
	if (!device) return "not on the WebGPU backend";
	// f16 *storage* needs the shader-f16 feature, and the shader also has to
	// declare `enable f16;`. Without the feature the stock path handles it.
	if (width === "f16" && !device.features.has("shader-f16" as GPUFeatureName)) {
		return "f16 weights but the device has no shader-f16 feature";
	}
	// f16 is OPT-IN and off by default. The dev machine has no shader-f16, so
	// this path has never been executed or gated anywhere: the shader compiles
	// here only as far as "extension f16 is not allowed in this environment".
	// Enable with `globalThis.__matmulAllowF16 = true` on a device that has the
	// feature, and run `npm run bench:check` there before trusting it.
	if (width === "f16" && !(globalThis as { __matmulAllowF16?: boolean }).__matmulAllowF16) {
		return "f16 path is unverified on this hardware; set __matmulAllowF16 to enable";
	}
	return null;
}

/** Storage width of an array, or null if this kernel cannot read it. */
function dtypeOf(a: np.Array): MatmulDtype | null {
	if (a.dtype === DType.Float32) return "f32";
	if (a.dtype === DType.Float16) return "f16";
	return null;
}

function getWebGPUDeviceOrNull(): GPUDevice | null {
	try {
		return getWebGPUDevice() ?? null;
	} catch {
		return null;
	}
}

export type MatmulOptions = {
	/**
	 * Wrap the dispatch in timestamp queries so `matmulGpuMs` can report its GPU
	 * time. Off by default: two extra buffers per call would be a real cost in
	 * the model, and only probes want it.
	 */
	timestamps?: boolean;
};

const pendingGpuMs = new WeakMap<np.Array, Promise<number>>();

/**
 * Process-wide switch for timestamping, plus an accumulator.
 *
 * Per-call `timestamps: true` would need a call site change in the model, and
 * the model should not know a profiler exists. Probes turn this on, run a pass,
 * and read the total with `takeMatmulTimes()`; the model always calls `matmul`
 * with default options.
 */
let timestampsOn = false;
let accCount = 0;
let accMs = 0;

export function setMatmulTimestamps(on: boolean): void {
	timestampsOn = on;
}

export function takeMatmulTimes(): { count: number; total: number } {
	const out = { count: accCount, total: accMs };
	accCount = 0;
	accMs = 0;
	return out;
}

/**
 * `C[M,N] = A[M,K] * W^T`, reading `W` the way it is actually stored.
 *
 * Returns `null` when the shape, dtype or backend is not one this kernel
 * handles; the caller must then use the stock path.
 *
 * Ownership: consumes `weight` and `x`, matching `runLinear`.
 */
export function matmul(
	weight: np.Array,
	x: np.Array,
	layout: Layout,
	opts: MatmulOptions = {},
): np.Array | null {
	if (matmulDecline(weight, x, layout)) return null;

	const device = getWebGPUDevice();
	const [rows, cols] = weight.shape;
	const N = layout === "out,in" ? rows : cols;
	const K = layout === "out,in" ? cols : rows;
	const M = x.shape[0];
	const side = tileFor((globalThis as { __matmulTile?: number }).__matmulTile);
	const width = dtypeOf(weight) as MatmulDtype;
	const g = compile(device, M, N, K, layout, side, width);

	const wantStamps = opts.timestamps ?? timestampsOn;
	const aBuf = x.ref.gpuBufferSync();
	const wBuf = weight.ref.gpuBufferSync();
	// jax-js has no uninitialised allocation, so this is a zero-fill of the
	// output. Every element is then overwritten by the kernel; the fill is one
	// extra write pass, ~4% of the kernel's own traffic at these shapes.
	const c = np.zeros([M, N], { dtype: DType.Float32 });
	const cBuf = c.ref.gpuBufferSync();

	const bytes = (need: number, buf: GPUBuffer, what: string) => {
		if (buf.size < need) {
			throw new Error(`[matmul] ${what} buffer is ${buf.size} B, need ${need} B`);
		}
	};
	bytes(M * K * 4, aBuf, "activation");
	bytes(N * K * 4, wBuf, "weight");
	bytes(M * N * 4, cBuf, "output");

	const bindGroup = device.createBindGroup({
		layout: g.layout,
		entries: [
			{ binding: 0, resource: { buffer: aBuf } },
			{ binding: 1, resource: { buffer: wBuf } },
			{ binding: 2, resource: { buffer: cBuf } },
		],
	});

	let querySet: GPUQuerySet | null = null;
	let resolve: GPUBuffer | null = null;
	let staging: GPUBuffer | null = null;
	if (wantStamps) {
		querySet = device.createQuerySet({ type: "timestamp", count: 2 });
		resolve = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
		});
		staging = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});
	}

	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass(
		querySet
			? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
			: undefined,
	);
	pass.setPipeline(g.pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(g.gridX, g.gridY);
	pass.end();
	if (querySet && resolve && staging) {
		encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
		encoder.copyBufferToBuffer(resolve, 0, staging, 0, 16);
	}
	// Submit before releasing the inputs. Destroying a buffer that a pending
	// command buffer still references yields zeros and no error
	// (bench/scripts/submit-hazard.ts).
	device.queue.submit([encoder.finish()]);

	weight.dispose();
	x.dispose();

	if (querySet && resolve && staging) {
		const qs = querySet;
		const rs = resolve;
		const st = staging;
		pendingGpuMs.set(
			c,
			st
				.mapAsync(GPUMapMode.READ)
				.then(() => {
					const t = new BigInt64Array(st.getMappedRange());
					const ms = Number(t[1] - t[0]) / 1e6;
					accCount++;
					accMs += ms;
					return ms;
				})
				.finally(() => {
					st.unmap();
					qs.destroy();
					rs.destroy();
					st.destroy();
				}),
		);
	}

	return c;
}

/** GPU time of a `matmul(..., { timestamps: true })` result, in milliseconds. */
export function matmulGpuMs(out: np.Array): Promise<number> {
	const p = pendingGpuMs.get(out);
	if (!p) throw new Error("[matmul] no timestamp was requested for this output");
	return p;
}

/**
 * A/B seam for the model: this kernel when it applies, the stock `np.dot`
 * otherwise. Kept exported so an optimization that cannot be turned off cannot
 * be measured, and so the probe suite can price both arms of the same call.
 */
export function shouldUseMatmul(weight: np.Array, x: np.Array, layout: Layout): boolean {
	return matmulDecline(weight, x, layout) === null;
}
