/**
 * A tiled shared-memory SGEMM, as a candidate replacement for `np.dot(x, w.T)`.
 *
 * Why: jax-js lowers `np.dot(x, weight.transpose())` to one thread per output
 * element with an inner loop that reads the weight through a *view* with strides
 * (1, in) -- the same strided access that cost the decode vocabulary projection
 * 3.8x. Prefill runs 640-2048-long reductions this way, so the kernel issues a
 * separate gather per multiply-add. Measured in-model on a GTX 1060 (f32,
 * S=1024): 302 GFLOPS over the six projections, against the card's ~3.9 TFLOPS
 * at its measured 1506 MHz -- about 8%.
 *
 * This kernel loads 128x16 tiles of both operands into shared memory (contiguous
 * along the reduction axis) and keeps 8x8 outputs per thread in *registers*, so
 * one weight element serves 128 output rows instead of one.
 *
 * Two implementation notes that are easy to get wrong:
 *
 *  - Accumulators must be individual `f32` variables, not an `array<f32,64>`.
 *    A WGSL local array is memory, and a dynamically indexed one is spilled to
 *    local memory (i.e. global DRAM). Declaring the array made this kernel run
 *    at 126 GFLOPS, *slower* than the code it replaces. jax-js's own codegen
 *    emits `var acc0: f32` for the same reason.
 *  - jax-js's timestamp tracing only instruments kernels it submits itself, so
 *    a hand-written dispatch is invisible to `profiler.startTrace()`. The
 *    probe times this kernel with its own query set, which is the only way to
 *    get a GPU-time number that host contention cannot inflate.
 *
 * Correctness is not assumed: callers cross-check against `np.dot`, and
 * `npm run bench:check` runs the whole model through it.
 */
import { DType, getWebGPUDevice, numpy as np } from "@jax-js/jax";

const TN = 128; // tile side, output elements
const TK = 16; // reduction chunk per stage
const DIM = 16; // threads per tile side
const REG = 8; // outputs per thread per side

function fmaBlock(): string {
	let out = "";
	for (let i = 0; i < REG; i++) {
		for (let j = 0; j < REG; j++) {
			out += `      acc${i}${j} = acc${i}${j} + av${i} * bv${j};\n`;
		}
	}
	return out;
}

function declAccs(init: string): string {
	let out = "";
	for (let i = 0; i < REG; i++) for (let j = 0; j < REG; j++) out += `  var acc${i}${j}: f32 = ${init};\n`;
	return out;
}

function loadAV(): string {
	let out = "";
	for (let i = 0; i < REG; i++) out += `    av${i} = As[(ar + ${i}u) * ${TK + 1}u + kk];\n`;
	return out;
}

function loadBV(): string {
	let out = "";
	for (let i = 0; i < REG; i++) out += `    bv${i} = Ws[kk * ${TN + 1}u + (ac + ${i}u)];\n`;
	return out;
}

function storeOut(M: number, N: number): string {
	let out = "";
	for (let i = 0; i < REG; i++) {
		for (let j = 0; j < REG; j++) {
			out +=
				`  { let m: u32 = m0 + ar + ${i}u; let n: u32 = n0 + ac + ${j}u;\n` +
				`    if ((m < M) && (n < N)) { c[m * N + n] = acc${i}${j}; } }\n`;
		}
	}
	return out;
}

function source(M: number, N: number, K: number): string {
	return `
const TN: u32 = ${TN}u;
const TK: u32 = ${TK}u;
const DIM: u32 = ${DIM}u;
const REG: u32 = ${REG}u;
const M: u32 = ${M}u;
const N: u32 = ${N}u;
const K: u32 = ${K}u;

@group(0) @binding(0) var<storage, read> a: array<f32>;        // [M,K]
@group(0) @binding(1) var<storage, read> w: array<f32>;        // [N,K] row-major
@group(0) @binding(2) var<storage, read_write> c: array<f32>;  // [M,N]

// +1 element of padding on the inner axis: without it, a column-wise read of a
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

${declAccs("0.0")}
  for (var k0: u32 = 0u; k0 < K; k0 = k0 + TK) {
    // A tile: ${TN}x${TK}, contiguous along k -- the reduction axis.
    for (var i: u32 = li; i < ${TN * TK}u; i = i + 256u) {
      let r: u32 = i / TK;
      let ck: u32 = i % TK;
      let m: u32 = m0 + r;
      let kk: u32 = k0 + ck;
      As[r * ${TK + 1}u + ck] = select(0.0, a[m * K + kk], (m < M) && (kk < K));
    }
    // W tile: ${TK} k-offsets x ${TN} rows, each row contiguous along k.
    for (var i: u32 = li; i < ${TK * TN}u; i = i + 256u) {
      let rk: u32 = i / TN;
      let cn: u32 = i % TN;
      let n: u32 = n0 + cn;
      let kk: u32 = k0 + rk;
      Ws[rk * ${TN + 1}u + cn] = select(0.0, w[n * K + kk], (n < N) && (kk < K));
    }
    workgroupBarrier();

    for (var kk: u32 = 0u; kk < TK; kk = kk + 1u) {
      var av0: f32; var av1: f32; var av2: f32; var av3: f32;
      var av4: f32; var av5: f32; var av6: f32; var av7: f32;
      var bv0: f32; var bv1: f32; var bv2: f32; var bv3: f32;
      var bv4: f32; var bv5: f32; var bv6: f32; var bv7: f32;
${loadAV()}${loadBV()}${fmaBlock()}    }
    workgroupBarrier();
  }

${storeOut(M, N)}}
`;
}

type Compiled = { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout; gx: number; gy: number };
const cache = new WeakMap<GPUDevice, Map<string, Compiled>>();

function compileGemm(device: GPUDevice, M: number, N: number, K: number): Compiled {
	let byKey = cache.get(device);
	if (!byKey) {
		byKey = new Map();
		cache.set(device, byKey);
	}
	const key = `${M}x${N}x${K}`;
	const hit = byKey.get(key);
	if (hit) return hit;

	const module = device.createShaderModule({ code: source(M, N, K) });
	const pipeline = device.createComputePipeline({
		layout: "auto",
		compute: { module, entryPoint: "main" },
	});
	// A silently invalid shader writes nothing, and zeros are a plausible-looking
	// (wrong) answer. Surface it so the caller can fall back to np.dot.
	device.pushErrorScope("validation");
	const compiled: Compiled = {
		pipeline,
		layout: pipeline.getBindGroupLayout(0),
		gx: Math.ceil(N / TN),
		gy: Math.ceil(M / TN),
	};
	void device.popErrorScope().then((e) => {
		if (e) console.error(`[tiled-gemm] compile failed: ${e.message}`);
	});
	byKey.set(key, compiled);
	return compiled;
}

function canTile(weight: np.Array, x: np.Array): boolean {
	if (weight.shape.length !== 2 || x.shape.length !== 2) return false;
	if (weight.dtype !== DType.Float32 || x.dtype !== DType.Float32) return false;
	const [rows, cols] = weight.shape;
	return cols === x.shape[1] && rows > 0 && x.shape[0] > 0;
}

export type TiledGemmOptions = {
	/** Wrap the dispatch in timestamp queries and return the GPU time in ms. */
	timestamps?: boolean;
};

/**
 * C[M,N] = A[M,K] * W[N,K]^T, with W read row-major and contiguous.
 * Returns `null` when this shape/dtype is not one the kernel handles.
 *
 * Ownership: consumes `x` and `w`, matching `runLinear`.
 */
export function tiledGemm(
	x: np.Array,
	w: np.Array,
	opts: TiledGemmOptions = {},
): { out: np.Array; gpuMs: number } | null {
	if (!canTile(w, x)) return null;
	const device = getWebGPUDevice();
	const [N, K] = w.shape;
	const M = x.shape[0];
	const g = compileGemm(device, M, N, K);

	const aBuf = x.ref.gpuBufferSync();
	const wBuf = w.ref.gpuBufferSync();
	const c = np.zeros([M, N], { dtype: DType.Float32 });
	const cBuf = c.ref.gpuBufferSync();

	if (aBuf.size < M * K * 4 || wBuf.size < N * K * 4 || cBuf.size < M * N * 4) {
		x.dispose();
		w.dispose();
		c.dispose();
		return null;
	}

	const bg = device.createBindGroup({
		layout: g.layout,
		entries: [
			{ binding: 0, resource: { buffer: aBuf } },
			{ binding: 1, resource: { buffer: wBuf } },
			{ binding: 2, resource: { buffer: cBuf } },
		],
	});

	let querySet: GPUQuerySet | null = null;
	let resolve: GPUBuffer | null = null;
	let dst: GPUBuffer | null = null;
	if (opts.timestamps) {
		querySet = device.createQuerySet({ type: "timestamp", count: 2 });
		resolve = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
		});
		dst = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
	}

	const enc = device.createCommandEncoder();
	const pass = enc.beginComputePass(
		querySet
			? {
					timestampWrites: {
						querySet,
						beginningOfPassWriteIndex: 0,
						endOfPassWriteIndex: 1,
					},
				}
			: undefined,
	);
	pass.setPipeline(g.pipeline);
	pass.setBindGroup(0, bg);
	pass.dispatchWorkgroups(g.gx, g.gy);
	pass.end();
	let gpuMs = 0;
	if (querySet && resolve && dst) {
		enc.resolveQuerySet(querySet, 0, 2, resolve, 0);
		enc.copyBufferToBuffer(resolve, 0, dst, 0, 16);
	}
	// Submit while the inputs are still alive: destroying a buffer under a
	// pending command buffer yields zeros with no error (bench/scripts/submit-hazard.ts).
	device.queue.submit([enc.finish()]);

	x.dispose();
	w.dispose();

	if (querySet && resolve && dst) {
		const stamps = querySet;
		const staging = dst;
		// Stash the read on the returned array so the caller can await it without
		// changing the ownership contract.
		(c as unknown as { __gpuMs: Promise<number> }).__gpuMs = staging
			.mapAsync(GPUMapMode.READ)
			.then(() => {
				const t = new BigInt64Array(staging.getMappedRange());
				gpuMs = Number(t[1] - t[0]) / 1e6;
				staging.unmap();
				return gpuMs;
			})
			.finally(() => {
				stamps.destroy();
				staging.destroy();
				resolve.destroy();
			});
	}
	return { out: c, gpuMs };
}

/** Read back the GPU time stashed by a `timestamps: true` call. */
export async function tiledGpuMs(out: np.Array): Promise<number> {
	const p = (out as unknown as { __gpuMs?: Promise<number> }).__gpuMs;
	if (!p) throw new Error("no timestamp was requested for this output");
	return p;
}
