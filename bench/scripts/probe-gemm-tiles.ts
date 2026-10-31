/**
 * Tile-shape sweep for the tiled matmul.
 *
 * The kernel currently reaches 891 GFLOPS, 23% of the card's 3855 GFLOPS at
 * 1506 MHz. A 128x128 tile with 8x8 registers is the textbook default, but the
 * register pressure of 64 f32 accumulators plus 16 operand values limits
 * occupancy on a Pascal part with 64K registers per SM, and the shape that wins
 * on a 10-SM card is not obvious from first principles.
 *
 * This sweeps (tile side, register block) on the two shapes that dominate the
 * pass and reports GPU-timestamp time, so the choice is measured. Every variant
 * is checked against np.dot: a tile that computes the wrong thing cannot win.
 */
import { defaultDevice, getWebGPUDevice, init, numpy as np } from "@jax-js/jax";

const PEAK_AT_1506 = 3855;
const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const round = (x: number, n = 3) => +x.toFixed(n);

type Variant = { TN: number; DIM: number; TK: number };

function source(M: number, N: number, K: number, v: Variant): string {
	const REG = v.TN / v.DIM;
	const unroll = (template: (i: number, j: number) => string) => {
		let out = "";
		for (let i = 0; i < REG; i++) for (let j = 0; j < REG; j++) out += template(i, j);
		return out;
	};
	let scalarDecls = "";
	for (let i = 0; i < REG; i++) scalarDecls += `      var av${i}: f32; var bv${i}: f32;\n`;

	return `
const TN: u32 = ${v.TN}u;
const TK: u32 = ${v.TK}u;
const DIM: u32 = ${v.DIM}u;
const REG: u32 = ${REG}u;
const M: u32 = ${M}u;
const N: u32 = ${N}u;
const K: u32 = ${K}u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

var<workgroup> As: array<f32, ${v.TN * (v.TK + 1)}>;
var<workgroup> Ws: array<f32, ${v.TK * (v.TN + 1)}>;

@compute @workgroup_size(${v.DIM * v.DIM})
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

${unroll((i, j) => `  var acc${i}${j}: f32 = 0.0;\n`)}
  for (var k0: u32 = 0u; k0 < K; k0 = k0 + TK) {
    for (var i: u32 = li; i < ${v.TN * v.TK}u; i = i + ${v.DIM * v.DIM}u) {
      let r: u32 = i / TK;
      let ck: u32 = i % TK;
      let m: u32 = m0 + r;
      let kk: u32 = k0 + ck;
      As[r * ${v.TK + 1}u + ck] = select(0.0, a[m * K + kk], (m < M) && (kk < K));
    }
    for (var i: u32 = li; i < ${v.TK * v.TN}u; i = i + ${v.DIM * v.DIM}u) {
      let rk: u32 = i / TN;
      let cn: u32 = i % TN;
      let n: u32 = n0 + cn;
      let kk: u32 = k0 + rk;
      Ws[rk * ${v.TN + 1}u + cn] = select(0.0, w[n * K + kk], (n < N) && (kk < K));
    }
    workgroupBarrier();

    for (var kk: u32 = 0u; kk < TK; kk = kk + 1u) {
${scalarDecls}
${unroll((i) => `    av${i} = As[(ar + ${i}u) * ${v.TK + 1}u + kk];\n`)}
${unroll((i) => `    bv${i} = Ws[kk * ${v.TN + 1}u + (ac + ${i}u)];\n`)}
${unroll((i, j) => `      acc${i}${j} = acc${i}${j} + av${i} * bv${j};\n`)}
    }
    workgroupBarrier();
  }

${unroll(
	(i, j) =>
		`  { let m: u32 = m0 + ar + ${i}u; let n: u32 = n0 + ac + ${j}u;\n` +
		`    if ((m < M) && (n < N)) { c[m * N + n] = acc${i}${j}; } }\n`,
)}
}
`;
}

const cache = new WeakMap<GPUDevice, Map<string, { p: GPUComputePipeline; l: GPUBindGroupLayout }>>();

function compile(dev: GPUDevice, M: number, N: number, K: number, v: Variant) {
	let m = cache.get(dev);
	if (!m) {
		m = new Map();
		cache.set(dev, m);
	}
	const key = `${M}x${N}x${K}:${v.TN}/${v.DIM}/${v.TK}`;
	let hit = m.get(key);
	if (hit) return hit;
	const module = dev.createShaderModule({ code: source(M, N, K, v) });
	const p = dev.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
	hit = { p, l: p.getBindGroupLayout(0) };
	m.set(key, hit);
	return hit;
}

async function run(dev: GPUDevice, x: np.Array, w: np.Array, v: Variant) {
	const [N, K] = w.shape;
	const M = x.shape[0];
	const g = compile(dev, M, N, K, v);
	const aBuf = x.ref.gpuBufferSync();
	const wBuf = w.ref.gpuBufferSync();
	const c = np.zeros([M, N], { dtype: np.float32 });
	const cBuf = c.ref.gpuBufferSync();
	const bg = dev.createBindGroup({
		layout: g.l,
		entries: [
			{ binding: 0, resource: { buffer: aBuf } },
			{ binding: 1, resource: { buffer: wBuf } },
			{ binding: 2, resource: { buffer: cBuf } },
		],
	});
	const qs = dev.createQuerySet({ type: "timestamp", count: 2 });
	const res = dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
	const dst = dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
	const enc = dev.createCommandEncoder();
	const pass = enc.beginComputePass({
		timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
	});
	pass.setPipeline(g.p);
	pass.setBindGroup(0, bg);
	pass.dispatchWorkgroups(Math.ceil(N / v.TN), Math.ceil(M / v.TN));
	pass.end();
	enc.resolveQuerySet(qs, 0, 2, res, 0);
	enc.copyBufferToBuffer(res, 0, dst, 0, 16);
	dev.queue.submit([enc.finish()]);
	await dst.mapAsync(GPUMapMode.READ);
	const t = new BigInt64Array(dst.getMappedRange());
	const ms = Number(t[1] - t[0]) / 1e6;
	dst.unmap();
	qs.destroy();
	res.destroy();
	dst.destroy();
	return { c, ms };
}

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const dev = getWebGPUDevice();

	const mk = (r: number, c: number, f: number) =>
		np.array(Float32Array.from({ length: r * c }, (_, i) => Math.sin(i * f)), { dtype: np.float32 })
			.reshape([r, c]);

	// The shapes that dominate: the MLP pair and the down projection.
	const cases = [
		{ name: "gate [1024,640]x[640,2048]", M: 1024, K: 640, N: 2048 },
		{ name: "down [1024,2048]x[2048,640]", M: 1024, K: 2048, N: 640 },
	];

	const variants: Variant[] = [
		{ TN: 128, DIM: 16, TK: 16 }, // current
		{ TN: 128, DIM: 16, TK: 8 },
		{ TN: 128, DIM: 16, TK: 32 },
		{ TN: 64, DIM: 16, TK: 16 }, // REG=4, lighter registers
		{ TN: 64, DIM: 8, TK: 16 }, // REG=8 on a 64 tile
		{ TN: 128, DIM: 32, TK: 16 }, // REG=4, 1024 threads
		{ TN: 256, DIM: 32, TK: 16 }, // REG=8, 1024 threads
		{ TN: 64, DIM: 16, TK: 32 },
	];

	const rows: Record<string, unknown>[] = [];
	for (const c of cases) {
		const x = mk(c.M, c.K, 0.0014);
		const w = mk(c.N, c.K, 0.0015);
		const want = (await np.dot(x.ref, w.ref.transpose()).data()) as Float32Array;
		let mag = 0;
		for (let i = 0; i < want.length; i++) mag = Math.max(mag, Math.abs(want[i]));

		for (const v of variants) {
			const reg = v.TN / v.DIM;
			if (reg * reg * 4 + 32 > 255) continue; // register budget guard
			try {
				// Warm.
				for (let i = 0; i < 2; i++) (await run(dev, x.ref, w.ref, v)).c.dispose();
				const times: number[] = [];
				let got: Float32Array | null = null;
				for (let i = 0; i < 6; i++) {
					const r = await run(dev, x.ref, w.ref, v);
					times.push(r.ms);
					if (i === 0) got = (await r.c.data()) as Float32Array;
					else r.c.dispose();
				}
				let maxAbs = 0;
				for (let i = 0; i < want.length; i++) maxAbs = Math.max(maxAbs, Math.abs(want[i] - got![i]));
				const flops = 2 * c.M * c.N * c.K;
				const ms = median(times);
				const gf = flops / 1e6 / ms;
				rows.push({
					case: c.name,
					tile: `${v.TN}/${v.DIM}/${v.TK}`,
					reg: `${reg}x${reg}`,
					threads: v.DIM * v.DIM,
					ms: round(ms),
					gflops: round(gf, 0),
					pctPeak: round((100 * gf) / PEAK_AT_1506, 1),
					err: +(maxAbs / mag).toExponential(1),
					abovePeak: gf > PEAK_AT_1506,
				});
			} catch (e) {
				rows.push({ case: c.name, tile: `${v.TN}/${v.DIM}/${v.TK}`, error: String((e as Error).message).slice(0, 120) });
			}
		}
	}
	return { peakGflops: PEAK_AT_1506, rows };
}
