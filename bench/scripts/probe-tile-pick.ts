/**
 * Choose the tile size per shape, from measurement.
 *
 * A single 128 or 64 tile cannot be best everywhere: changing `TN` from 128 to
 * 64 moved prefill-1024 from 2472 to 3087 tok/s while dropping prefill-512 from
 * 2585 to 2415. The reason is occupancy against the dependency chain -- a 128
 * tile is 8x8 registers per thread (64 accumulators) and fills a 10-SM card with
 * fewer, fatter blocks, while a 64 tile is 4x4 and gets more blocks in flight.
 * Which one wins depends on M and N.
 *
 * This measures every distinct shape the model uses, at every S the suite
 * benchmarks, for both tile sizes, on GPU timestamps, and prints the winner. The
 * table it produces is what `pickTile` in matmul.ts encodes.
 */
import { defaultDevice, getWebGPUDevice, init, numpy as np } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import type { Gemma3 } from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";
import { matmul, matmulGpuMs } from "../../src/lib/gemma3/matmul";

const PEAK_AT_1506 = 3855;
const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const round = (x: number, n = 3) => +x.toFixed(n);

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model: Gemma3 = loadModelF32(safetensors.parse(await res.arrayBuffer()));
	const layer = model.layers[0];

	const shapes = [
		{ name: "qProj", N: 1024, K: 640 },
		{ name: "kProj", N: 256, K: 640 },
		{ name: "vProj", N: 256, K: 640 },
		{ name: "oProj", N: 640, K: 1024 },
		{ name: "gateProj", N: 2048, K: 640 },
		{ name: "upProj", N: 2048, K: 640 },
		{ name: "downProj", N: 640, K: 2048 },
	];

	const mk = (r: number, c: number, f: number) =>
		np.array(Float32Array.from({ length: r * c }, (_, i) => Math.sin(i * f)), { dtype: np.float32 })
			.reshape([r, c]);

	const rows: Record<string, unknown>[] = [];
	for (const S of [128, 256, 512, 1024]) {
		for (const sh of shapes) {
			// Weight in the model's real [N, K] row-major layout.
			const w = mk(sh.N, sh.K, 0.0014);
			const x = mk(S, sh.K, 0.0015);
			const flops = 2 * S * sh.N * sh.K;

			const times: Record<string, number> = {};
			for (const tile of [128, 64]) {
				(globalThis as { __matmulTile?: number }).__matmulTile = tile;
				const t: number[] = [];
				for (let i = 0; i < 7; i++) {
					const r = matmul(w.ref, x.ref, "out,in", { timestamps: true })!;
					t.push(await matmulGpuMs(r));
					r.dispose();
				}
				times[tile] = round(median(t));
			}
			(globalThis as { __matmulTile?: number }).__matmulTile = undefined;

			const best = times[128] <= times[64] ? 128 : 64;
			const gf = (t: number) => round(flops / 1e6 / t, 0);
			if (gf(Math.min(times[128], times[64])) > PEAK_AT_1506) {
				throw new Error(`${S} ${sh.name} above peak`);
			}
			rows.push({
				S,
				name: sh.name,
				M: S,
				N: sh.N,
				K: sh.K,
				t128: times[128],
				gf128: gf(times[128]),
				t64: times[64],
				gf64: gf(times[64]),
				best,
				gain: round(Math.max(times[128], times[64]) / Math.min(times[128], times[64]), 2),
			});
		}
	}
	// Totals per S, for the two "always one tile" policies and the mixture.
	const perS: Record<string, unknown>[] = [];
	for (const S of [128, 256, 512, 1024]) {
		const rs = rows.filter((r) => r.S === S);
		const sum = (k: string) => round(rs.reduce((a, r) => a + (r[k] as number), 0), 2);
		perS.push({
			S,
			all128: sum("t128"),
			all64: sum("t64"),
			best: round(rs.reduce((a, r) => a + Math.min(r.t128 as number, r.t64 as number), 0), 2),
			picks: rs.map((r) => `${r.name}:${r.best}`).join(" "),
		});
	}
	return { peakGflops: PEAK_AT_1506, perS, rows };
}
