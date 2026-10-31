/**
 * Validate the production `matmul` kernel against np.dot, shape by shape.
 *
 * Written because the correctness gate caught a wrong answer that the isolated
 * bench kernel got right: the model integration passes weight layouts and
 * reshapes the bench version did not, and a plausible-looking-but-wrong matmul
 * is the worst possible failure (it yields fluent garbage). This checks each
 * real shape, both layouts, and every S the model uses.
 */
import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { matmul, matmulDecline } from "../../src/lib/gemma3/matmul";

const round = (x: number, n = 6) => +x.toFixed(n);

function check(
	name: string,
	x: np.Array,
	w: np.Array,
	layout: "out,in" | "in,out",
): Record<string, unknown> {
	const ref = np.dot(x.ref, w.ref.transpose());
	const got = matmul(w.ref, x.ref, layout);
	if (!got) {
		return { name, layout, declined: matmulDecline(w.ref, x.ref, layout) };
	}
	// dataSync() consumes; nothing may dispose these afterwards.
	const a = ref.dataSync() as Float32Array;
	const b = got.dataSync() as Float32Array;
	let mag = 0;
	let maxAbs = 0;
	let worst = -1;
	for (let i = 0; i < a.length; i++) {
		const d = Math.abs(a[i] - b[i]);
		mag = Math.max(mag, Math.abs(a[i]));
		if (d > maxAbs) {
			maxAbs = d;
			worst = i;
		}
	}
	return {
		name,
		layout,
		shape: `[${x.shape}] x w[${w.shape}]`,
		relErr: round(maxAbs / Math.max(1e-9, mag)),
		worstIndex: worst,
		outLen: a.length,
	};
}

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");

	const mk = (r: number, c: number, f: number) =>
		np.array(Float32Array.from({ length: r * c }, (_, i) => Math.sin(i * f)), { dtype: np.float32 })
			.reshape([r, c]);

	const out: Record<string, unknown>[] = [];

	// The exact shapes, several S values.
	for (const S of [11, 23, 128, 512, 1024]) {
		for (const [name, N, K, f] of [
			["qProj", 1024, 640, 0.001],
			["kProj", 256, 640, 0.0011],
			["oProj", 640, 1024, 0.0013],
			["gateProj", 2048, 640, 0.0014],
			["downProj", 640, 2048, 0.0016],
		] as [string, number, number, number][]) {
			const x = mk(S, K, f);
			// Weight as it really is stored: [out, in] row-major.
			const w = mk(N, K, f + 0.0001);
			out.push(check(`S=${S} ${name}`, x, w, "out,in"));
		}
	}

	// Non-multiples of the 128 tile: M and N both ragged.
	for (const [S, N, K] of [
		[11, 7, 640],
		[130, 200, 640],
		[1, 1024, 640],
		[3, 5, 9],
	]) {
		const x = mk(S, K, 0.002);
		const w = mk(N, K, 0.0021);
		out.push(check(`ragged S=${S} N=${N} K=${K}`, x, w, "out,in"));
	}

	// The "in,out" layout, which the attention core uses for QK.
	{
		const S = 1024;
		const q = mk(S * 4, 256, 0.0009);
		const k = mk(1024, 256, 0.0011);
		out.push(check("QK via out,in", q, k, "out,in"));
	}

	const bad = out.filter((r) => typeof r.relErr === "number" && (r.relErr as number) > 2e-4);
	const declined = out.filter((r) => typeof r.declined === "string");
	return {
		summary: { total: out.length, bad: bad.length, declined: declined.length },
		bad,
		declined: declined.slice(0, 5),
		rows: out.filter((r) => typeof r.relErr === "number").slice(0, 40),
	};
}
