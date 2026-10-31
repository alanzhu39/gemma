/**
 * Does `weight.transpose()` cost anything in the projection?
 *
 * `runLinear` computes np.dot(x, weight.transpose()). For the tied embedding the
 * weight is [262144, 640] row-major, so W^T is a strided view. If the kernel
 * reads it column-wise, each lane touches a different row 2560 B apart.
 *
 * Compares against a physically transposed copy made once, so the matvec reads
 * contiguously. This is the difference between "strided view" and "no transpose
 * at all" with everything else held constant.
 *
 * Three warmups everywhere: one is not enough for a 640 MiB tensor to be
 * resident, and that mistake previously made a bare dot look 2.6x faster than it
 * was.
 */
import { blockUntilReady, defaultDevice, init, numpy as np } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { loadModelF32 } from "../f32-model";

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model = loadModelF32(safetensors.parse(await res.arrayBuffer()));
	const W = model.tokenEmbed.ref; // [262144, 640]
	const bytes = W.shape[0] * 640 * 4;
	const x = np.ones([640]);

	// A real transposed copy: [640, 262144] contiguous. Costs 1.28 GiB of traffic
	// once, so only worth it if the matvec then runs much faster.
	// A transpose in jax-js is a view, so force a real buffer with an
	// elementwise op. This costs 1.28 GiB of traffic once.
	const t0 = performance.now();
	const WTbuf = W.ref.transpose().mul(1);
	await blockUntilReady(WTbuf);
	const transposeMs = performance.now() - t0;

	const time = async (label: string, fn: () => np.Array) => {
		for (let i = 0; i < 3; i++) {
			const w = fn();
			await w.data();
		}
		const t1 = performance.now();
		for (let i = 0; i < 5; i++) {
			const r = fn();
			await r.data();
		}
		const ms = (performance.now() - t1) / 5;
		const gbps = bytes / 1e9 / (ms / 1000);
		if (gbps > 192) throw new Error(`${label}: elided`);
		return { variant: label, ms: +ms.toFixed(2), gbps: +gbps.toFixed(1) };
	};

	const out: Record<string, number | string>[] = [];
	out.push({ variant: "one-time transpose cost", ms: +transposeMs.toFixed(1), gbps: 0 });
	out.push(await time("strided W^T (current)", () => np.dot(x.ref, W.ref.transpose())));
	out.push(await time("contiguous W^T", () => np.dot(x.ref, WTbuf.ref)));
	return out;
}
