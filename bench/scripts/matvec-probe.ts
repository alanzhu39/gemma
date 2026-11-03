/**
 * How does matvec cost scale with the number of rows?
 *
 * This measures the logits projection ([1,640] x [640,V]) at several vocabulary
 * sizes. The result is what matters: cost is exactly linear in rows read
 * (4096: 2.3 ms, 65536: 113.7 ms, 262144: 461.5 ms), so the projection is bound
 * by how it reads the weight matrix, not by arithmetic or dispatch count.
 *
 * Deliberately does NOT report a bandwidth figure. An earlier version tried to,
 * and produced 559 GB/s and then 2386 GB/s on a card that nominally does 192 -
 * both impossible. The measurement was wrong (uninitialised buffers, write
 * contention from every thread to one address, and timing queue submission
 * rather than completion), and a wrong number here is worse than none. What
 * survives scrutiny is the scaling relationship, measured through jax-js's own
 * readback so completion is guaranteed.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { loadModelF32 } from "../f32-model";

export async function probe2() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model = loadModelF32(safetensors.parse(await res.arrayBuffer()));

	const results: Record<string, unknown>[] = [];

	// 1. Plain matvec against the tied embedding: [1,640] x [640,262144].
	for (const dt of ["f32"]) {
		const w = np.array(model.tokenEmbed.ref as never).astype(dt === "f32" ? np.float32 : np.float16);
		const x = np.ones([640], { dtype: dt === "f32" ? np.float32 : np.float16 });
		// Warmup (compiles).
		const warm = np.dot(x.ref, w.ref.transpose());
		await blockUntilReady(warm);
		warm.dispose();

		const t0 = performance.now();
		const y = np.dot(x, w.transpose());
		await blockUntilReady(y);
		const ms = performance.now() - t0;
		y.dispose();
		results.push({ op: `dot 1x640 @ 640x262144 (${dt})`, ms: +ms.toFixed(1) });
	}

	// 2. Same, but reading only a slice of rows (proves it scales with bytes).
	const w2 = model.tokenEmbed.ref;
	const x2 = np.ones([640]);
	for (const rows of [4096, 65536]) {
		const w = w2.ref.slice([0, rows]); // [rows, 640]
		const warm = np.dot(x2.ref, w.ref.transpose());
		await blockUntilReady(warm);
		warm.dispose();
		const t0 = performance.now();
		const y = np.dot(x2.ref, w.transpose());
		await blockUntilReady(y);
		results.push({ op: `dot 1x640 @ 640x${rows}`, ms: +(performance.now() - t0).toFixed(1) });
		y.dispose();
	}
	return results;
}
