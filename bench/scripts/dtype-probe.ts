/**
 * Is the slowness specific to matvec, or is every large op slow here?
 *
 * All three matvec forms measured ~462 ms (1.5 GB/s on a ~192 GB/s card), and
 * batching to M=4 did not change it. That rules out the transpose and rules out
 * arithmetic intensity. The remaining explanation is either (a) this backend on
 * this adapter is slow for everything, or (b) something specific to reductions.
 *
 * This measures plain elementwise work over the same 640 MiB to separate them.
 */
import { blockUntilReady, defaultDevice, init, numpy as np } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";

async function timed(fn: () => unknown): Promise<number> {
	const t0 = performance.now();
	await blockUntilReady(fn() as never);
	return performance.now() - t0;
}

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const file = safetensors.parse(await (await fetch("/weights/model.safetensors")).arrayBuffer());
	const emb = file.tensors["model.embed_tokens.weight"];
	const W = np.array(Float32Array.from(emb.data as Float16Array), {
		dtype: np.float32,
		shape: emb.shape,
	}); // [262144, 640], 640 MiB

	const bytes = W.shape[0] * W.shape[1] * 4;
	const rows: Record<string, number | string>[] = [];
	const measure = async (label: string, fn: () => unknown, trafficBytes = bytes) => {
		await timed(fn);
		const ms = await timed(fn);
		rows.push({ op: label, ms: +ms.toFixed(1), gbps: +(trafficBytes / 1e9 / (ms / 1000)).toFixed(1) });
	};

	// Pure elementwise: read 640 MiB, write 640 MiB => 1280 MiB of traffic.
	await measure("elementwise mul (read+write 640 MiB)", () => W.ref.mul(2), bytes * 2);

	// Reduction over the whole matrix: read 640 MiB only.
	await measure("sum over axis -1 (read 640 MiB)", () => W.ref.sum(-1), bytes);

	// The matvec we care about: read 640 MiB only.
	const x = np.ones([640], { dtype: np.float32 });
	await measure("dot(x, W^T) (read 640 MiB)", () => np.dot(x.ref, W.ref.transpose()), bytes);

	return { bytes, rows };
}
