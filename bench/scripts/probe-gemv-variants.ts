/**
 * Why is the projection 9.2 GB/s when a plain read of the same bytes is 101+?
 *
 * The dot lowers to one Kernel with a Reduction (one dispatch, no materialised
 * intermediate), so "it writes 640 MiB twice" is not the explanation.
 *
 * This isolates the variables one at a time, all against the SAME 640 MiB
 * weight tensor and the same [640] vector:
 *
 *   1. sum over the whole matrix      - pure read, no per-row dot
 *   2. dot with a [1,640] vector      - what the model does (batched shape)
 *   3. dot with a [640] vector        - same, unbatched
 *   4. the same dot on a [65536,640] slice  - is cost linear in rows?
 *   5. the same dot at f32 vs the model's actual dtype path
 *
 * If (1) is fast and (2) is slow, the reduction's access pattern is the problem.
 * If (4) scales linearly, it is bandwidth-bound but badly; if it does not, there
 * is a fixed per-call cost.
 */
import { blockUntilReady, defaultDevice, init, numpy as np } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { loadModelF32 } from "../f32-model";

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model = loadModelF32(safetensors.parse(await res.arrayBuffer()));
	const W = model.tokenEmbed.ref; // [262144, 640] f32
	const rows = W.shape[0];

	// Every measurement must read a value back out. Without a readback jax-js
	// never dispatches the work, and the first version of this probe reported
	// 1973 GB/s on a 192 GB/s card - the work was elided, not fast.
	const t = async (label: string, fn: () => np.Array, bytesMoved: number, reps = 5) => {
		const warm = fn();
		await warm.data();
		const t0 = performance.now();
		for (let i = 0; i < reps; i++) {
			const r = fn();
			await r.data();
		}
		const ms = (performance.now() - t0) / reps;
		const gbps = bytesMoved / 1e9 / (ms / 1000);
		if (gbps > 192) throw new Error(`${label}: impossible ${gbps.toFixed(0)} GB/s - work was elided`);
		return { op: label, ms: +ms.toFixed(2), gbps: +gbps.toFixed(1) };
	};

	const out: Record<string, number | string>[] = [];
	const fullBytes = rows * 640 * 4;

	// 1. Sum the entire matrix: a pure streaming read, no per-row output.
	const flat = W.ref.reshape([-1]);
	out.push(await t("sum(whole matrix) [pure read]", () => flat.ref.sum(-1), fullBytes));

	// 2. The model's shape: [1,640] dot.
	const x1 = np.ones([1, 640]);
	out.push(await t("dot([1,640], W^T) [model shape]", () => np.dot(x1.ref, W.ref.transpose()), fullBytes));

	// 3. Unbatched [640] dot.
	const x0 = np.ones([640]);
	out.push(await t("dot([640], W^T) [unbatched]", () => np.dot(x0.ref, W.ref.transpose()), fullBytes));

	// 4. A quarter of the rows, to test linearity.
	const quarter = W.ref.slice([0, rows / 4]);
	const qBytes = (rows / 4) * 640 * 4;
	out.push(await t("dot([640], quarter W^T)", () => np.dot(x0.ref, quarter.ref.transpose()), qBytes));

	// 5. Same quarter via sum, to compare access patterns at equal bytes.
	const qflat = quarter.ref.reshape([-1]);
	out.push(await t("sum(quarter) [pure read]", () => qflat.ref.sum(-1), qBytes));

	return out;
}
