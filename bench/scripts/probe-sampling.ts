/**
 * How much of a decode step is token sampling rather than the model?
 *
 * `sampleLogits` does three expensive-looking things per token:
 *   - argsort() over the full 262144-element vocabulary
 *   - .data() readback of the top-k softmax (forces a pipeline drain)
 *   - a second .data() readback for the chosen index
 *
 * Two readbacks per token on a pipeline that was just fed 469 dispatches means
 * two full drains. This times each piece separately against a realistic logits
 * vector, so the cost can be attributed rather than guessed.
 */
import { blockUntilReady, defaultDevice, init, numpy as np } from "@jax-js/jax";

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");

	const V = 262144;
	const out: Record<string, number | string>[] = [];

	// A logits-like vector: a few large values and a long tail.
	const raw = new Float32Array(V);
	for (let i = 0; i < V; i++) raw[i] = Math.sin(i * 0.001) * 10;
	const logits = np.array(raw, { dtype: np.float32 });

	const time = async (label: string, fn: () => Promise<unknown>, reps = 5) => {
		await fn(); // warm
		const t0 = performance.now();
		for (let i = 0; i < reps; i++) await fn();
		return { op: label, ms: +((performance.now() - t0) / reps).toFixed(2) };
	};

	// 1. argsort alone, no readback.
	out.push(await time("argsort(262144)", async () => {
		const a = logits.ref.argsort();
		await blockUntilReady(a);
		a.dispose();
	}));

	// 2. argsort + one readback (what topKIndices effectively costs).
	out.push(await time("argsort + .data()", async () => {
		const a = logits.ref.argsort();
		await a.data();
	}));

	// 3. The full current sampling path shape: argsort, slice, softmax, readback.
	out.push(await time("argsort + slice64 + softmax + .data()", async () => {
		const { nn } = await import("@jax-js/jax");
		const idx = np.flip(logits.ref.argsort()).slice([0, 64]);
		const probs = nn.softmax(logits.ref.slice(idx.ref));
		await probs.data();
		idx.dispose();
	}));

	// 4. A single element readback, as the second .data() does.
	const tok = np.array([1], { dtype: np.int32 });
	out.push(await time("single-element .data()", async () => {
		await tok.ref.data();
	}));

	// 5. Baseline: an empty sync, to separate drain cost from work.
	out.push(await time("blockUntilReady only", async () => {
		await blockUntilReady(logits.ref);
	}));

	logits.dispose();
	tok.dispose();
	return out;
}
