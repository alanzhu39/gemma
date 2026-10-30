/**
 * Kernel-to-kernel comparison of the three logits projections, on resident data.
 *
 * The whole-step A/B on this box is unreadable: one variant's per-round times
 * ranged 39-219 ms for identical work, because the machine runs at ~99% external
 * CPU load. Medians of that cannot resolve a 5% difference and should not be
 * used to claim or deny one.
 *
 * External interference only ever *adds* time, so the MINIMUM over many reps is
 * the statistic that survives contention: it is the run that got the least
 * disturbed. That is what this reports, alongside the median so the gap between
 * them is visible rather than hidden.
 *
 * The three candidates:
 *   strided    runLinear on tokenEmbed (a [vocab,dim] weight read transposed)
 *   contiguous runLinear on tokenEmbedT (a real [dim,vocab] copy)
 *   kernel     warp-per-row GEMV straight down the [vocab,dim] rows
 *
 * Every result is checked against a CPU reference before timing, and anything
 * above the card's read ceiling is rejected as a measurement bug.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { runLinear, runLogitsProjection, runLogitsStockPath } from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

const PEAK_GBPS = 192;
const REPS = 15;

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model = loadModelF32(safetensors.parse(await res.arrayBuffer()));

	const V = model.tokenEmbed.shape[0];
	const C = model.tokenEmbed.shape[1];
	const bytes = V * C * 4;

	// A realistic activation: the latent is an RMSNorm output, so order 1.
	const x = np.array(
		Float32Array.from({ length: C }, (_, i) => ((i % 13) - 6) * 0.02),
		{ dtype: np.float32 },
	);

	const variants: { name: string; run: () => np.Array }[] = [
		{ name: "strided", run: () => runLogitsStockPath(model, x.ref, "strided") },
		{ name: "contiguous", run: () => runLogitsStockPath(model, x.ref, "transposed") },
		{ name: "kernel", run: () => runLogitsProjection(model, x.ref) },
	];

	const ref = Float32Array.from(await x.ref.data() as Float32Array);
	const w = await model.tokenEmbed.ref.data() as Float32Array;
	const results: Record<string, unknown>[] = [];

	for (const v of variants) {
		// Verify before timing: a kernel that writes nothing yields zeros, which
		// would time beautifully.
		const check = v.run();
		const got = (await check.data()) as Float32Array;
		let bad = 0;
		let worst = 0;
		let zeros = 0;
		for (let r = 0; r < V; r++) {
			if (got[r] === 0) zeros++;
			let acc = 0;
			for (let k = 0; k < C; k++) acc += w[r * C + k] * ref[k];
			const d = Math.abs(acc - got[r]);
			if (d > 1e-2 * Math.max(1, Math.abs(acc))) bad++;
			worst = Math.max(worst, d);
		}

		const waits: number[] = [];
		for (let i = 0; i < REPS; i++) {
			const y = v.run();
			const t0 = performance.now();
			await blockUntilReady(y);
			waits.push(performance.now() - t0);
			await y.data();
		}
		waits.sort((a, b) => a - b);
		const min = waits[0];
		const median = waits[Math.floor(waits.length / 2)];
		const minGbps = bytes / 1e9 / (min / 1000);

		results.push({
			variant: v.name,
			badRows: bad,
			exactZeroRows: zeros,
			worstAbsDiff: +worst.toExponential(2),
			minMs: +min.toFixed(2),
			minGbps: +minGbps.toFixed(1),
			medianMs: +median.toFixed(2),
			medianGbps: +(bytes / 1e9 / (median / 1000)).toFixed(1),
			...(minGbps > PEAK_GBPS ? { error: `impossible ${minGbps.toFixed(0)} GB/s` } : {}),
		});
	}

	x.dispose();
	return { vocab: V, cols: C, mib: +(bytes / 2 ** 20).toFixed(0), reps: REPS, results };
}
