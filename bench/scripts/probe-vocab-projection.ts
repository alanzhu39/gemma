/**
 * The vocabulary projection, measured three ways, because they disagree.
 *
 * The projection is `runLinear({weight: tokenEmbed}, latent[-1])` and it reads
 * 640 MiB to emit 262144 numbers. Three numbers matter and they are not the
 * same thing:
 *
 *  1. isolated stock   - `np.dot` on resident data, nothing queued ahead.
 *                        The honest kernel-to-kernel comparison against
 *                        warp-per-row.
 *  2. isolated warp    - the custom kernel alone, same data, same conditions.
 *  3. in-model         - the real decode step, with the projection timed
 *                        separately from the model around it.
 *
 * Only (3) is a result about the model. (1) and (3) differing by 2x is the
 * standing warning in this project: whatever the isolated kernel does, the
 * in-model number is the one that pays.
 *
 * Every variant is verified against a CPU reference over the whole vocabulary
 * before a time is reported, and bandwidth above the card's read ceiling is
 * reported as a measurement bug rather than a fast kernel.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import {
	createGemma3State,
	runGemma3Step,
	runLinear,
	runLogitsProjection,
} from "../../src/lib/gemma3/model";
import { projectVocab } from "../../src/lib/gemma3/vocab-projection";
import { loadModelF32 } from "../f32-model";

/** Above this, the measurement is wrong. Measured read ceiling is ~101-140 GB/s. */
const PEAK_GBPS = 192;

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model = loadModelF32(safetensors.parse(await res.arrayBuffer()));

	const V = model.tokenEmbed.shape[0];
	const C = model.tokenEmbed.shape[1];
	const bytes = V * C * 4;
	const out: Record<string, unknown>[] = [];

	// ---- 1 & 2: isolated, on a resident activation -------------------------
	const x = np.array(
		Float32Array.from({ length: C }, (_, i) => ((i % 13) - 6) * 0.02),
		{ dtype: np.float32 },
	);

	const time = async (label: string, run: () => Promise<np.Array>) => {
		await (await run()).data(); // warm: compiles, and proves it produces data
		const waits: number[] = [];
		const readbacks: number[] = [];
		for (let i = 0; i < 7; i++) {
			const t0 = performance.now();
			const y = await run();
			const t1 = performance.now();
			await blockUntilReady(y);
			const t2 = performance.now();
			await y.data();
			const t3 = performance.now();
			waits.push(t2 - t1);
			readbacks.push(t3 - t2);
			void t0;
		}
		const med = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
		const ms = med(waits);
		const gbps = bytes / 1e9 / (ms / 1000);
		return {
			label,
			ms: +ms.toFixed(2),
			gbps: +gbps.toFixed(1),
			minMs: +Math.min(...waits).toFixed(2),
			maxMs: +Math.max(...waits).toFixed(2),
			readbackMs: +med(readbacks).toFixed(2),
			...(gbps > PEAK_GBPS ? { error: `impossible ${gbps.toFixed(0)} GB/s > ${PEAK_GBPS} peak` } : {}),
		};
	};

	out.push(
		await time("isolated: stock runLinear (np.dot)", async () =>
			runLinear({ weight: tree.ref(model).tokenEmbed }, x.ref),
		),
	);

	// Verify the custom kernel against a CPU reference over the whole vocabulary
	// before timing it. A kernel that writes nothing produces zeros, which is a
	// plausible-looking logits vector.
	const kernelOut = projectVocab(tree.ref(model).tokenEmbed, x.ref);
	if (!kernelOut) {
		out.push({ label: "isolated: warp-per-row", error: "kernel declined this shape/dtype" });
	} else {
		// `data()` consumes the array, so take refs of everything the reference
		// loop still needs before reading anything.
		const w = (await model.tokenEmbed.ref.data()) as Float32Array;
		const xv = (await x.ref.data()) as Float32Array;
		const got = (await kernelOut.data()) as Float32Array;
		let bad = 0;
		let worst = 0;
		let zeros = 0;
		for (let r = 0; r < V; r++) {
			if (got[r] === 0) zeros++;
			let ref = 0;
			for (let k = 0; k < C; k++) ref += w[r * C + k] * xv[k];
			const d = Math.abs(ref - got[r]);
			if (d > 1e-2 * Math.max(1, Math.abs(ref))) bad++;
			worst = Math.max(worst, d);
		}
		out.push({
			label: "kernel verification (all rows)",
			rows: V,
			badRows: bad,
			exactZeroRows: zeros,
			worstAbsDiff: +worst.toExponential(2),
		});
		out.push(
			await time("isolated: warp-per-row", async () => {
				const y = projectVocab(tree.ref(model).tokenEmbed, x.ref);
				if (!y) throw new Error("kernel declined");
				return y;
			}),
		);
	}

	// ---- 3: in-model -------------------------------------------------------
	// Same shape as bench/bench.ts: prefill a real context, then time each part
	// of a decode step separately. 638 crosses the cache-growth boundary.
	const ctx = new Array(640).fill(0).map((_, i) => 100 + (i % 200000));
	let state = createGemma3State(model);
	const pre = runGemma3Step(tree.ref(model), state, np.array(ctx, { dtype: np.uint32 }));
	await blockUntilReady(pre.latent);
	pre.latent.dispose();
	state = pre.state;

	const steps: Record<string, number>[] = [];
	for (let i = 0; i < 6; i++) {
		const tA = performance.now();
		const out2 = runGemma3Step(tree.ref(model), state, np.array([100 + i], { dtype: np.uint32 }));
		const tB = performance.now();
		await blockUntilReady(out2.latent);
		const tC = performance.now();

		// Stock projection, isolated from the model: nothing else queued.
		const iso0 = performance.now();
		const stockLogits = runLinear(
			{ weight: tree.ref(model).tokenEmbed },
			out2.latent.ref.slice([-1]).flatten(),
		);
		const iso1 = performance.now();
		await blockUntilReady(stockLogits);
		const iso2 = performance.now();
		await stockLogits.data();
		const iso3 = performance.now();

		// The same projection through the production path (kernel if available).
		const k0 = performance.now();
		const kLogits = runLogitsProjection(model, out2.latent.ref.slice([-1]).flatten());
		const k1 = performance.now();
		await blockUntilReady(kLogits);
		const k2 = performance.now();
		await kLogits.data();
		const k3 = performance.now();

		state = out2.state;
		out2.latent.dispose();

		steps.push({
			step: i,
			hostIssueModel: +(tB - tA).toFixed(1),
			gpuWaitModel: +(tC - tB).toFixed(1),
			stockIssue: +(iso1 - iso0).toFixed(1),
			stockWait: +(iso2 - iso1).toFixed(1),
			stockReadback: +(iso3 - iso2).toFixed(1),
			warpIssue: +(k1 - k0).toFixed(1),
			warpWait: +(k2 - k1).toFixed(1),
			warpReadback: +(k3 - k2).toFixed(1),
		});
	}

	const mean = (k: keyof (typeof steps)[number]) =>
		+(steps.reduce((a, s) => a + (s[k] as number), 0) / steps.length).toFixed(1);
	const stockMs = mean("stockWait");
	const warpMs = mean("warpWait");

	return {
		vocab: V,
		cols: C,
		mib: +(bytes / 2 ** 20).toFixed(0),
		isolated: out,
		inModel: {
			steps,
			stockWaitMs: stockMs,
			warpWaitMs: warpMs,
			stockGbps: +(bytes / 1e9 / (stockMs / 1000)).toFixed(1),
			warpGbps: +(bytes / 1e9 / (warpMs / 1000)).toFixed(1),
			savedMs: +(stockMs - warpMs).toFixed(1),
		},
	};
}
