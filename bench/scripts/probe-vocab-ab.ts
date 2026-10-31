/**
 * Definitive A/B of the vocabulary projection, shaped exactly like bench.ts.
 *
 * The first version of this probe drained the model with `blockUntilReady`
 * before timing the projection, which measured the kernel in isolation but not
 * the thing `bench/bench.ts` measures: there, the projection is dispatched
 * straight after the model with no sync between them, so it pipelines. A number
 * that disagrees with the harness is not evidence about the harness.
 *
 * So this reproduces `timeDecode` precisely - `runGemma3Step`, then the
 * projection, then one `blockUntilReady` and a `.data()` - and merely swaps
 * *which* projection is called. Three variants, because there are three real
 * candidates and they must not be confused with each other:
 *
 *   stock        runLinear on tokenEmbed          (the strided view)
 *   stockT       runLinear on tokenEmbedT         (contiguous copy)
 *   kernel       the warp-per-row kernel
 *
 * Variants are interleaved within one session and their order rotated every
 * round, because this box runs at ~99% external CPU load and a monotonic drift
 * would otherwise be attributed to whichever variant ran first.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import {
	createGemma3State,
	runGemma3Step,
	runLogitsProjection,
	runLogitsStockPath,
	type Gemma3State,
} from "../../src/lib/gemma3/model";
import { loadModelF32 } from "../f32-model";

const VARIANTS = ["stock", "stockT", "kernel"] as const;
type Variant = (typeof VARIANTS)[number];
const ROUNDS = 9;
const STEPS_PER_ROUND = 4;

export async function probe() {
	await init("webgpu");
	defaultDevice("webgpu");
	const res = await fetch("/weights/model.safetensors");
	const model = loadModelF32(safetensors.parse(await res.arrayBuffer()));

	const V = model.tokenEmbed.shape[0];
	const C = model.tokenEmbed.shape[1];
	const bytes = V * C * 4;
	const hasT = !!model.tokenEmbedT;

	const ctx = new Array(1024).fill(0).map((_, i) => 100 + (i % 200000));
	let base = createGemma3State(model);
	const pre = runGemma3Step(tree.ref(model), base, np.array(ctx, { dtype: np.uint32 }));
	await blockUntilReady(pre.latent);
	pre.latent.dispose();
	base = pre.state;

	/**
	 * A step consumes its state: `positionTensor` is advanced on-device and each
	 * KV-cache entry is replaced by a fresh `np.where` result. To run several
	 * variants from the same prefilled context, hand each one its own clone.
	 *
	 * Sharing the underlying buffers is safe because nothing writes in place:
	 * `np.where` builds a new array and the old one is only released, so a
	 * cloned `.ref` keeps the original alive and unmodified.
	 */
	const cloneState = (src: Gemma3State): Gemma3State => ({
		kvCaches: src.kvCaches.map((c) => ({ k: c.k.ref, v: c.v.ref })),
		position: src.position,
		positionTensor: src.positionTensor.ref,
		collectWeights: src.collectWeights,
	});
	const disposeState = (s: Gemma3State) => {
		for (const c of s.kvCaches) {
			c.k.dispose();
			c.v.dispose();
		}
		s.positionTensor.dispose();
	};

	const project = (which: Variant, latentRow: np.Array) => {
		if (which === "kernel") return runLogitsProjection(model, latentRow);
		if (which === "stockT") return runLogitsStockPath(model, latentRow, "transposed");
		return runLogitsStockPath(model, latentRow, "strided");
	};

	// Warm every variant once so no round pays a shader compile.
	for (const which of VARIANTS) {
		let st = cloneState(base);
		for (let i = 0; i < 2; i++) {
			const out = runGemma3Step(tree.ref(model), st, np.array([100 + i], { dtype: np.uint32 }));
			st = out.state;
			const logits = project(which, out.latent.ref.slice([-1]).flatten());
			await logits.data();
			out.latent.dispose();
		}
		disposeState(st);
	}

	const samples: Record<Variant, number[]> = { stock: [], stockT: [], kernel: [] };
	const rounds: Record<string, unknown>[] = [];

	for (let r = 0; r < ROUNDS; r++) {
		// Rotate the order so position in the round cannot favour a variant.
		const order = VARIANTS.map((_, i) => VARIANTS[(i + r) % VARIANTS.length]);
		const row: Record<string, number> = {};

		for (const which of order) {
			let st = cloneState(base);
			const times: number[] = [];
			for (let i = 0; i < STEPS_PER_ROUND; i++) {
				const t0 = performance.now();
				const out = runGemma3Step(tree.ref(model), st, np.array([100 + i], { dtype: np.uint32 }));
				st = out.state;
				// Exactly bench.ts: no sync between the model and the projection.
				const logits = project(which, out.latent.ref.slice([-1]).flatten());
				await blockUntilReady(logits);
				const ms = performance.now() - t0;
				await logits.data();
				out.latent.dispose();
				times.push(ms);
			}
			const med = [...times].sort((a, b) => a - b)[Math.floor(times.length / 2)];
			row[`${which}Ms`] = +med.toFixed(1);
			samples[which].push(med);
		}
		rounds.push({ round: r, order: order.join(","), ...row });
	}

	const med = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
	const summary: Record<string, unknown> = {};
	for (const which of VARIANTS) {
		const ms = med(samples[which]);
		summary[which] = {
			ms: +ms.toFixed(1),
			tps: +(1000 / ms).toFixed(2),
			minMs: +Math.min(...samples[which]).toFixed(1),
			maxMs: +Math.max(...samples[which]).toFixed(1),
			impliedProjectionMs: null,
		};
	}
	// Paired per-round ratio against stock, which is what drift cancels out of.
	const ratios = rounds.map((x) => (x.stockMs as number) / (x.kernelMs as number)).sort((a, b) => a - b);
	const ratiosT = rounds.map((x) => (x.stockMs as number) / (x.stockTMs as number)).sort((a, b) => a - b);

	return {
		vocab: V,
		mib: +(bytes / 2 ** 20).toFixed(0),
		hasTransposedCopy: hasT,
		summary,
		paired: {
			kernelVsStock: {
				median: +ratios[Math.floor(ratios.length / 2)].toFixed(2),
				min: +ratios[0].toFixed(2),
				max: +ratios[ratios.length - 1].toFixed(2),
			},
			kernelVsStockT: {
				median: +(summary.kernel as { ms: number }).ms === 0
					? null
					: +(((summary.stockT as { ms: number }).ms) / ((summary.kernel as { ms: number }).ms)).toFixed(2),
			},
			stockTVsStock: {
				median: +ratiosT[Math.floor(ratiosT.length / 2)].toFixed(2),
				min: +ratiosT[0].toFixed(2),
				max: +ratiosT[ratiosT.length - 1].toFixed(2),
			},
		},
		rounds,
	};
}
