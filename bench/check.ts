/**
 * Correctness check for inference optimizations.
 *
 * An optimization that changes what the model computes is not an optimization.
 * This runs a fixed, small workload and compares the output logits against a
 * golden reference recorded from a known-good revision.
 *
 * The comparison is tolerance-based rather than exact: GPU floating point is
 * not bitwise reproducible across shader changes (different summation orders
 * change rounding), so kernel rewrites legitimately shift the last few bits.
 * What must not change is the distribution of logits or the argmax.
 *
 * Golden files are keyed by precision (f16/f32) because the two paths round
 * differently and comparing across them would produce false failures.
 */
import { blockUntilReady, defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import { fromSafetensors, runGemma3Step, runLinear, createGemma3State } from "../src/lib/gemma3/model";
import type { Gemma3 } from "../src/lib/gemma3/model";
import { loadModelF32 } from "./f32-model";

export type Golden = {
	precision: "f16" | "f32";
	/** Fixed token ids used to generate the reference. */
	tokens: number[];
	/** Final-position logits, downsampled to keep the file small. */
	logits: number[];
	/** Indices of the logits we kept. */
	indices: number[];
	argmax: number;
	/** Shape of the latent the logits were computed from. */
	latentShape: number[];
};

const TOKENS = [2, 651, 892, 4497, 603, 322, 1128, 571, 407, 4724, 235265];

/** Fixed subset of the vocabulary, so goldens stay small and diffs readable. */
function sampleIndices(vocab: number): number[] {
	const out: number[] = [];
	for (let i = 0; i < 64; i++) out.push(Math.floor((i * (vocab - 1)) / 63));
	return out;
}

function hasF16(adapter: GPUAdapter) {
	return adapter.features.has("shader-f16" as GPUFeatureName);
}

async function loadModel(precision: "f16" | "f32"): Promise<Gemma3> {
	const res = await fetch("/weights/model.safetensors");
	if (!res.ok) throw new Error(`failed to fetch weights: ${res.status}`);
	const file = safetensors.parse(await res.arrayBuffer());
	// The f16 loader lives in the model module; the f32 one is local to bench.
	return precision === "f16" ? fromSafetensors(file) : loadModelF32(file);
}

/** Run the fixed workload and return the sampled final-position logits. */
export async function computeLogits(precision: "f16" | "f32"): Promise<Golden> {
	const devices = await init("webgpu");
	if (!devices.includes("webgpu")) throw new Error("WebGPU required");
	defaultDevice("webgpu");

	const model = await loadModel(precision);
	let state = createGemma3State(model);
	const input = np.array(TOKENS, { dtype: np.uint32 });
	const out = runGemma3Step(tree.ref(model), state, input);
	await blockUntilReady(out.latent);
	const latent = out.latent;
	const latentShape = latent.shape.slice();

	// Project only the final position: that is what feeds sampling.
	const logits = runLinear(
		{ weight: tree.ref(model).tokenEmbed },
		latent.ref.slice([-1]).flatten(),
	);
	const all = (await logits.data()) as Float16Array | Float32Array;
	const vocab = all.length;
	const indices = sampleIndices(vocab);
	const vals = indices.map((i) => Number(all[i]));

	let argmax = 0;
	for (let i = 1; i < all.length; i++) if (all[i] > all[argmax]) argmax = i;

	// `data()` consumes `logits`; only `latent` still needs freeing.
	latent.dispose();
	state.kvCaches.forEach((c) => {
		c.k.dispose();
		c.v.dispose();
	});

	return { precision, tokens: TOKENS, logits: vals, indices, argmax, latentShape };
}

function maxAbsDiff(a: number[], b: number[]): number {
	let m = 0;
	for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
	return m;
}

/** Relative error against the magnitude of the reference, which is scale-free. */
function relativeError(a: number[], b: number[]): number {
	let worst = 0;
	for (let i = 0; i < a.length; i++) {
		const denom = Math.max(1e-3, Math.abs(b[i]));
		worst = Math.max(worst, Math.abs(a[i] - b[i]) / denom);
	}
	return worst;
}

export async function runCheck(golden: Golden | null, opts: { tolerance?: number } = {}) {
	const devices = await init("webgpu");
	if (!devices.includes("webgpu")) throw new Error("WebGPU required");
	defaultDevice("webgpu");
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) throw new Error("no adapter");
	const precision: "f16" | "f32" = hasF16(adapter) ? "f16" : "f32";

	const actual = await computeLogits(precision);

	// Sanity checks that hold regardless of whether we have a golden.
	const finite = actual.logits.every((x) => Number.isFinite(x));
	const hasSpread = new Set(actual.logits.map((x) => x.toFixed(3))).size > 4;
	const base = {
		precision,
		actual,
		finite,
		hasSpread,
		argmax: actual.argmax,
		latentShape: actual.latentShape,
	};

	if (!golden) {
		return {
			...base,
			status: finite && hasSpread ? "no-golden" : "fail",
			message: golden
				? ""
				: `no golden for precision ${precision}; recording a new reference is required (npm run check:golden)`,
		};
	}

	if (golden.precision !== precision) {
		// Comparing f16 against f32 goldens is meaningless; say so instead of
		// reporting a spurious numeric failure.
		return {
			...base,
			status: "skipped",
			message: `golden is ${golden.precision} but this device runs ${precision}`,
		};
	}

	const tolerance = opts.tolerance ?? 0.02;
	const absDiff = maxAbsDiff(actual.logits, golden.logits);
	const relErr = relativeError(actual.logits, golden.logits);
	const argmaxMatch = actual.argmax === golden.argmax;

	const ok = finite && hasSpread && argmaxMatch && relErr <= tolerance;

	return {
		...base,
		status: ok ? "pass" : "fail",
		golden,
		absDiff,
		relErr,
		argmaxMatch,
		tolerance,
		message: ok
			? `logits match golden (max rel err ${relErr.toExponential(2)})`
			: [
					!finite && "non-finite logits",
					!hasSpread && "logits collapsed (all similar)",
					!argmaxMatch && `argmax ${actual.argmax} != golden ${golden.argmax}`,
					relErr > tolerance && `relative error ${relErr.toFixed(4)} > ${tolerance}`,
				]
					.filter(Boolean)
					.join("; "),
	};
}
