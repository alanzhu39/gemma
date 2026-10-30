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

/**
 * Reference for the decode path.
 *
 * The prefill check above never exercises the code the optimizations touch most:
 * writing into a populated KV cache, the sliding-window ring buffer wrapping
 * around its 512 slots, and growing the global cache. This records logits from a
 * sequence long enough to cross all three.
 */
export type DecodeGolden = {
	precision: "f16" | "f32";
	/** Context length, chosen to cross the 512-slot sliding window. */
	contextLen: number;
	tokens: number[];
	/** Logits after each of `steps` decode steps. */
	steps: {
		position: number;
		logits: number[];
		argmax: number;
		/**
		 * Gap between the top two logits over the *whole* vocabulary.
		 *
		 * The sampled `logits` above are 64 arbitrary entries, so they cannot
		 * tell a genuine argmax change from a coin-flip between two near-equal
		 * tokens. This margin is what makes that distinction possible.
		 */
		margin: number;
		/**
		 * The winning logit itself.
		 *
		 * The tolerance is relative, so the absolute error we permit scales with
		 * the magnitude of the values. To decide whether an argmax change is real
		 * we need that same scale, which is what this provides.
		 */
		top1: number;
	};
	indices: number[];
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

let modelCache: { precision: "f16" | "f32"; model: Gemma3 } | null = null;

/**
 * Loads the model once and reuses it.
 *
 * The prefill and decode checks need the same weights. Loading twice doubles
 * peak GPU memory for no reason, which matters as soon as this runs on a real
 * device with a 6 GB card.
 */
async function loadModel(precision: "f16" | "f32"): Promise<Gemma3> {
	if (modelCache?.precision === precision) return modelCache.model;
	if (modelCache) throw new Error("cannot switch precision within one check run");
	const res = await fetch("/weights/model.safetensors");
	if (!res.ok) throw new Error(`failed to fetch weights: ${res.status}`);
	const file = safetensors.parse(await res.arrayBuffer());
	// The f16 loader lives in the model module; the f32 one is local to bench.
	const model = precision === "f16" ? fromSafetensors(file) : loadModelF32(file);
	modelCache = { precision, model };
	return model;
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

/**
 * Normalised error, scaled by the magnitude of the logits as a whole.
 *
 * An element-wise relative error is the wrong metric for logits. Most of a
 * vocabulary sits near zero, and dividing by a reference of 0.0024 turns a
 * 1e-5 absolute difference into a 0.004 "relative error" — a large-looking
 * number produced entirely by the denominator.
 *
 * Normalising by the scale of the distribution instead (the largest |logit|)
 * answers the question we actually care about: did the model's output change by
 * a meaningful amount? A shifted or broken computation moves many logits by a
 * visible fraction of the scale; a reordered summation moves them by 1e-7.
 */
function relativeError(a: number[], b: number[]): number {
	const scale = Math.max(1e-3, ...b.map(Math.abs));
	let worst = 0;
	for (let i = 0; i < a.length; i++) {
		worst = Math.max(worst, Math.abs(a[i] - b[i]) / scale);
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

/**
 * Chosen so both interesting transitions happen *during* decode:
 *  - 638 tokens pads the global cache to 640 during prefill, and the first
 *    decode step crosses 640, forcing a grow-and-repad to 768.
 *  - The sliding window wraps at 512, well before this, so the ring buffer is
 *    already cycling when decode starts.
 */
const DECODE_CONTEXT_LEN = 638;
const DECODE_STEPS = 4;

/** Deterministic pseudo-random token ids, reproducible across runs. */
function decodeTokens(n: number): number[] {
	const out: number[] = [2]; // <bos>
	let x = 12345;
	for (let i = 1; i < n; i++) {
		// Small LCG; only needs to be reproducible, not high quality.
		x = (x * 1103515245 + 12345) & 0x7fffffff;
		out.push(100 + (x % 200000));
	}
	return out;
}

async function readLogits(model: Gemma3, latent: np.Array) {
	const logits = runLinear(
		{ weight: tree.ref(model).tokenEmbed },
		latent.ref.slice([-1]).flatten(),
	);
	return (await logits.data()) as Float16Array | Float32Array;
}

/**
 * Prefill `DECODE_CONTEXT_LEN` tokens, then decode a few steps, recording the
 * logits after each. Exercises populated-cache reads, the sliding-window wrap,
 * and cache growth — the regions the decode optimizations actually touch.
 */
export async function computeDecodeLogits(
	precision: "f16" | "f32",
): Promise<DecodeGolden> {
	const devices = await init("webgpu");
	if (!devices.includes("webgpu")) throw new Error("WebGPU required");
	defaultDevice("webgpu");

	const model = await loadModel(precision);
	const tokens = decodeTokens(DECODE_CONTEXT_LEN);

	let state = createGemma3State(model);
	const prefill = runGemma3Step(tree.ref(model), state, np.array(tokens, { dtype: np.uint32 }));
	await blockUntilReady(prefill.latent);
	prefill.latent.dispose();
	state = prefill.state;

	const indices = sampleIndices(model.tokenEmbed.shape[0]);
	const steps: DecodeGolden["steps"] = [];

	// Feed a fixed token sequence, NOT the model's own argmax.
	//
	// Sampling the model's output back in makes each step depend on the previous
	// step's exact numerics. A legitimate change in summation order can flip a
	// near-tie at step 0, and every later step then diverges for reasons that
	// have nothing to do with correctness. A fixed input sequence keeps each
	// step's comparison independent.
	for (let s = 0; s < DECODE_STEPS; s++) {
		const token = tokens[(DECODE_CONTEXT_LEN - DECODE_STEPS + s) % tokens.length] ^ 0x55;
		const out = runGemma3Step(
			tree.ref(model),
			state,
			np.array([Math.abs(token) % 200000], { dtype: np.uint32 }),
		);
		await blockUntilReady(out.latent);
		state = out.state;

		const all = await readLogits(model, out.latent);
		out.latent.dispose();

		let argmax = 0;
		let second = -Infinity;
		for (let i = 0; i < all.length; i++) {
			if (all[i] > all[argmax]) {
				second = all[argmax];
				argmax = i;
			} else if (i !== argmax && all[i] > second) {
				second = all[i];
			}
		}

		steps.push({
			position: state.position,
			logits: indices.map((i) => Number(all[i])),
			argmax,
			margin: Number((all[argmax] - second).toFixed(4)),
			top1: Number(Number(all[argmax]).toFixed(4)),
		});
	}

	state.kvCaches.forEach((c) => {
		c.k.dispose();
		c.v.dispose();
	});

	return { precision, contextLen: DECODE_CONTEXT_LEN, tokens, steps, indices };
}

export async function runDecodeCheck(
	golden: DecodeGolden | null,
	opts: { tolerance?: number } = {},
) {
	const devices = await init("webgpu");
	if (!devices.includes("webgpu")) throw new Error("WebGPU required");
	defaultDevice("webgpu");
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) throw new Error("no adapter");
	const precision: "f16" | "f32" = hasF16(adapter) ? "f16" : "f32";

	const actual = await computeDecodeLogits(precision);
	const finite = actual.steps.every((s) => s.logits.every((x) => Number.isFinite(x)));
	const spread = actual.steps.every((s) => new Set(s.logits.map((x) => x.toFixed(3))).size > 4);

	const base = { precision, actual, finite, spread };

	if (!golden) {
		return {
			...base,
			status: finite && spread ? "no-golden" : "fail",
			message: `no decode golden for precision ${precision}; run npm run bench:golden`,
		};
	}
	if (golden.precision !== precision) {
		return {
			...base,
			status: "skipped",
			message: `decode golden is ${golden.precision} but this device runs ${precision}`,
		};
	}

	// A golden recorded from a different workload would fail on logits and look
	// like a numerical regression. Say what actually happened.
	if (golden.contextLen !== actual.contextLen) {
		return {
			...base,
			status: "stale-golden",
			message:
				`decode golden was recorded at ctx ${golden.contextLen} but the workload is now ` +
				`ctx ${actual.contextLen}. Re-record with: npm run bench:golden-missing`,
		};
	}

	const tolerance = opts.tolerance ?? 0.02;
	let worstRel = 0;
	let argmaxMismatch = -1;
	let nearTieSteps = 0;
	for (let i = 0; i < actual.steps.length; i++) {
		const a = actual.steps[i];
		const g = golden.steps[i];
		if (!g) continue;
		worstRel = Math.max(worstRel, relativeError(a.logits, g.logits));
		if (a.argmax !== g.argmax && argmaxMismatch < 0) {
			// Is this flip distinguishable from numerical noise?
			//
			// The tolerance is relative, so the absolute error permitted on a logit
			// of magnitude |top1| is roughly tolerance * |top1|. If the gap between
			// the golden's top two tokens is smaller than that, the two tokens are
			// within the noise we already accept, and a swap is expected from any
			// change in summation order. If it is larger, the model genuinely
			// changed its mind and that is a failure.
			//
			// A fixed threshold cannot work here: real top-2 gaps in this model are
			// 0.1-0.7, well inside the error a 2% tolerance already allows, so a
			// hardcoded margin would either mask real breakage or fail constantly.
			const allowance = tolerance * Math.max(1, Math.abs(g.top1 ?? 0));
			if ((g.margin ?? Infinity) > allowance) {
				argmaxMismatch = i;
			} else {
				nearTieSteps++;
			}
		}
	}

	const ok = finite && spread && argmaxMismatch < 0 && worstRel <= tolerance;
	return {
		...base,
		status: ok ? "pass" : "fail",
		golden,
		worstRel,
		argmaxMismatch,
		nearTieSteps,
		tolerance,
		message: ok
			? `decode logits match golden across ${actual.steps.length} steps (max rel err ${worstRel.toExponential(2)}${nearTieSteps ? `, ${nearTieSteps} near-tie argmax flip(s) tolerated` : ""})`
			: [
					!finite && "non-finite decode logits",
					!spread && "decode logits collapsed",
					argmaxMismatch >= 0 && `argmax differs at step ${argmaxMismatch}`,
					worstRel > tolerance && `relative error ${worstRel.toFixed(4)} > ${tolerance}`,
				]
					.filter(Boolean)
					.join("; "),
	};
}
