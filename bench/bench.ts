/**
 * Benchmarks for the Gemma 3 inference loop.
 *
 * Runs inside the browser page (driven by bench/run.mjs) because the engine is
 * WebGPU-only. Kept free of Svelte so it can be imported from a plain Vite
 * entry point and reused by whichever harness drives it.
 *
 * Measurement rules:
 *  - Every measured region is bracketed by a device synchronisation
 *    (`blockUntilReady`) so queued GPU work cannot leak into the next sample.
 *  - Warmup iterations run first and are discarded; shader compilation and
 *    buffer growth are one-time costs, not steady-state throughput.
 *  - Reported decode numbers are per-token latency at steady state, which is
 *    what actually determines interactive responsiveness.
 */
import {
	blockUntilReady,
	defaultDevice,
	init,
	numpy as np,
	tree,
	type Device,
} from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";
import {
	createGemma3State,
	fromSafetensors,
	runGemma3Step,
	runLinear,
	type Gemma3,
	type Gemma3State,
} from "../src/lib/gemma3/model";
import { loadModelF32 } from "./f32-model";

export type Workloads = {
	meta: { model: string; generatedAt: string };
	prefill: { name: string; tokens: number[] }[];
	decode: { name: string; contextLen: number; steps: number; tokens: number[] }[];
	batch: { name: string; batchSize: number; tokens: number[][] }[];
};

export type Sample = {
	workload: string;
	kind: "prefill" | "decode" | "batch";
	/** Milliseconds for the measured region (median across repeats). */
	ms: number;
	/** Tokens processed in the measured region. */
	tokens: number;
	/** tokens / second. For decode this is per-token latency inverted. */
	tps: number;
	detail?: Record<string, number>;
};

export type BenchOptions = {
	warmup?: number;
	repeats?: number;
	device?: Device;
	/** Restrict to these workload names (substring match). */
	only?: string[];
	onProgress?: (msg: string) => void;
};

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

/**
 * The published weights are FP16 and need the `shader-f16` WebGPU feature.
 * Not every adapter has it (notably CPU fallbacks such as SwiftShader), so we
 * fall back to an FP32 copy of the same weights. That is numerically
 * equivalent for our purposes and keeps the harness usable on any machine, but
 * it is slower and must be reported explicitly so nobody mistakes a CPU number
 * for a GPU one.
 */
type Precision = "f16" | "f32";

function hasShaderF16(adapter: GPUAdapter): boolean {
	// `features` is the portable accessor; `has()` is also present in Chromium.
	return adapter.features.has("shader-f16" as GPUFeatureName);
}

async function loadModel(precision: Precision): Promise<Gemma3> {
	const res = await fetch("/weights/model.safetensors");
	if (!res.ok) throw new Error(`failed to fetch weights: ${res.status}`);
	const file = safetensors.parse(await res.arrayBuffer());
	return precision === "f16" ? fromSafetensors(file) : loadModelF32(file);
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

function median(xs: number[]): number {
	const s = [...xs].sort((a, b) => a - b);
	if (s.length === 0) return NaN;
	const m = Math.floor(s.length / 2);
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function pct(xs: number[], p: number): number {
	const s = [...xs].sort((a, b) => a - b);
	if (s.length === 0) return NaN;
	const i = Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))));
	return s[i];
}

function summarize(xs: number[]) {
	const min = Math.min(...xs);
	const p90 = pct(xs, 90);
	return {
		median: median(xs),
		min,
		p90,
		mean: xs.reduce((a, b) => a + b, 0) / xs.length,
		/**
		 * How noisy this measurement was, as p90/min.
		 *
		 * This is the signal that actually matters for trustworthiness, and it
		 * works the same on a CPU rasteriser and on real hardware. CPU contention
		 * is a useful early warning but it is not the same thing: a GPU run can be
		 * 95% CPU-contended and still perfectly stable, because the host thread
		 * only submits commands. Measured on a GTX 1060, exactly that happened.
		 */
		spread: Number((p90 / Math.max(1e-9, min)).toFixed(3)),
	};
}

function disposeState(state: Gemma3State) {
	for (const c of state.kvCaches) {
		c.k.dispose();
		c.v.dispose();
	}
}

/**
 * Advance one step and force completion. Returns the wall time in ms.
 * Reading back one element guarantees we time finished work rather than work
 * that is merely queued in a command buffer.
 */
async function stepAndSync(model: Gemma3, state: Gemma3State, input: np.Array) {
	const out = runGemma3Step(tree.ref(model), state, input);
	await blockUntilReady(out.latent);
	// `.ref` so the readback does not consume the array we still hand back.
	await out.latent.ref.slice([-1]).data();
	return out;
}

// ---------------------------------------------------------------------------
// Workloads
// ---------------------------------------------------------------------------

/** One prefill pass over `tokens` from a fresh state. */
async function timePrefill(model: Gemma3, tokens: number[]): Promise<number> {
	const state = createGemma3State(model);
	const input = np.array(tokens, { dtype: np.uint32 });
	const t0 = performance.now();
	const out = await stepAndSync(model, state, input);
	const dt = performance.now() - t0;
	out.latent.dispose();
	// `runGemma3Step` returns the same cache arrays it was given (updated in
	// place), so `out.state` is the single owner and the only thing to free.
	disposeState(out.state);
	return dt;
}

/**
 * Prefill `context`, then time `steps` single-token decode steps.
 * Returns ms for the measured decode region.
 */
async function timeDecode(
	model: Gemma3,
	context: number[],
	steps: number,
): Promise<number> {
	let state = createGemma3State(model);
	const prefillInput = np.array(context, { dtype: np.uint32 });
	const prefill = await stepAndSync(model, state, prefillInput);
	prefill.latent.dispose();
	state = prefill.state;

	// Start decoding from the last context token.
	let token = context[context.length - 1];
	const t0 = performance.now();
	for (let i = 0; i < steps; i++) {
		const input = np.array([token], { dtype: np.uint32 });
		const out = runGemma3Step(tree.ref(model), state, input);
		state = out.state;
		// Logits projection is part of a real generation loop, so include it.
		const logits = runLinear(
			{ weight: tree.ref(model).tokenEmbed },
			out.latent.ref.slice([-1]).flatten(),
		);
		await blockUntilReady(logits);
		logits.dispose();
		out.latent.dispose();
		// Deterministic synthetic token so every repeat does identical work.
		token = 1000 + (i % 7);
	}
	const dt = performance.now() - t0;

	// Warmup state from the caller is disposable; the final one is not needed.
	disposeState(state);
	return dt;
}

/**
 * Concurrency workload: `batchSize` independent conversations advanced in
 * lockstep, all in flight, with a single synchronisation per round.
 *
 * This measures the engine as it exists today (one sequence per step, several
 * sequences interleaved). True batching — projecting [B,S] tokens through one
 * set of weights per step — needs a batched engine API, so it is not measured
 * here; `batchSupported` reports whether that path is available yet.
 */
async function timeInterleaved(
	model: Gemma3,
	prompts: number[][],
	steps: number,
): Promise<number> {
	const states = prompts.map(() => createGemma3State(model));
	const tokens = prompts.map((p) => p[p.length - 1]);

	// Prefill every sequence before timing.
	for (let b = 0; b < prompts.length; b++) {
		const input = np.array(prompts[b], { dtype: np.uint32 });
		const out = await stepAndSync(model, states[b], input);
		out.latent.dispose();
		states[b] = out.state;
	}

	const t0 = performance.now();
	for (let s = 0; s < steps; s++) {
		// Dispatch every sequence before syncing: this is what lets the device
		// overlap work across sequences and is the point of the measurement.
		const pending = prompts.map((_, b) => {
			const input = np.array([tokens[b]], { dtype: np.uint32 });
			const out = runGemma3Step(tree.ref(model), states[b], input);
			states[b] = out.state;
			const logits = runLinear(
				{ weight: tree.ref(model).tokenEmbed },
				out.latent.ref.slice([-1]).flatten(),
			);
			out.latent.dispose();
			return logits;
		});
		await blockUntilReady(pending);
		pending.forEach((p) => p.dispose());
		for (let b = 0; b < tokens.length; b++) tokens[b] = 1000 + ((s + b) % 7);
	}
	const dt = performance.now() - t0;

	states.forEach(disposeState);
	return dt;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

export async function runBenchmarks(workloads: Workloads, opts: BenchOptions = {}) {
	const { warmup = 2, repeats = 5, only, onProgress = () => {} } = opts;

	const devices = await init("webgpu");
	if (!devices.includes("webgpu")) throw new Error("WebGPU is required but not available");
	defaultDevice("webgpu");

	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) throw new Error("no WebGPU adapter available");
	const info = (adapter.info ?? {}) as Record<string, unknown>;
	const arch = String(info.architecture ?? "").toLowerCase();
	const desc = String(info.description ?? "").toLowerCase();
	const isSoftwareRenderer =
		adapter.isFallbackAdapter === true ||
		arch.includes("swiftshader") ||
		desc.includes("swiftshader") ||
		desc.includes("llvmpipe");

	const precision: Precision = hasShaderF16(adapter) ? "f16" : "f32";
	if (precision === "f32") {
		onProgress("adapter has no shader-f16; falling back to float32 weights");
	}

	onProgress(`loading weights (${precision})`);
	const model = await loadModel(precision);
	onProgress("loaded weights");

	const samples: Sample[] = [];
	const want = (name: string) => !only?.length || only.some((o) => name.includes(o));

	// Warm the GPU's clocks before any measurement. A Pascal card idles at
	// ~139 MHz against a 1911 MHz boost and takes ~100 ms to ramp, so the first
	// short workload is measured at idle clock. Observed as a 1.8x p90/min spread
	// on prefill-128 while every longer workload was flat. One throwaway full-size
	// prefill is enough to bring the clocks up.
	onProgress("warming up the GPU clocks");
	{
		const biggest = workloads.prefill.reduce((a, b) => (b.tokens.length > a.tokens.length ? b : a));
		for (let i = 0; i < 3; i++) await timePrefill(model, biggest.tokens);
	}

	// ---- prefill ----
	for (const w of workloads.prefill) {
		if (!want(w.name)) continue;
		onProgress(`${w.name}: warmup`);
		for (let i = 0; i < warmup; i++) await timePrefill(model, w.tokens);
		const times: number[] = [];
		for (let i = 0; i < repeats; i++) {
			onProgress(`${w.name}: repeat ${i + 1}/${repeats}`);
			times.push(await timePrefill(model, w.tokens));
		}
		const s = summarize(times);
		samples.push({
			workload: w.name,
			kind: "prefill",
			ms: s.median,
			tokens: w.tokens.length,
			tps: (w.tokens.length * 1000) / s.median,
			detail: { minMs: s.min, p90Ms: s.p90, meanMs: s.mean, spread: s.spread, runs: repeats },
		});
	}

	// ---- decode ----
	for (const w of workloads.decode) {
		if (!want(w.name)) continue;
		onProgress(`${w.name}: warmup`);
		for (let i = 0; i < warmup; i++) await timeDecode(model, w.tokens, w.steps);
		const perToken: number[] = [];
		for (let i = 0; i < repeats; i++) {
			onProgress(`${w.name}: repeat ${i + 1}/${repeats}`);
			perToken.push((await timeDecode(model, w.tokens, w.steps)) / w.steps);
		}
		const s = summarize(perToken);
		samples.push({
			workload: w.name,
			kind: "decode",
			ms: s.median,
			tokens: 1,
			tps: 1000 / s.median,
			detail: {
				contextLen: w.contextLen,
				stepsPerRun: w.steps,
				minMs: s.min,
				p90Ms: s.p90,
				spread: s.spread,
				runs: repeats,
			},
		});
	}

	// ---- concurrency ----
	for (const w of workloads.batch) {
		if (!want(w.name)) continue;
		const steps = 16;
		onProgress(`${w.name}: warmup`);
		await timeInterleaved(model, w.tokens, steps);
		const perToken: number[] = [];
		for (let i = 0; i < repeats; i++) {
			onProgress(`${w.name}: repeat ${i + 1}/${repeats}`);
			perToken.push((await timeInterleaved(model, w.tokens, steps)) / steps / w.batchSize);
		}
		const s = summarize(perToken);
		samples.push({
			workload: w.name,
			kind: "batch",
			ms: s.median,
			tokens: 1,
			tps: 1000 / s.median,
			detail: {
				batchSize: w.batchSize,
				// Aggregate goodput across concurrent conversations.
				totalTps: (1000 / s.median) * w.batchSize,
				batchSupported: 0,
				// Spread belongs here too: without it the batch numbers looked
				// trustworthy while the prefill/decode ones were being gated.
				minMs: s.min,
				p90Ms: s.p90,
				spread: s.spread,
				runs: repeats,
			},
		});
	}

	return {
		samples,
		sysInfo: {
			adapterInfo: info,
			isSoftwareRenderer,
			precision,
			userAgent: navigator.userAgent,
			backend: "webgpu",
			adaptiveFallback: adapter.isFallbackAdapter === true,
		},
		generatedAt: new Date().toISOString(),
	};
}
