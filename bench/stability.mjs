/**
 * Long-run stability check for the decode loop.
 *
 * A per-token cost that grows with the number of tokens is the failure mode
 * that a 4-step correctness check cannot see: a shader compiled per position,
 * a cache that repads on every step, or a state field that is reallocated
 * instead of advanced all look fine for four steps and then dominate a real
 * generation. This runs a few hundred decode steps on the real model and
 * asserts that the per-step WebGPU traffic is flat.
 *
 * What is asserted (all structural: counts are deterministic, so this is valid
 * on the SwiftShader fallback where wall-clock is not):
 *
 *   1. Pipelines compiled per step do not grow with the step index. Steady
 *      state must be 0 after warmup: a nonzero value means some argument is
 *      still being folded into the shader as a literal, so every token compiles
 *      a new pipeline and the shader cache grows without bound.
 *   2. Dispatches and allocations per step do not grow, and the step-to-step
 *      spread is bounded. Growth here means per-step work depends on context
 *      length somewhere it should not (beyond the cache the model actually
 *      attends over).
 *   3. Logits stay finite and the position advances by exactly one per step.
 *
 * Usage:
 *   node bench/stability.mjs                       # 640-token context, 300 steps
 *   node bench/stability.mjs --ctx 128 --steps 300
 *   node bench/stability.mjs --steps 50            # quick smoke
 *
 * Takes the bench lock: this runs the model for minutes.
 */
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { acquireLock } from "./lock.mjs";
import { chromeArgs, chromeEnv } from "./chrome.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");

const argv = process.argv.slice(2);
const opt = (name, dflt) => {
	const i = argv.indexOf(name);
	return i >= 0 ? argv[i + 1] : dflt;
};
const ctx = Number(opt("--ctx", 640));
const nSteps = Number(opt("--steps", 300));
const tolerance = Number(opt("--tolerance", 0.02));

// The whole sequence (context + every decoded token) has to exist up front: the
// steps are teacher-forced so each one does identical work and the counts are
// comparable. decode-ctx1024 is the longest fixed token sequence available.
const workloads = JSON.parse(fs.readFileSync(path.join(HERE, "workloads.json"), "utf8"));
const source = workloads.decode.find((w) => w.name === "decode-ctx1024").tokens;
if (ctx + nSteps > source.length) {
	console.error(
		`ctx ${ctx} + ${nSteps} steps = ${ctx + nSteps} tokens, but only ${source.length} are available. ` +
			`Lower --ctx or --steps.`,
	);
	process.exit(2);
}
const context = source.slice(0, ctx);
const decode = source.slice(ctx, ctx + nSteps);

const release = await acquireLock({
	onWait: (h) => console.log(`[stability] waiting for pid ${h.pid} to finish`),
});
process.on("SIGINT", () => {
	void release();
	process.exit(130);
});

const server = await createServer({
	root: ROOT,
	configFile: path.join(HERE, "vite.config.ts"),
	server: { port: 0, strictPort: false },
	logLevel: "error",
});
await server.listen();
const base = server.resolvedUrls.local[0].replace(/\/$/, "");
const browser = await chromium.launch({
	headless: true,
	env: { ...process.env, ...chromeEnv() },
	args: chromeArgs({ software: argv.includes("--software") }),
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[page error]", e.message));
await page.addInitScript(() => {
	const c = (globalThis.__gpuCounts = {
		dispatch: 0,
		submit: 0,
		createBuffer: 0,
		createBufferBytes: 0,
		writeBuffer: 0,
		mapAsync: 0,
		bindGroup: 0,
		pipe: 0,
		pipeAsync: 0,
		shader: 0,
	});
	const wrap = (proto, name, fn) => {
		if (!proto || !proto[name]) return;
		const orig = proto[name];
		proto[name] = function (...a) {
			fn(...a);
			return orig.apply(this, a);
		};
	};
	wrap(GPUComputePassEncoder?.prototype, "dispatchWorkgroups", () => c.dispatch++);
	wrap(GPUQueue?.prototype, "submit", () => c.submit++);
	wrap(GPUQueue?.prototype, "writeBuffer", () => c.writeBuffer++);
	wrap(GPUDevice?.prototype, "createBuffer", (d) => {
		c.createBuffer++;
		c.createBufferBytes += d.size;
	});
	wrap(GPUDevice?.prototype, "createBindGroup", () => c.bindGroup++);
	wrap(GPUDevice?.prototype, "createComputePipeline", () => c.pipe++);
	wrap(GPUDevice?.prototype, "createComputePipelineAsync", () => c.pipeAsync++);
	wrap(GPUDevice?.prototype, "createShaderModule", () => c.shader++);
	wrap(GPUBuffer?.prototype, "mapAsync", () => c.mapAsync++);
});
await page.goto(`${base}/bench/index.html`, { waitUntil: "load", timeout: 120_000 });

let exitCode = 0;
try {
	const r = await page.evaluate(
		async ({ context, decode }) => (await import("/bench/stability.ts")).runStability(context, decode),
		{ context, decode },
	);

	// Least-squares slope of a per-step series, in units per step.
	const slope = (xs) => {
		const n = xs.length;
		const mx = (n - 1) / 2;
		const my = xs.reduce((a, b) => a + b, 0) / n;
		let num = 0;
		let den = 0;
		for (let i = 0; i < n; i++) {
			num += (i - mx) * (xs[i] - my);
			den += (i - mx) ** 2;
		}
		return den === 0 ? 0 : num / den;
	};
	const range = (xs) => [Math.min(...xs), Math.max(...xs)];

	// Warmup steps are excluded: the first one pays shader compilation for every
	// kernel, and the first few after a cache growth pay the repad.
	const warmup = 4;
	const steady = r.steps.slice(warmup);
	const series = (k) => steady.map((s) => s[k]);

	console.log(`precision ${r.precision}, ctx ${ctx}, ${nSteps} decode steps`);
	console.log(`cache at end: sliding ${r.caps.sliding}, global ${r.caps.global}`);
	console.log(`prefill ${(r.prefillMs / 1000).toFixed(0)}s, decode ${(r.decodeMs / 1000).toFixed(0)}s (CPU fallback, relative only)`);
	console.log("");
	console.log(`per-step traffic over ${steady.length} steady-state steps:`);
	for (const k of ["dispatch", "submit", "createBuffer", "createBufferBytes", "bindGroup", "pipe", "pipeAsync", "shader"]) {
		const xs = series(k);
		const [lo, hi] = range(xs);
		console.log(
			`  ${k.padEnd(18)} min=${String(lo).padStart(8)} max=${String(hi).padStart(8)} ` +
				`slope=${slope(xs).toExponential(2)}/step`,
		);
	}
	console.log("");
	console.log(`first 8 steps (disp/pipes/allocKiB): ${r.steps.slice(0, 8).map((s) => `${s.dispatch}/${s.pipeAsync + s.pipe}/${(s.createBufferBytes / 1024).toFixed(0)}`).join(" ")}`);
	console.log(`last 8 steps  (disp/pipes/allocKiB): ${r.steps.slice(-8).map((s) => `${s.dispatch}/${s.pipeAsync + s.pipe}/${(s.createBufferBytes / 1024).toFixed(0)}`).join(" ")}`);

	const problems = [];
	const pipes = series("pipeAsync").map((v, i) => v + series("pipe")[i]);
	const pipeTotal = pipes.reduce((a, b) => a + b, 0);
	if (pipeTotal > 0) {
		problems.push(
			`${pipeTotal} compute pipeline(s) compiled during ${steady.length} steady-state steps ` +
				`(first at step ${warmup + pipes.findIndex((v) => v > 0)}). ` +
				`Steady state must compile none: a nonzero count means an argument is still folded ` +
				`into the shader as a literal, so every token rebuilds it and the shader cache grows ` +
				`for the whole generation.`,
		);
	}
	if (slope(pipes) > 0) problems.push(`pipeline count per step grows with step index (slope ${slope(pipes).toExponential(2)})`);

	const dispatchSlope = slope(series("dispatch"));
	if (dispatchSlope > 0.01) {
		problems.push(
			`dispatches per step grow with step index (slope ${dispatchSlope.toFixed(3)}/step): ` +
				`per-token work depends on how long the sequence is`,
		);
	}
	const allocSlope = slope(series("createBufferBytes"));
	if (allocSlope > 1) {
		problems.push(
			`allocation per step grows by ${(allocSlope / 1024).toFixed(1)} KiB/step: something is ` +
				`re-allocated rather than advanced (look for a state field rebuilt from a JS value)`,
		);
	}
	// The cache legitimately grows during the run (repad to the next 128
	// boundary), so the dispatch series is allowed a few level shifts, not drift.
	const [dLo, dHi] = range(series("dispatch"));
	if (dHi - dLo > Math.max(8, dLo * 0.05)) {
		problems.push(
			`dispatch count per step varies between ${dLo} and ${dHi}, more than a cache-growth step ` +
				`can explain`,
		);
	}
	if (!r.finite) problems.push("non-finite logits appeared during the run");
	if (r.position !== ctx + nSteps) {
		problems.push(`position ended at ${r.position}, expected ${ctx + nSteps}`);
	}
	if (r.argmaxCount !== nSteps + 1) {
		problems.push(`argmax computed for ${r.argmaxCount} positions, expected ${nSteps + 1}`);
	}
	if (r.selfErr > tolerance) {
		problems.push(`decode drifted from a single prefill pass by rel err ${r.selfErr.toExponential(2)}`);
	}

	if (problems.length) {
		console.log("");
		for (const p of problems) console.log(`FAIL ${p}`);
		if (pipeTotal > 0) {
			console.log(
				"\nThis is a per-token shader compile, not a correctness failure. Known open item: the\n" +
					"input token is built as a JS number array, so its value is folded into the graph as\n" +
					"a constant and each new token id needs a new shader.",
			);
		}
		exitCode = 1;
	} else {
		console.log("");
		console.log(
			`PASS flat over ${steady.length} steps: 0 pipelines compiled, dispatches ` +
				`${dLo}-${dHi}/step, allocation drift ${allocSlope.toFixed(1)} B/step, ` +
				`argmax stable, decode vs prefill rel err ${r.selfErr.toExponential(2)}`,
		);
	}
} catch (e) {
	console.error(e);
	exitCode = 1;
} finally {
	await browser.close();
	await server.close();
	await release();
}
process.exit(exitCode);
