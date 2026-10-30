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
 *   1. No synchronous pipeline compiles in steady state. Synchronous creation is
 *      jax-js recompiling a kernel it could not reuse, and it must not happen
 *      per token.
 *
 *   2. Asynchronous compiles are bounded by the number of distinct folded
 *      values, and their rate does not grow with the step index. This is the
 *      distinction that matters: a position literal rewrites the shader every
 *      step, so the count climbs forever; a token literal compiles once per
 *      distinct token id and then reuses, so its total is bounded and its slope
 *      decays. Asserting "zero compiles" conflates the two.
 *
 *   3. Dispatches per step do not grow with the step index, and their
 *      step-to-step spread stays inside what a cache repad explains.
 *
 *   4. Allocation per step is flat once the cache stops growing. Allocation is a
 *      step function - the KV cache is repadded to the next 128 rows at a known
 *      step - so the check looks at the tail after the last capacity change
 *      rather than fitting a line through the jumps.
 *
 *   5. Logits stay finite, the position advances by exactly one per step, and
 *      the last decode step still agrees with a single prefill of the same
 *      tokens.
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
	const syncPipes = series("pipe");

	// Pipelines, split by creation path, because the two mean different things.
	//
	// Synchronous creation is jax-js compiling a kernel it could not reuse: a
	// per-step cost, and it must be zero in steady state.
	//
	// Asynchronous creation is how a new *shape* or a new folded constant gets
	// compiled. It is legitimate for that to happen a bounded number of times -
	// one per distinct value, one per cache capacity - and it is not legitimate
	// for it to keep happening as the run goes on. So the assertion is on the
	// trend, not the total: a run that compiles the same number of pipelines
	// every step would have slope ~total/steps, while a run that is amortising a
	// finite set of distinct values has slope that decays toward zero.
	const syncTotal = syncPipes.reduce((a, b) => a + b, 0);
	if (syncTotal > 0) {
		problems.push(
			`${syncTotal} synchronously-compiled pipeline(s) in ${steady.length} steady-state steps ` +
				`(first at step ${warmup + syncPipes.findIndex((v) => v > 0)}): a kernel is being ` +
				`recompiled per token instead of reused`,
		);
	}

	// The bound that matters for a folded literal: a shader is keyed on the whole
	// graph, so each distinct (folded value, cache capacity) pair compiles once
	// and is then reused. A token id that repeats reuses its shader; the same
	// token after the cache grows to a new capacity does not, because the cache
	// shape is baked into the kernels that attend over it.
	const distinctKeys = new Set(steady.map((s) => `${s.token}@${s.globalCapacity}`)).size;
	const distinctTokens = new Set(steady.map((s) => s.token)).size;
	const asyncTotal = series("pipeAsync").reduce((a, b) => a + b, 0);
	if (asyncTotal > distinctKeys) {
		problems.push(
			`${asyncTotal} pipelines compiled for ${distinctKeys} distinct (token, capacity) pairs ` +
				`(${distinctTokens} distinct tokens over ${new Set(steady.map((s) => s.globalCapacity)).size} ` +
				`cache capacities): more than one per distinct specialised value, so something beyond ` +
				`the token literal is also being specialised per step`,
		);
	}
	const pipeSlope = slope(pipes);
	if (pipeSlope > 0.05) {
		problems.push(
			`pipeline compiles per step grow with step index (slope ${pipeSlope.toExponential(2)}/step): ` +
				`the compile count is not amortising, so the shader cache grows for the whole generation`,
		);
	}

	const dispatchSlope = slope(series("dispatch"));
	if (dispatchSlope > 0.01) {
		problems.push(
			`dispatches per step grow with step index (slope ${dispatchSlope.toFixed(3)}/step): ` +
				`per-token work depends on how long the sequence is`,
		);
	}

	// Allocation grows in steps, not linearly: the KV cache is repadded to the
	// next 128 rows whenever the context crosses a boundary, which is a one-off
	// jump at a known step. Fitting a line to that is meaningless, so the check
	// is: after the last capacity change, allocation must be flat. A leak keeps
	// climbing; a repad stops.
	let lastGrowth = -1;
	for (let i = 1; i < steady.length; i++) {
		if (steady[i].globalCapacity !== steady[i - 1].globalCapacity) lastGrowth = i;
	}
	const tailStart = Math.max(lastGrowth + 1, steady.length - Math.max(16, steady.length >> 2));
	const tail = steady.slice(tailStart);
	const tailAlloc = tail.map((s) => s.createBufferBytes);
	const [aLo, aHi] = range(tailAlloc);
	const tailSpread = aHi - aLo;
	// 1 KiB of slack: a genuine per-step reallocation is at least one buffer.
	if (tailSpread > 1024) {
		problems.push(
			`allocation per step varies by ${(tailSpread / 1024).toFixed(1)} KiB across the last ` +
				`${tail.length} steps (no cache growth in that window): something is re-allocated ` +
				`rather than advanced (look for a state field rebuilt from a JS value)`,
		);
	}
	const tailAllocSlope = slope(tailAlloc);
	if (tailAllocSlope > 256) {
		problems.push(
			`allocation per step still climbing by ${(tailAllocSlope / 1024).toFixed(1)} KiB/step ` +
				`over the last ${tail.length} steps, after the final cache growth at step ${warmup + lastGrowth}`,
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
		if (asyncTotal > 0 && syncTotal === 0) {
			console.log(
				`\nNote: ${asyncTotal} asynchronous compile(s) over ${steady.length} steps, slope ` +
					`${pipeSlope.toExponential(2)}/step. If the only literal left is the input token id, ` +
					`this is a bounded set of compiles that decays as token values repeat - not the ` +
					`per-step recompile that a position literal causes. Verify with a shader dump before ` +
					`treating it as a defect.`,
			);
		}
		exitCode = 1;
	} else {
		console.log("");
		console.log(
			`PASS flat over ${steady.length} steps: 0 sync pipelines, ${asyncTotal} async over ` +
				`${distinctKeys} distinct (token, capacity) pairs (slope ${pipeSlope.toExponential(2)}/step), ` +
				`dispatches ${dLo}-${dHi}/step, allocation flat within ${(tailSpread / 1024).toFixed(1)} KiB ` +
				`over the last ${tail.length} steps, argmax stable, decode vs prefill rel err ` +
				`${r.selfErr.toExponential(2)}`,
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
