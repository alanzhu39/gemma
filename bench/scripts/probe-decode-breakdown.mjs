/**
 * Driver for bench/scripts/probe-decode-breakdown.ts.
 *
 *   node bench/scripts/probe-decode-breakdown.mjs [--ctx 1024] [--steps 8]
 *
 * Serves the repo over Vite, opens headless Chromium on the real GPU, injects
 * WebGPU API-call timers before any page script runs, then reports the phase
 * breakdown as a table. Holds bench/lock.mjs's exclusive lock for the whole run
 * so another benchmark cannot contend for the GPU and corrupt the numbers.
 */
import { chromium } from "playwright";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { acquireLock } from "../lock.mjs";
import { chromeArgs, chromeEnv, getGpuDevices, isSoftwareRenderer } from "../chrome.mjs";
import { startContentionSample } from "../contention.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../..");

const argv = process.argv.slice(2);
const opt = (name, dflt) => {
	const i = argv.indexOf(name);
	return i >= 0 ? argv[i + 1] : dflt;
};
const ctx = Number(opt("--ctx", 1024));
const steps = Number(opt("--steps", 8));

const release = await acquireLock({
	onWait: (h) => console.log(`[breakdown] waiting for pid ${h.pid} to finish`),
});
process.on("SIGINT", () => {
	void release();
	process.exit(130);
});

const server = await createServer({
	root: ROOT,
	configFile: path.join(ROOT, "bench/vite.config.ts"),
	server: { port: 0, strictPort: false },
	logLevel: "error",
});
await server.listen();
const base = server.resolvedUrls.local[0].replace(/\/$/, "");

const browser = await chromium.launch({
	headless: true,
	env: { ...process.env, ...chromeEnv() },
	args: chromeArgs(),
});
const devices = await getGpuDevices(browser);
const renderer = devices[0] ?? "";
const software = isSoftwareRenderer(renderer);
if (software) {
	console.error(`[breakdown] refusing to report timings on a software renderer: ${renderer}`);
	await browser.close();
	await server.close();
	await release();
	process.exit(1);
}

const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[page error]", e.message));
page.on("console", (m) => {
	if (m.type() === "error") console.error("[page]", m.text());
});

// Count and time every WebGPU API call the engine makes. performance.now()
// around a synchronous call measures host time spent there, which is exactly
// the "jax-js overhead or GPU work" question.
await page.addInitScript(() => {
	const c = (globalThis.__gpuT = {
		submit: 0, submitMs: 0,
		dispatch: 0,
		createBuffer: 0, createBufferMs: 0,
		createBindGroup: 0, createBindGroupMs: 0,
		encoder: 0, encoderMs: 0,
		mapAsync: 0, mapAsyncMs: 0,
		writeBuffer: 0, writeBufferMs: 0,
		destroy: 0, destroyMs: 0,
	});
	const wrap = (proto, name, fn) => {
		if (!proto || !proto[name]) return;
		const orig = proto[name];
		proto[name] = function (...a) {
			const t0 = performance.now();
			const r = orig.apply(this, a);
			fn(performance.now() - t0);
			return r;
		};
	};
	wrap(globalThis.GPUComputePassEncoder?.prototype, "dispatchWorkgroups", () => c.dispatch++);
	wrap(globalThis.GPUQueue?.prototype, "submit", (dt) => {
		c.submit++;
		c.submitMs += dt;
	});
	wrap(globalThis.GPUQueue?.prototype, "writeBuffer", (dt) => {
		c.writeBuffer++;
		c.writeBufferMs += dt;
	});
	wrap(globalThis.GPUDevice?.prototype, "createBuffer", (dt) => {
		c.createBuffer++;
		c.createBufferMs += dt;
	});
	wrap(globalThis.GPUDevice?.prototype, "createBindGroup", (dt) => {
		c.createBindGroup++;
		c.createBindGroupMs += dt;
	});
	wrap(globalThis.GPUDevice?.prototype, "createCommandEncoder", (dt) => {
		c.encoder++;
		c.encoderMs += dt;
	});
	wrap(globalThis.GPUBuffer?.prototype, "mapAsync", (dt) => {
		c.mapAsync++;
		c.mapAsyncMs += dt;
	});
	wrap(globalThis.GPUBuffer?.prototype, "destroy", (dt) => {
		c.destroy++;
		c.destroyMs += dt;
	});
});

await page.goto(`${base}/bench/index.html`, { waitUntil: "load", timeout: 120_000 });

const finishContention = startContentionSample();
let result;
try {
	result = await page.evaluate(
		async ({ ctx, steps }) => {
			globalThis.__ctx = ctx;
			globalThis.__steps = steps;
			const { probe } = await import("/bench/scripts/probe-decode-breakdown.ts");
			return await probe();
		},
		{ ctx, steps },
	);
} finally {
	const contention = finishContention();
	if (result) result.contention = contention;
	await browser.close();
	await server.close();
	await release();
}

const f = (x) => (typeof x === "number" ? x.toFixed(2).padStart(9) : String(x).padStart(9));
const line = (k, v, extra = "") => console.log(`  ${String(k).padEnd(28)} ${f(v)} ms ${extra}`);

// Persist the raw result before printing, so a formatting bug cannot lose a run.
const fs = await import("node:fs");
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const rawFile = path.join(HERE, "..", "results", `breakdown-${stamp}.json`);
fs.writeFileSync(rawFile, JSON.stringify(result, null, 2) + "\n");
console.log(`\nraw: ${path.relative(ROOT, rawFile)}`);

console.log(`\nrenderer: ${renderer}`);
console.log(`config:   ${JSON.stringify(result.config)}`);
if (result.contention) {
	const c = result.contention;
	console.log(
		`contention: other processes used ${
			typeof c.otherUtilisation === "number" ? (c.otherUtilisation * 100).toFixed(0) + "%" : "?"
		} of ${c.cores} cores during the run; quiet=${c.quiet}`,
	);
}

const p = result.phases;
console.log(`\n=== A. real step, phase by phase (median of ${steps}) ===`);
line("model issue (host)", p.modelIssueMs);
line("model wait (GPU)", p.modelWaitMs);
line("projection issue", p.projIssueMs);
line("projection wait", p.projWaitMs);
line("sampling (real path)", p.samplingMs);
line("TOTAL", p.totalMs, `(min ${p.minTotalMs.toFixed(2)}, p90/min ${p.spread})`);
const sum = p.modelIssueMs + p.modelWaitMs + p.projIssueMs + p.projWaitMs + p.samplingMs;
line("sum of phases", sum, `(residual ${(p.totalMs - sum).toFixed(2)})`);

const s = result.samplingBreakdown;
console.log(`\n=== sampling breakdown (median) ===`);
line("div issue", s.divIssueMs);
line("argsort issue", s.argsortIssueMs);
line("argsort wait", s.argsortWaitMs);
line("softmax issue", s.softmaxIssueMs);
line("softmax .data() readback", s.probsReadbackMs);
line("token .data() readback", s.tokenReadbackMs);
line(
	"sum",
	s.divIssueMs + s.argsortIssueMs + s.argsortWaitMs + s.softmaxIssueMs + s.probsReadbackMs + s.tokenReadbackMs,
	`(top-64 prob sum ${s.probsSum})`,
);

console.log(`\n=== B. removing work from the step (median) ===`);
for (const [k, v] of Object.entries(result.stepVariants)) {
	line(k, v.medianMs, `min ${v.minMs} max ${v.maxMs} | re-sync ${v.resyncMs.toFixed(2)} ms`);
	console.log(
		`    submits ${v.submits} (${v.submitMs.toFixed(2)} ms in queue.submit)  ` +
			`dispatches ${v.dispatches}  encoders ${v.encoder} (${v.encoderMs.toFixed(2)} ms)  ` +
			`createBuffer ${v.createBuffer} (${v.createBufferMs.toFixed(2)} ms)  ` +
			`bindGroup ${v.createBindGroup} (${v.createBindGroupMs.toFixed(2)} ms)  ` +
			`destroy ${v.destroy} (${v.destroyMs.toFixed(2)} ms)  ` +
			`mapAsync ${v.mapAsync} (${v.mapAsyncMs.toFixed(2)} ms)`,
	);
}
const mo = result.stepVariants.modelOnly;
const mp = result.stepVariants.modelProj;
console.log(`  projection incremental cost = modelProj - modelOnly = ${(mp.medianMs - mo.medianMs).toFixed(2)} ms`);

const pr = result.projection;
console.log(`\n=== C. projection alone (${pr.shape}) ===`);
console.log(`  via runLinear: ${pr.viaRunLinearMs.join(" ")}`);
console.log(`  via np.dot:    ${pr.viaNpDotMs.join(" ")}`);
line("median", pr.medianMs, `${pr.gbps} GB/s of ${pr.peakGbps} peak`);
console.log(
	`  cross-check: max|runLinear - np.dot| = ${pr.maxAbsDiffBetweenPaths}, all finite = ${pr.finite}`,
);

const si = result.samplerIsolated;
console.log(`\n=== D. sampler alone, resident logits, ${si.runs.length} timed runs ===`);
line("median total", si.medianTotalMs, `(best ${si.minTotalMs.toFixed(2)})`);
console.log(
	`  host argmax ${si.hostArgmax}  argsort top-1 ${si.verify.argsortTop1}  match = ${si.argsortTop1MatchesHost}`,
);
console.log(
	`  per-run: ${si.runs
		.map(
			(r) =>
				`[sort ${(r.argsortIssue + r.argsortWait).toFixed(1)} ` +
				`sm ${(r.softmaxIssue + r.probsReadback).toFixed(1)} tok ${r.tokenReadback.toFixed(1)}]`,
		)
		.join(" ")}`,
);

console.log(`\n=== E. GPU kernel time from timestamp trace (one step, ctx ${ctx}) ===`);
if (typeof result.kernelTrace === "string") console.log(`  ${result.kernelTrace}`);
else {
	for (const [k, v] of Object.entries(result.kernelTrace)) {
		console.log(`  ${k.padEnd(30)} ${String(v.totalMs).padStart(9)} ms  x${v.count}`);
	}
}

// F. The missing-time locator: GPU completion timeline per submit.
const tlOf = (v) => {
	const rows = v.timeline.filter((t) => t.n > 1);
	if (!rows.length) return null;
	return {
		n: rows.map((t) => t.n).sort((a, b) => a - b)[rows.length >> 1],
		span: rows.map((t) => t.spanMs).sort((a, b) => a - b)[rows.length >> 1],
		medianGap: rows.map((t) => t.medianGapMs).sort((a, b) => a - b)[rows.length >> 1],
		p90Gap: rows.map((t) => t.p90GapMs).sort((a, b) => a - b)[rows.length >> 1],
		maxGap: Math.max(...rows.map((t) => t.maxGapMs)),
	};
};
console.log(`\n=== F. GPU completion timeline per submit (median across steps) ===`);
for (const [k, v] of Object.entries(result.stepVariants)) {
	const t = tlOf(v);
	if (!t) continue;
	console.log(
		`  ${k.padEnd(12)} submits ${String(t.n).padStart(4)}  span ${t.span.toFixed(1).padStart(7)} ms  ` +
			`gap/submit median ${t.medianGap.toFixed(2)} ms  p90 ${t.p90Gap.toFixed(2)}  max ${t.maxGap.toFixed(1)}`,
	);
	console.log(
		`               implied GPU work = span - (n-1)*medianGap = ` +
			`${(t.span - (t.n - 1) * t.medianGap).toFixed(1)} ms`,
	);
}
if (result.projection.timeline) {
	const t = result.projection.timeline;
	console.log(`  projection   submits ${t.n}  span ${t.spanMs} ms  maxGap ${t.maxGapMs}`);
}

console.log(`\n=== G. submission shape, 468 dependent kernels (median of 3) ===`);
for (const s of result.submitShapes) {
	console.log(
		`  ${s.shape.padEnd(30)} ${String(s.ms).padStart(8)} ms  ${String(s.usPerKernel).padStart(6)} us/kernel`,
	);
}


console.log(`\nfull step totals: ${result.fullSteps.map((r) => r.total).join(" ")}`);
