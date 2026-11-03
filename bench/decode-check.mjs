/**
 * Decode-path correctness runner.
 *
 *   node bench/decode-check.mjs --ctx 636 --steps 8 --record /tmp/ref-<sha>.json
 *   node bench/decode-check.mjs --ctx 636 --steps 8 --ref /tmp/ref-<sha>.json
 *
 * Always checks decode's final step against a full-sequence prefill. With
 * --ref, also compares every decode step against a recorded run (record it
 * from the unmodified tree first). Exits non-zero on failure.
 */
import { chromium } from "playwright";
import fs from "node:fs";
import fsp from "node:fs/promises";
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
const ctx = Number(opt("--ctx", 636));
const nSteps = Number(opt("--steps", 8));
const recordFile = opt("--record", null);
const refFile = opt("--ref", null);
const tolerance = Number(opt("--tolerance", 0.02));

function relErr(a, b) {
	let worst = 0;
	for (let i = 0; i < a.length; i++) {
		worst = Math.max(worst, Math.abs(a[i] - b[i]) / Math.max(1e-3, Math.abs(b[i])));
	}
	return worst;
}

const workloads = JSON.parse(fs.readFileSync(path.join(HERE, "workloads.json"), "utf8"));
const source = workloads.decode.find((w) => w.name === "decode-ctx1024").tokens;
if (ctx + nSteps > source.length) throw new Error(`ctx+steps must be <= ${source.length}`);
const context = source.slice(0, ctx);
const decode = source.slice(ctx, ctx + nSteps);

const release = await acquireLock({
	onWait: (h) => console.log(`[decode-check] waiting for pid ${h.pid} to finish`),
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
// Count WebGPU API traffic so decode steps can be compared structurally; wall
// time on a shared CPU fallback is too noisy to mean anything.
await page.addInitScript(() => {
	const c = (globalThis.__gpuCounts = {
		dispatch: 0,
		submit: 0,
		createBuffer: 0,
		createBufferBytes: 0,
		copyB2B: 0,
		copyB2BBytes: 0,
		writeBuffer: 0,
		writeBufferBytes: 0,
		mapAsync: 0,
		bindGroup: 0,
	});
	const wrap = (proto, name, fn) => {
		if (!proto || !proto[name]) return;
		const orig = proto[name];
		proto[name] = function (...a) {
			fn(...a);
			return orig.apply(this, a);
		};
	};
	wrap(globalThis.GPUComputePassEncoder?.prototype, "dispatchWorkgroups", () => c.dispatch++);
	wrap(globalThis.GPUQueue?.prototype, "submit", () => c.submit++);
	wrap(globalThis.GPUQueue?.prototype, "writeBuffer", (_b, _o, d, _do, size) => {
		c.writeBuffer++;
		c.writeBufferBytes += size ?? d.byteLength;
	});
	wrap(globalThis.GPUDevice?.prototype, "createBuffer", (d) => {
		c.createBuffer++;
		c.createBufferBytes += d.size;
	});
	wrap(globalThis.GPUDevice?.prototype, "createBindGroup", () => c.bindGroup++);
	wrap(globalThis.GPUCommandEncoder?.prototype, "copyBufferToBuffer", (_a, _b, _c, _d, size) => {
		c.copyB2B++;
		c.copyB2BBytes += size;
	});
	wrap(globalThis.GPUBuffer?.prototype, "mapAsync", () => c.mapAsync++);
});
await page.goto(`${base}/bench/index.html`, { waitUntil: "load", timeout: 120_000 });

let exitCode = 0;
try {
	const r = await page.evaluate(
		async ({ context, decode }) => (await import("/bench/decode-check.ts")).runDecodeCheck(context, decode),
		{ context, decode },
	);
	const last = r.steps[r.steps.length - 1];
	const selfErr = relErr(last.logits, r.prefillLast.logits);
	const selfOk = last.argmax === r.prefillLast.argmax && selfErr <= tolerance;
	const ms = r.steps.slice(1).map((s) => s.ms);
	const sorted = [...ms].sort((a, b) => a - b);
	console.log(`precision ${r.precision}, ctx ${ctx}, ${nSteps} decode steps`);
	console.log(`step ms: ${ms.map((x) => x.toFixed(0)).join(" ")}`);
	console.log(`median step ms (CPU fallback, relative only): ${sorted[sorted.length >> 1].toFixed(1)}`);
	console.log(
		`decode vs full prefill: argmax ${last.argmax}/${r.prefillLast.argmax}, rel err ${selfErr.toExponential(2)} -> ${selfOk ? "OK" : "FAIL"}`,
	);
	if (!selfOk) exitCode = 1;
	// Per-step WebGPU traffic, median over the steady-state steps (skip the first,
	// which also pays shader compilation).
	const keys = Object.keys(r.steps[1]?.counts ?? {});
	const steady = r.steps.slice(2);
	if (keys.length && steady.length) {
		const med = (k) => {
			const v = steady.map((s) => s.counts[k]).sort((a, b) => a - b);
			return v[v.length >> 1];
		};
		console.log(`per decode step (median): ${keys.map((k) => `${k}=${med(k)}`).join(" ")}`);
		console.log(`per step createBufferBytes: ${steady.map((s) => s.counts.createBufferBytes).join(" ")}`);
	}

	if (recordFile) {
		await fsp.writeFile(recordFile, JSON.stringify(r) + "\n");
		console.log(`recorded ${recordFile}`);
	}
	if (refFile) {
		const ref = JSON.parse(await fsp.readFile(refFile, "utf8"));
		let worst = 0;
		let argmaxBad = 0;
		for (let i = 0; i < ref.steps.length; i++) {
			worst = Math.max(worst, relErr(r.steps[i].logits, ref.steps[i].logits));
			if (r.steps[i].argmax !== ref.steps[i].argmax) argmaxBad++;
		}
		const refOk = argmaxBad === 0 && worst <= tolerance;
		console.log(
			`vs ref: ${argmaxBad} argmax mismatches, worst rel err ${worst.toExponential(2)} -> ${refOk ? "OK" : "FAIL"}`,
		);
		if (!refOk) exitCode = 1;
	}
} catch (e) {
	console.error(e);
	exitCode = 1;
}
await browser.close();
await server.close();
await release();
process.exit(exitCode);
