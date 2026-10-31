/**
 * Driver for bench/scripts/alloc-cost.ts.
 *
 *   node bench/scripts/alloc-cost.mjs                 # ctx 1024, 6 decode steps
 *   node bench/scripts/alloc-cost.mjs --ctx 128 --steps 4
 *
 * Takes the bench lock, launches the same headless Chromium + Vite setup as
 * bench/run.mjs, and runs the probe in the page. Writes the raw JSON to
 * bench/results/alloc-cost-<stamp>.json so a claim can be checked against the
 * file rather than remembered.
 */
import { chromium } from "playwright";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { acquireLock } from "../lock.mjs";
import { startContentionSample } from "../contention.mjs";
import { chromeArgs, chromeEnv, getGpuDevices, isSoftwareRenderer } from "../chrome.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BENCH = path.resolve(HERE, "..");
const ROOT = path.resolve(BENCH, "..");
const WEIGHTS = path.join(BENCH, ".cache", "model.safetensors");
const RESULTS_DIR = path.join(BENCH, "results");

function parseArgs(argv) {
	const out = { ctxLen: 1024, steps: 6, software: false };
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (a === "--ctx") out.ctxLen = Number(argv[++i]);
		else if (a === "--steps") out.steps = Number(argv[++i]);
		else if (a === "--software") out.software = true;
	}
	return out;
}

const args = parseArgs(process.argv.slice(2));
if (!fs.existsSync(WEIGHTS)) {
	console.error(`Missing weights at ${WEIGHTS}\nRun: npm run bench:fetch`);
	process.exit(1);
}

const release = await acquireLock({
	onWait: (h) =>
		console.log(`[alloc] waiting for another run to finish (pid ${h.pid}, started ${new Date(h.startedAt).toISOString()})`),
});
for (const sig of ["SIGINT", "SIGTERM"]) {
	process.on(sig, () => {
		void release();
		process.exit(130);
	});
}

let result, rendererString, contention;
try {
	const finishContention = startContentionSample();
	const server = await createServer({
		root: ROOT,
		configFile: path.join(BENCH, "vite.config.ts"),
		server: { port: 0, strictPort: false },
		logLevel: "error",
	});
	await server.listen();
	const base = server.resolvedUrls.local[0].replace(/\/$/, "");

	const browser = await chromium.launch({
		headless: true,
		env: { ...process.env, ...chromeEnv() },
		args: chromeArgs({ software: args.software }),
	});
	rendererString = (await getGpuDevices(browser))[0] ?? "";

	const page = await browser.newPage();
	page.on("console", (m) => {
		const t = m.text();
		if (t.startsWith("[alloc]")) console.log(t);
	});
	page.on("pageerror", (e) => console.error("[page error]", e.message));
	await page.goto(`${base}/bench/index.html`, { waitUntil: "load", timeout: 120_000 });

	// jax-js creates its own device; the page cannot hook it any other way.
	// A destroyed-buffer bug is silent in the result and only announces itself
	// here, so record it and fail the run rather than print it.
	await page.addInitScript(() => {
		window.__gpuErrors = [];
		const proto = GPUAdapter.prototype;
		const original = proto.requestDevice;
		proto.requestDevice = function (...a) {
			const p = original.apply(this, a);
			p.then((device) => {
				device.addEventListener("uncapturederror", (e) => {
					window.__gpuErrors.push(String(e.error?.message ?? e.error));
				});
			}).catch(() => {});
			return p;
		};
	});
	await page.reload({ waitUntil: "load" });

	result = await page.evaluate(async (o) => {
		const { run } = await import("/bench/scripts/alloc-cost.ts");
		return await run({ ctxLen: o.ctxLen, steps: o.steps });
	}, args);

	const gpuErrors = await page.evaluate(() => window.__gpuErrors ?? []);
	result.gpuErrors = [...(result.gpuErrors ?? []), ...gpuErrors];
	contention = finishContention();
	await browser.close();
	await server.close();
} finally {
	await release();
}

result.sysInfo = {
	rendererString,
	isSoftwareRenderer: isSoftwareRenderer(rendererString),
	contention,
	git: (() => {
		try {
			const g = (...a) => execFileSync("git", a, { cwd: ROOT, encoding: "utf8" }).trim();
			return {
				rev: g("rev-parse", "HEAD"),
				branch: g("rev-parse", "--abbrev-ref", "HEAD"),
				dirty: g("status", "--porcelain") !== "",
			};
		} catch {
			return null;
		}
	})(),
};

console.log(`\n[alloc] renderer: ${rendererString || "(unknown)"}`);
if (result.sysInfo.isSoftwareRenderer) {
	console.log("[alloc] !! software renderer: absolute times are CPU-only, relative only");
}
console.log("[alloc] phase A (real decode step)");
const a = result.phaseA;
console.log(`  median step              ${a.medianStepMs} ms`);
console.log(`  buffers created per step ${a.medianBuffersPerStep}`);
console.log(`  alloc time per step      ${result.summary.medianAllocMsPerStep} ms (${result.summary.allocShareOfStepPct}% of step)`);
console.log(`    in model phase         ${a.medianAllocModelMs} ms`);
console.log(`    in projection phase    ${a.medianAllocProjMs} ms`);
console.log(`  bind group creation      ${a.medianBindGroupMs} ms`);
console.log(`  projection wait          ${a.medianProjectionWaitMs} ms (${result.summary.projectionShareOfStepPct}% of step)`);
console.log(`  independent CPU check    worst |gpu-cpu| ${a.numericCheck.worstAbsDiff} (ok=${a.numericCheck.ok})`);
console.log(`  first-step sanity        argmax ${a.firstStepArgmax}, top logit ${a.firstStepTopLogit}, finite ${a.firstStepFiniteFraction}`);
console.log("[alloc] phase B (projection in isolation, 640 MiB)");
for (const r of result.phaseB.results) {
	const spread = r.samples ? `  [${r.minMs}-${r.maxMs} over ${r.samples.length}]` : "";
	console.log(
		`  ${String(r.case).padEnd(58)} ${String(r.medianMs).padStart(7)} ms  ${r.gbps ? String(r.gbps).padStart(6) + " GB/s" : ""}${spread}`,
	);
}
console.log(`  verify ${JSON.stringify(result.phaseB.verify)}`);
if (result.phaseB.overPeak.length) {
	console.log(`  !! ${result.phaseB.overPeak.length} case(s) above 192 GB/s: measurement bug, not a result`);
	for (const r of result.phaseB.overPeak) console.log(`     ${r.case}: ${r.gbps} GB/s`);
}
if (result.phaseB.gpuErrors?.length) {
	console.log(`  !! GPU errors: ${result.phaseB.gpuErrors.join(" | ")}`);
}

await fsp.mkdir(RESULTS_DIR, { recursive: true });
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const file = path.join(RESULTS_DIR, `alloc-cost-${stamp}.json`);
await fsp.writeFile(file, JSON.stringify(result, null, 2) + "\n");
console.log(`\n[alloc] wrote ${path.relative(process.cwd(), file)}`);

const failed =
	(result.gpuErrors?.length ?? 0) > 0 ||
	(result.phaseB.overPeak?.length ?? 0) > 0 ||
	!result.phaseB.verify?.verified ||
	!a.numericCheck.ok ||
	a.firstStepFiniteFraction !== 1;
process.exit(failed ? 1 : 0);
