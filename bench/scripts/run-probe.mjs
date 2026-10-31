/**
 * Runs one of the WebGPU probes in `bench/scripts/` inside headless Chromium.
 *
 * The probes are TypeScript modules that export `probe()` (or `probe2()`), and
 * they need a real GPU, a secure context, and the model weights. Until now each
 * one was driven by hand from a scratch script, which meant the surrounding
 * setup (flags, raised limits, lock, timing discipline) could differ between two
 * runs of the same probe. That is exactly how a 50 GB/s and a 121 GB/s figure
 * for one kernel both ended up being "right".
 *
 *   node bench/scripts/run-probe.mjs gemv-inmodel
 *   node bench/scripts/run-probe.mjs bandwidth-probe --json
 *
 * Takes the bench lock, because a probe that reads 640 MiB competes for the same
 * bandwidth as any benchmark running beside it.
 */
import { chromium } from "playwright";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { acquireLock } from "../lock.mjs";
import { chromeArgs, chromeEnv, getGpuDevices, isSoftwareRenderer } from "../chrome.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../..");

const argv = process.argv.slice(2);
const name = argv.find((a) => !a.startsWith("--"));
if (!name) {
	console.error("usage: node bench/scripts/run-probe.mjs <probe-name> [--json]");
	process.exit(2);
}
const asJson = argv.includes("--json");

const release = await acquireLock({
	onWait: (h) => console.log(`[probe] waiting for pid ${h.pid} to finish`),
});
process.on("SIGINT", () => {
	void release();
	process.exit(130);
});

const server = await createServer({
	root: ROOT,
	configFile: path.join(HERE, "../vite.config.ts"),
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
const page = await browser.newPage();
page.on("console", (m) => {
	const t = m.text();
	if (!t.startsWith("[vite]")) console.log(t);
});
page.on("pageerror", (e) => console.error("[page error]", e.message));

// A probe that produces a validation error must fail loudly. A silently skipped
// kernel is the failure mode this project has hit more than once.
await page.addInitScript(() => {
	window.__gpuErrors = [];
	const proto = GPUAdapter.prototype;
	const original = proto.requestDevice;
	proto.requestDevice = function (...args) {
		const p = original.apply(this, args);
		p.then((device) => {
			device.addEventListener("uncapturederror", (e) =>
				window.__gpuErrors.push(String(e.error?.message ?? e.error)),
			);
		}).catch(() => {});
		return p;
	};
});

await page.goto(`${base}/bench/index.html`, { waitUntil: "load", timeout: 120_000 });

const devices = await getGpuDevices(browser);
const renderer = devices[0] ?? "";
if (isSoftwareRenderer(renderer)) {
	console.log("[probe] !! software renderer: absolute numbers are CPU-bound, relative only");
}

let result;
let failed = false;
try {
	result = await page.evaluate(async (n) => {
		const mod = await import(`/bench/scripts/${n}.ts`);
		const fn = mod.probe ?? mod.probe2;
		if (!fn) throw new Error(`${n}.ts exports neither probe() nor probe2()`);
		return await fn();
	}, name);
	const gpuErrors = await page.evaluate(() => window.__gpuErrors ?? []);
	if (gpuErrors.length) {
		console.error(`[probe] ${gpuErrors.length} WebGPU error(s):`);
		for (const e of gpuErrors.slice(0, 10)) console.error(`  - ${String(e).slice(0, 300)}`);
		failed = true;
	}
} finally {
	await browser.close();
	await server.close();
	await release();
}

if (asJson) console.log(JSON.stringify(result, null, 2));
else if (Array.isArray(result)) console.table(result);
else console.log(JSON.stringify(result, null, 2));

process.exit(failed ? 1 : 0);
