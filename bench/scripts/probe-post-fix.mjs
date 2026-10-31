/**
 * Driver for bench/scripts/probe-post-fix.ts.
 *
 *   node bench/scripts/probe-post-fix.mjs [--ctx 1024]
 *
 * Prints, per region, how much GPU time went into kernels and how much into
 * gaps between them. Takes bench/lock.mjs's lock for the whole run.
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
const opt = (n, d) => (argv.includes(n) ? argv[argv.indexOf(n) + 1] : d);
const ctx = Number(opt("--ctx", 1024));
const steps = Number(opt("--steps", 8));

const release = await acquireLock({
	onWait: (h) => console.log(`[gaps] waiting for pid ${h.pid}`),
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
const renderer = (await getGpuDevices(browser))[0] ?? "";
if (isSoftwareRenderer(renderer)) {
	console.error(`[gaps] software renderer, refusing: ${renderer}`);
	await browser.close();
	await server.close();
	await release();
	process.exit(1);
}
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[page error]", e.message));
await page.goto(`${base}/bench/index.html`, { waitUntil: "load", timeout: 120_000 });

let result;
try {
	result = await page.evaluate(async ({ ctx }) => {
		globalThis.__ctx = ctx;
		const { probe } = await import("/bench/scripts/probe-post-fix.ts");
		return await probe();
	}, { ctx });
} finally {
	await browser.close();
	await server.close();
	await release();
}

const fs = await import("node:fs");
const file = path.join(HERE, "..", "results", `post-fix-${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
fs.writeFileSync(file, JSON.stringify({ renderer, ...result }, null, 2) + "\n");

const m = result.medians;
console.log(`\nrenderer: ${renderer}`);
console.log(`post-fix production path, ctx ${result.ctx}, f32, ${result.steps} steps\n`);
console.log(`  BENCH region (model + projection + sync)  ${String(m.benchMs).padStart(7)} ms/token  => ${result.tokPerSec} tok/s`);
console.log(`  + real sampler                            ${String(m.samplingMs).padStart(7)} ms      => ${m.genMs} ms/token in a loop`);
console.log(`\n  GPU kernel time                           ${String(m.kernelMs).padStart(7)} ms  (${(m.kernelMs / m.benchMs * 100) | 0}%)`);
console.log(`    projection kernel (${m.kernels} kernels total)      ${String(m.projectionKernelMs).padStart(7)} ms  (${(m.projectionKernelMs / m.benchMs * 100) | 0}%)`);
console.log(`    all other kernels                       ${String(m.otherKernelMs).padStart(7)} ms  (${(m.otherKernelMs / m.benchMs * 100) | 0}%)`);
console.log(`  GPU idle between kernels                  ${String(m.gapMs).padStart(7)} ms  (${(m.gapMs / m.benchMs * 100) | 0}%)`);
console.log(`  host issue + readback + queue drain       ${String(m.hostResidualMs).padStart(7)} ms  (${(m.hostResidualMs / m.benchMs * 100) | 0}%)`);
console.log(`\n  per-step bench ms: ${result.benches.join(" ")}`);
console.log(`\n  kernel totals by name (median step):`);
for (const k of result.kernelsByName) console.log(`    ${k.name.padEnd(24)} ${String(k.ms).padStart(7)} ms  x${k.count}`);
console.log(`\nraw: ${path.relative(ROOT, file)}`);
