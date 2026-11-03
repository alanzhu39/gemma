/**
 * Driver for bench/scripts/probe-final-ab.ts.
 *
 *   node bench/scripts/probe-final-ab.mjs [--ctx 1024]
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
const steps = Number(opt("--steps", 2));
const trials = Number(opt("--trials", 3));

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
		const { probe } = await import("/bench/scripts/probe-final-ab.ts");
		return await probe();
	}, { ctx });
} finally {
	await browser.close();
	await server.close();
	await release();
}

const fs = await import("node:fs");
const file = path.join(HERE, "..", "results", `final-ab-${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
fs.writeFileSync(file, JSON.stringify({ renderer, ...result }, null, 2) + "\n");

const A = result.summary.A;
console.log(`\nrenderer: ${renderer}`);
console.log(`interleaved A/B, ctx ${result.ctx}, f32, ${result.trials} trials x ${result.stepsPerTrial} steps/arm\n`);
console.log("  arm   what                                                   median      min      p90");
for (const [k, v] of Object.entries(result.summary)) {
  console.log(`  ${k}   ${result.arms[k].padEnd(56)} ${String(v.median).padStart(6)} ms ${String(v.min).padStart(7)} ${String(v.p90).padStart(7)}`);
}
console.log(`\n  saved vs as-shipped:  B ${result.savedVsA.B} ms   C ${result.savedVsA.C} ms   D ${result.savedVsA.D} ms`);
console.log(`  speedup vs as-shipped: B ${(A.median / result.summary.B.median).toFixed(2)}x  C ${(A.median / result.summary.C.median).toFixed(2)}x  D ${(A.median / result.summary.D.median).toFixed(2)}x`);
console.log(`\n  verify (each arm vs A, same position):`);
for (const [k, v] of Object.entries(result.verify)) {
  console.log(`    ${k}: ${typeof v === "string" ? v : `max |diff| ${v.maxAbsDiff} (relative ${v.relative}), argmax match ${v.argmaxMatch}`}`);
}
console.log(`  submit batching: ${JSON.stringify(result.patchStats)}`);
console.log(`\nraw: ${path.relative(ROOT, file)}`);
