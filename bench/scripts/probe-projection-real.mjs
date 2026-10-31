/**
 * Driver for bench/scripts/probe-projection-real.ts.
 *
 *   node bench/scripts/probe-projection-real.mjs [--ctx 1024]
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
		const { probe } = await import("/bench/scripts/probe-projection-real.ts");
		return await probe();
	}, { ctx });
} finally {
	await browser.close();
	await server.close();
	await release();
}

const fs = await import("node:fs");
const file = path.join(HERE, "..", "results", `projection-real-${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
fs.writeFileSync(file, JSON.stringify({ renderer, ...result }, null, 2) + "\n");

console.log(`\nrenderer: ${renderer}`);
console.log(`weights ${result.weights.shape.join(" x ")} ${result.weights.mib} MiB; input is a real host buffer\n`);
console.log(`  strided     ${String(result.strided.median).padStart(6)} ms  ${String(result.strided.gbps).padStart(5)} GB/s   ${result.strided.note}`);
console.log(`              ${result.strided.values.join(" ")}`);
console.log(`  contiguous  ${String(result.contiguous.median).padStart(6)} ms  ${String(result.contiguous.gbps).padStart(5)} GB/s   ${result.contiguous.note}`);
console.log(`              ${result.contiguous.values.join(" ")}`);
console.log(`  ratio ${result.ratio}x`);
const v = result.verify;
console.log(`\nverify: max |diff| ${v.maxAbsDiff} on logit scale ${v.logitScale} (relative ${v.relative}), argmax match ${v.argmaxMatch} -> ${v.argmaxMatch && v.relative <= 1e-3 ? "OK" : "MISMATCH"}`);
console.log(`\nraw: ${path.relative(ROOT, file)}`);
