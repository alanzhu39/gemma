/**
 * Correctness gate runner.
 *
 *   node bench/check.mjs                 # verify against bench/goldens/
 *   node bench/check.mjs --record        # write/refresh the golden
 *   node bench/check.mjs --tolerance 0.05
 *
 * Exits non-zero on failure so it can gate merges in CI or be run by agents
 * before they report an optimization as done.
 */
import { chromium } from "playwright";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { acquireLock } from "./lock.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const WEIGHTS = path.join(HERE, ".cache", "model.safetensors");
const GOLDEN_DIR = path.join(HERE, "goldens");

const argv = process.argv.slice(2);
const record = argv.includes("--record");
const tolIdx = argv.indexOf("--tolerance");
const tolerance = tolIdx >= 0 ? Number(argv[tolIdx + 1]) : undefined;

function chromeEnv() {
	const a = path.join(HERE, ".cache", "chrome-deps", "usr", "lib", "x86_64-linux-gnu");
	const b = path.join(HERE, ".cache", "chrome-deps", "lib", "x86_64-linux-gnu");
	const parts = [a, b].filter((p) => fs.existsSync(p));
	if (!parts.length) return {};
	const existing = process.env.LD_LIBRARY_PATH ? `:${process.env.LD_LIBRARY_PATH}` : "";
	return { LD_LIBRARY_PATH: parts.join(":") + existing };
}

if (!fs.existsSync(WEIGHTS)) {
	console.error(`Missing weights. Run: npm run bench:fetch`);
	process.exit(1);
}

// The correctness gate runs the model too; serialise it against benchmarks so
// we are not measuring each other.
const release = acquireLock({
	onWait: (h) => console.log(`[check] waiting for pid ${h.pid} to finish`),
});
process.on("SIGINT", () => (release(), process.exit(130)));

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
	args: [
		"--no-sandbox",
		"--enable-unsafe-webgpu",
		"--enable-unsafe-swiftshader",
		"--use-angle=swiftshader",
		"--ignore-gpu-blocklist",
		"--disable-dev-shm-usage",
	],
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[page error]", e.message));

await page.goto(`${base}/bench/index.html`, { waitUntil: "load", timeout: 120_000 });

await fsp.mkdir(GOLDEN_DIR, { recursive: true });

let exitCode = 0;
try {
	if (record) {
		const golden = await page.evaluate(async () => {
			const m = await import("/bench/check.ts");
			const adapter = await navigator.gpu.requestAdapter();
			const precision = adapter.features.has("shader-f16") ? "f16" : "f32";
			return m.computeLogits(precision);
		});
		const file = path.join(GOLDEN_DIR, `${golden.precision}.json`);
		await fsp.writeFile(file, JSON.stringify(golden, null, 2) + "\n");
		console.log(`recorded ${path.relative(process.cwd(), file)}`);
		console.log(`  argmax=${golden.argmax} latent=${JSON.stringify(golden.latentShape)}`);
		console.log(`  logits[0..4]=${golden.logits.slice(0, 5).map((x) => x.toFixed(4)).join(", ")}`);
	} else {
		const result = await page.evaluate(
			async ({ tolerance }) => {
				const m = await import("/bench/check.ts");
				const adapter = await navigator.gpu.requestAdapter();
				const precision = adapter.features.has("shader-f16") ? "f16" : "f32";
				let golden = null;
				try {
					const res = await fetch(`/bench/goldens/${precision}.json`);
					if (res.ok) golden = await res.json();
				} catch {
					golden = null;
				}
				return m.runCheck(golden, { tolerance });
			},
			{ tolerance },
		);

		console.log(`[check] precision: ${result.precision}`);
		console.log(`[check] status:    ${result.status.toUpperCase()}`);
		console.log(`[check] ${result.message}`);

		if (result.status === "fail") exitCode = 1;
		if (result.status === "no-golden") {
			console.log(`\nNo golden recorded for ${result.precision}.`);
			console.log(`Run: npm run check:golden`);
			exitCode = 2;
		}
		if (result.status === "skipped") {
			console.log("Skipping numeric comparison (precision mismatch).");
		}
	}
} finally {
	await browser.close();
	await server.close();
	release();
}

process.exit(exitCode);
