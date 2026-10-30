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
import { chromeArgs, chromeEnv, getGpuDevices, isSoftwareRenderer } from "./chrome.mjs";
import { execFileSync } from "node:child_process";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const WEIGHTS = path.join(HERE, ".cache", "model.safetensors");
const GOLDEN_DIR = path.join(HERE, "goldens");

const argv = process.argv.slice(2);
const record = argv.includes("--record");
// --record-missing only writes references that do not exist yet, so adding a new
// check cannot silently rewrite an existing reference (which would erase the
// evidence that an optimization changed the output).
const recordMissing = argv.includes("--record-missing");
const force = argv.includes("--force");

function git(cmd) {
	return execFileSync("git", cmd, { cwd: ROOT, encoding: "utf8" }).trim();
}

if (record || recordMissing) {
	// A reference recorded from a tree with model changes captures those changes
	// as "correct", which defeats the entire gate.
	const dirty = git([
		"status",
		"--porcelain",
		"--",
		"src/",
		"bench/check.ts",
		"bench/f32-model.ts",
		"patches/",
		"node_modules/@jax-js",
	]);
	if (dirty && !force) {
		console.error("Refusing to record a golden: the model or check code has uncommitted changes.");
		console.error(dirty);
		console.error("\nCommit them first, so the reference is tied to a revision.");
		console.error("Override with --force only if you are certain this is the intended behaviour.");
		process.exit(1);
	}
}
const tolIdx = argv.indexOf("--tolerance");
const tolerance = tolIdx >= 0 ? Number(argv[tolIdx + 1]) : undefined;

if (!fs.existsSync(WEIGHTS)) {
	console.error(`Missing weights. Run: npm run bench:fetch`);
	process.exit(1);
}

// The correctness gate runs the model too; serialise it against benchmarks so
// we are not measuring each other.
const release = await acquireLock({
	onWait: (h) => console.log(`[check] waiting for pid ${h.pid} to finish`),
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
	// Same flags as the benchmark, so the gate checks the path that was measured.
	args: chromeArgs({ software: argv.includes("--software") }),
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[page error]", e.message));

// The gate must exercise the same code path the benchmark measures. If the
// renderer here differs from what a benchmark run would use, say so loudly
// rather than silently checking the wrong precision.
const gpuDevices = await getGpuDevices(browser);
const rendererString = gpuDevices[0] ?? "";
const software = isSoftwareRenderer(rendererString);
console.log(`[check] renderer: ${rendererString || "(unknown)"}`);
if (software) {
	console.log("[check] software renderer: checking the f32 path only");
}

await page.goto(`${base}/bench/index.html`, { waitUntil: "load", timeout: 120_000 });

await fsp.mkdir(GOLDEN_DIR, { recursive: true });

// If the page navigates or reloads mid-run, page.evaluate rejects with a
// destroyed-context error and the run is lost after minutes of work. Retry once
// from a fresh page rather than failing on a transient.
const evaluate = async (fn, arg) => {
	for (let attempt = 1; ; attempt++) {
		try {
			return await page.evaluate(fn, arg);
		} catch (e) {
			const msg = String(e?.message ?? e);
			if (attempt < 2 && /context was destroyed|navigation/i.test(msg)) {
				console.warn("[check] page navigated mid-run; reloading and retrying once");
				await page.goto(`${base}/bench/index.html`, { waitUntil: "load", timeout: 120_000 });
				continue;
			}
			throw e;
		}
	}
};

let exitCode = 0;
try {
	if (record) {
		// Two references: a single prefill pass, and a decode trajectory long
		// enough to wrap the sliding window and grow the global cache. The decode
		// one covers the code paths the optimizations actually change.
		const { prefill, decode } = await evaluate(async () => {
			const m = await import("/bench/check.ts");
			const adapter = await navigator.gpu.requestAdapter();
			const precision = adapter.features.has("shader-f16") ? "f16" : "f32";
			return {
				prefill: await m.computeLogits(precision),
				decode: await m.computeDecodeLogits(precision),
			};
		});

		const provenance = {
			gitRev: git(["rev-parse", "HEAD"]),
			branch: git(["rev-parse", "--abbrev-ref", "HEAD"]),
			recordedAt: new Date().toISOString(),
			renderer: rendererString,
			flags: chromeArgs({ software: argv.includes("--software") }),
		};

		const write = async (file, doc, label) => {
			if (recordMissing && fs.existsSync(file)) {
				console.log(`keeping existing ${path.relative(process.cwd(), file)}`);
				return;
			}
			await fsp.writeFile(file, JSON.stringify({ ...doc, provenance }, null, 2) + "\n");
			console.log(`recorded ${path.relative(process.cwd(), file)} (${label})`);
		};

		const file = path.join(GOLDEN_DIR, `${prefill.precision}.json`);
		await write(file, prefill, `argmax=${prefill.argmax}`);
		const df = path.join(GOLDEN_DIR, `${prefill.precision}-decode.json`);
		await write(df, decode, `ctx=${decode.contextLen} argmax=${decode.steps.map((s) => s.argmax).join(",")}`);
	} else {
		const { prefill, decode } = await evaluate(
			async ({ tolerance }) => {
				const m = await import("/bench/check.ts");
				const adapter = await navigator.gpu.requestAdapter();
				const precision = adapter.features.has("shader-f16") ? "f16" : "f32";
				const load = async (name) => {
					try {
						const res = await fetch(`/bench/goldens/${precision}${name}.json`);
						return res.ok ? await res.json() : null;
					} catch {
						return null;
					}
				};
				return {
					prefill: await m.runCheck(await load(""), { tolerance }),
					decode: await m.runDecodeCheck(await load("-decode"), { tolerance }),
				};
			},
			{ tolerance },
		);

		for (const [label, result] of [
			["prefill", prefill],
			["decode ", decode],
		]) {
			console.log(`[check] ${label} precision: ${result.precision}`);
			console.log(`[check] ${label} status:    ${result.status.toUpperCase()}`);
			console.log(`[check] ${label} ${result.message}`);
			if (result.status === "fail") exitCode = 1;
			if ((result.status === "no-golden" || result.status === "stale-golden") && exitCode === 0) {
				exitCode = 2;
			}
			if (result.status === "skipped") {
				console.log(`[check] ${label} skipped (precision mismatch)`);
			}
		}
		if (exitCode === 2) {
			console.log("\nMissing golden. Record one with: npm run bench:golden");
		}
	}
} finally {
	await browser.close();
	await server.close();
	await release();
}

process.exit(exitCode);
