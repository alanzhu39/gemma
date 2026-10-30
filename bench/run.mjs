/**
 * Benchmark driver.
 *
 * Serves bench/ over HTTP (WebGPU requires a secure context, and the model
 * weights are large enough that we never want them inlined), opens a headless
 * Chromium page, runs the in-page benchmark, and writes results to
 * bench/results/.
 *
 *   node bench/run.mjs                  # full suite
 *   node bench/run.mjs --only decode    # subset
 *   node bench/run.mjs --repeats 7      # more samples
 */
import { chromium } from "playwright";
import http from "node:http";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer as createViteServer } from "vite";
import { acquireLock } from "./lock.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const WEIGHTS = path.join(HERE, ".cache", "model.safetensors");
const RESULTS_DIR = path.join(HERE, "results");

function parseArgs(argv) {
	const out = { repeats: 5, warmup: 2, only: [], label: null, json: false, software: false, noWait: false };
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (a === "--repeats") out.repeats = Number(argv[++i]);
		else if (a === "--warmup") out.warmup = Number(argv[++i]);
		else if (a === "--only") out.only = String(argv[++i]).split(",");
		else if (a === "--label") out.label = argv[++i];
		else if (a === "--json") out.json = true;
		else if (a === "--software") out.software = true;
		else if (a === "--no-wait") out.noWait = true;
	}
	return out;
}

/**
 * Chromium flags for WebGPU.
 *
 * Deliberately does NOT pin an ANGLE backend by default: pinning one is what
 * would hide a real GPU on a normal machine. `--software` forces the CPU
 * fallback, which is only for smoke-testing the harness itself.
 */
function chromeArgs({ software = false } = {}) {
	const base = [
		"--no-sandbox",
		// WebGPU is not on by default in headless Chromium.
		"--enable-unsafe-webgpu",
		"--ignore-gpu-blocklist",
		"--disable-dev-shm-usage",
	];
	return software ? [...base, "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] : base;
}

/**
 * Chrome needs a pile of system libraries that are not installed here and that
 * we cannot install without root. bench/setup-chrome-deps.sh downloads them as
 * .deb files and unpacks them under bench/.cache/chrome-deps, so we just point
 * the loader at them.
 */
function chromeEnv() {
	const deps = path.join(HERE, ".cache", "chrome-deps", "usr", "lib", "x86_64-linux-gnu");
	const deps2 = path.join(HERE, ".cache", "chrome-deps", "lib", "x86_64-linux-gnu");
	const parts = [deps, deps2].filter((p) => fs.existsSync(p));
	if (!parts.length) return {};
	const existing = process.env.LD_LIBRARY_PATH ? `:${process.env.LD_LIBRARY_PATH}` : "";
	return { LD_LIBRARY_PATH: parts.join(":") + existing };
}

async function startVite() {
	const server = await createViteServer({
		root: ROOT,
		configFile: path.join(HERE, "vite.config.ts"),
		server: { port: 0, strictPort: false },
		logLevel: "error",
	});
	await server.listen();
	// Vite may bind IPv6-only (::1) and picks its own port. Trust the resolved
	// URL instead of assuming 127.0.0.1:5173.
	const resolved = server.resolvedUrls?.local?.[0];
	if (!resolved) throw new Error("vite did not report a local URL");
	const base = resolved.replace(/\/$/, "");
	const port = Number(new URL(base).port);

	return { server, port, base };
}

async function main() {
	const args = parseArgs(process.argv.slice(2));

	if (!fs.existsSync(WEIGHTS)) {
		console.error(`Missing weights at ${WEIGHTS}\nRun: npm run bench:fetch`);
		process.exit(1);
	}

	// Serialise runs: agents share this machine, and two concurrent benchmarks
	// contend for the same CPU/GPU and invalidate both sets of numbers.
	const release = acquireLock({
		wait: !args.noWait,
		onWait: (holder) =>
			console.log(
				`[bench] waiting for another run to finish (pid ${holder.pid}, started ${new Date(holder.startedAt).toISOString()})`,
			),
	});
	// Release on every exit path, including Ctrl-C.
	for (const sig of ["SIGINT", "SIGTERM"]) process.on(sig, () => (release(), process.exit(130)));

	const { server, base } = await startVite();
	const url = `${base}/bench/index.html`;

	const browser = await chromium.launch({
		headless: true,
		env: { ...process.env, ...chromeEnv() },
		args: chromeArgs({ software: args.software }),
	});

	// Chromium does not expose GPUAdapter.info in this build, so ask the browser
	// itself what renderer it is using. This is the only reliable way to tell a
	// hardware GPU from a CPU fallback here.
	const gpuDevices = await browser
		.newBrowserCDPSession()
		.then((s) => s.send("SystemInfo.getInfo"))
		.then((info) => (info.gpu?.devices ?? []).map((d) => d.deviceString ?? ""))
		.catch(() => []);
	const rendererString = gpuDevices[0] ?? "";

	const page = await browser.newPage();
	page.on("console", (m) => {
		const t = m.text();
		if (t.startsWith("[bench]")) console.log(t);
	});
	page.on("pageerror", (e) => console.error("[page error]", e.message));

	await page.goto(url, { waitUntil: "load", timeout: 120_000 });

	const result = await page.evaluate(
		async ({ repeats, warmup, only }) => {
			const { runBenchmarks } = await import("/bench/bench.ts");
			const workloads = await (await fetch("/bench/workloads.json")).json();
			return await runBenchmarks(workloads, {
				repeats,
				warmup,
				only,
				onProgress: (m) => console.log(`[bench] ${m}`),
			});
		},
		{ repeats: args.repeats, warmup: args.warmup, only: args.only },
	);

	await browser.close();
	await server.close();
	release();

	// Trust the browser's own renderer string over the adapter feature heuristics
	// in the page: those cannot see through a masked adapter.
	const sw = /swiftshader|llvmpipe|software/i.test(rendererString);
	result.sysInfo = {
		...result.sysInfo,
		rendererString,
		gpuDevices,
		isSoftwareRenderer: sw,
	};

	await fsp.mkdir(RESULTS_DIR, { recursive: true });
	const stamp = new Date().toISOString().replace(/[:.]/g, "-");
	const label = args.label ? `${args.label}-` : "";
	const file = path.join(RESULTS_DIR, `${label}${stamp}.json`);
	await fsp.writeFile(file, JSON.stringify(result, null, 2) + "\n");

	// Also keep a "latest" pointer so tooling and agents do not have to guess.
	await fsp.writeFile(
		path.join(RESULTS_DIR, "latest.json"),
		JSON.stringify({ file: path.basename(file), ...result }, null, 2) + "\n",
	);

	printSummary(result, file);

	if (!args.json) {
		console.log(`\nWrote ${path.relative(process.cwd(), file)}`);
	}
}

function printSummary(result, file) {
	const { samples, sysInfo } = result;
	console.log("\n=== Gemma 3 inference benchmark ===");
	console.log(`renderer: ${sysInfo.rendererString || JSON.stringify(sysInfo.adapterInfo) || "(unknown)"}`);
	if (sysInfo.isSoftwareRenderer) {
		console.log("!! WARNING: software renderer (SwiftShader). Numbers are CPU-bound and");
		console.log("!! will not reflect real-GPU performance. Use for relative comparisons only.");
	}
	console.log("");
	const kindHeader = { prefill: "Prefill", decode: "Decode", batch: "Batch" };
	for (const kind of ["prefill", "decode", "batch"]) {
		const rows = samples.filter((s) => s.kind === kind);
		if (!rows.length) continue;
		console.log(`${kindHeader[kind]}`);
		for (const s of rows) {
			if (kind === "prefill") {
				console.log(`  ${s.workload.padEnd(16)} ${s.tps.toFixed(1).padStart(9)} tok/s   (${s.ms.toFixed(1)} ms)`);
			} else if (kind === "decode") {
				console.log(`  ${s.workload.padEnd(16)} ${s.tps.toFixed(2).padStart(9)} tok/s   (${s.ms.toFixed(2)} ms/token)`);
			} else {
				const d = s.detail ?? {};
				const b = d.batchedSupported ? ` | batched ${d.batchedTotalTps.toFixed(2)} tok/s total` : "";
				console.log(`  ${s.workload.padEnd(16)} ${s.tps.toFixed(2).padStart(9)} tok/s/seq   (${(d.totalTps ?? 0).toFixed(2)} tok/s total)${b}`);
			}
		}
		console.log("");
	}
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
