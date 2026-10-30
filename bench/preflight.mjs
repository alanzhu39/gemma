#!/usr/bin/env node
/**
 * Reports whether this machine can run the benchmark meaningfully.
 *
 * Checks, in order: Chromium launch, WebGPU adapter, whether the adapter is a
 * hardware device, and whether it supports `shader-f16` (the FP16 model needs
 * it; without it the harness has to fall back to float32).
 *
 *   node bench/preflight.mjs
 *
 * Exit code 0 if a hardware adapter with shader-f16 is available, 1 otherwise.
 */
import { chromium } from "playwright";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromeArgs, chromeEnv } from "./chrome.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));


const browser = await chromium.launch({
	headless: true,
	// Shared with the benchmark and the gate, so preflight can never report a
	// different renderer than the one the real runs actually use.
	env: { ...process.env, ...chromeEnv() },
	args: chromeArgs(),
});

// WebGPU requires a secure context. `about:blank` and `data:` URLs are not
// secure contexts, so serve the probe from localhost like the real harness.
const server = http.createServer((req, res) => {
	res.setHeader("content-type", "text/html");
	res.end("<html><body>preflight</body></html>");
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const origin = `http://127.0.0.1:${server.address().port}/`;

const page = await browser.newPage();
await page.goto(origin);

// The adapter's own `info` is masked in this Chromium build, so ask the browser
// which renderer it actually chose. A CPU fallback (SwiftShader/llvmpipe) will
// otherwise look exactly like a hardware GPU.
const devices = await browser
	.newBrowserCDPSession()
	.then((s) => s.send("SystemInfo.getInfo"))
	.then((info) => (info.gpu?.devices ?? []).map((d) => d.deviceString ?? ""))
	.catch(() => []);

const gpu = await page.evaluate(async () => {
	if (!navigator.gpu) return { available: false };
	const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
	if (!adapter) return { available: true, adapter: false };
	const info = { ...(adapter.info ?? {}) };
	return {
		available: true,
		adapter: true,
		info,
		fallback: adapter.isFallbackAdapter === true,
		f16: adapter.features.has("shader-f16"),
		limits: {
			maxBufferSize: adapter.limits.maxBufferSize,
			maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
			maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
		},
	};
});

await browser.close();
server.close();

const rendererString = devices[0] ?? "";
const arch = String(gpu.info?.architecture ?? "").toLowerCase();
const desc = String(gpu.info?.description ?? "").toLowerCase();
const software =
	gpu.fallback === true ||
	/swiftshader|llvmpipe|software/i.test(rendererString) ||
	arch.includes("swiftshader") ||
	desc.includes("swiftshader") ||
	desc.includes("llvmpipe");

console.log("WebGPU preflight");
console.log("----------------");
console.log(`navigator.gpu:      ${gpu.available ? "present" : "MISSING"}`);
if (gpu.available) {
	console.log(`adapter:            ${gpu.adapter ? "obtained" : "NONE"}`);
	if (gpu.adapter) {
		console.log(`description:        ${gpu.info?.description || "(not exposed)"}`);
		console.log(`renderer (CDP):     ${rendererString || "(unknown)"}`);
		console.log(`architecture:       ${gpu.info?.architecture || "(not exposed)"}`);
		console.log(`device:             ${gpu.info?.device || "(not exposed)"}`);
		console.log(`vendor:             ${gpu.info?.vendor || "(not exposed)"}`);
		console.log(`software renderer:  ${software ? "YES" : "no"}`);
		console.log(`shader-f16:         ${gpu.f16 ? "yes" : "NO"}`);
		console.log(`maxBufferSize:      ${(gpu.limits.maxBufferSize / 1e9).toFixed(2)} GB`);
		console.log(`maxStorageBinding:  ${(gpu.limits.maxStorageBufferBindingSize / 1e9).toFixed(2)} GB`);
		console.log(`workgroup storage:  ${gpu.limits.maxComputeWorkgroupStorageSize} bytes`);
	}
}

console.log("");

if (!gpu.available || !gpu.adapter) {
	console.log("RESULT: unusable — no WebGPU adapter.");
	process.exit(1);
}

if (software) {
	console.log("RESULT: CPU fallback only.");
	console.log("");
	console.log("No hardware GPU is reachable from this process, so the adapter is a");
	console.log("software rasteriser. Benchmarks will run (the harness widens weights to");
	console.log("float32 when shader-f16 is absent), but absolute numbers describe CPU");
	console.log("performance, not GPU performance.");
	console.log("");
	if (fs.existsSync("/dev/dri/renderD128")) {
		const st = fs.statSync("/dev/dri/renderD128");
		const mode = (st.mode & 0o777).toString(8);
		console.log(`/dev/dri/renderD128 exists (mode ${mode}, owner ${st.uid}:${st.gid}).`);
		console.log("Typical fix on a Debian/Ubuntu host:");
		console.log("  sudo usermod -aG render $USER   # then log out and back in");
		console.log("  sudo apt install -y mesa-vulkan-drivers libvulkan1");
	} else {
		console.log("No /dev/dri/renderD128 present: this machine has no accessible GPU.");
	}
	process.exit(1);
}

if (!gpu.f16) {
	console.log("RESULT: hardware GPU, but no shader-f16.");
	console.log("The FP16 weights cannot run natively; the harness will use float32,");
	console.log("which is slower and not representative of the deployed build.");
	process.exit(1);
}

console.log("RESULT: ready — hardware GPU with shader-f16.");
process.exit(0);
