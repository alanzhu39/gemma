/**
 * Chromium launch configuration shared by the benchmark and the correctness
 * gate.
 *
 * These must agree. If the gate pinned SwiftShader while the benchmark ran on
 * the real GPU, the f16 path would never be checked, and the gate would pass
 * while production broke.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

/**
 * Chromium flags for WebGPU.
 *
 * Deliberately does NOT pin an ANGLE backend by default: pinning one is what
 * would hide a real GPU on a normal machine. `--software` forces the CPU
 * fallback, which is only for smoke-testing the harness itself.
 */
export function chromeArgs({ software = false } = {}) {
	const base = [
		"--no-sandbox",
		// WebGPU is not on by default in headless Chromium.
		"--enable-unsafe-webgpu",
		"--ignore-gpu-blocklist",
		"--disable-dev-shm-usage",
	];
	if (software) {
		return [...base, "--use-angle=swiftshader", "--enable-unsafe-swiftshader"];
	}
	// Without these, headless Chromium selects ANGLE's SwiftShader (CPU) backend
	// even when a driver is installed and the device node is accessible. Measured
	// on this machine: default -> SwiftShader, with these flags -> NVK on the
	// GTX 1060. They are harmless on a machine with no Vulkan driver: Chromium
	// falls back on its own.
	return [
		...base,
		"--enable-features=Vulkan",
		"--use-angle=vulkan",
		"--use-vulkan=native",
	];
}

/**
 * Chromium needs a pile of system libraries that are not installed here and
 * that we cannot install without root. bench/setup-chrome-deps.sh downloads
 * them as .deb files and unpacks them under bench/.cache/chrome-deps, so we
 * just point the loader at them.
 */
export function chromeEnv() {
	const a = path.join(HERE, ".cache", "chrome-deps", "usr", "lib", "x86_64-linux-gnu");
	const b = path.join(HERE, ".cache", "chrome-deps", "lib", "x86_64-linux-gnu");
	const parts = [a, b].filter((p) => fs.existsSync(p));
	if (!parts.length) return {};
	const existing = process.env.LD_LIBRARY_PATH ? `:${process.env.LD_LIBRARY_PATH}` : "";
	// NVK refuses to probe Pascal unless this opt-in is set; without it the
	// driver reports zero devices and Chromium silently uses SwiftShader.
	return {
		LD_LIBRARY_PATH: parts.join(":") + existing,
		NVK_I_WANT_A_BROKEN_VULKAN_DRIVER: "1",
	};
}

/**
 * Ask the browser which renderer it actually chose.
 *
 * The page cannot tell reliably: Chromium masks `GPUAdapter.info`, and
 * `isFallbackAdapter` is false for SwiftShader, so a CPU rasteriser is
 * indistinguishable from a GPU inside JavaScript.
 */
export async function getGpuDevices(browser) {
	const session = await browser.newBrowserCDPSession();
	try {
		const info = await session.send("SystemInfo.getInfo");
		return (info.gpu?.devices ?? []).map((d) => d.deviceString ?? "");
	} catch {
		return [];
	} finally {
		await session.detach().catch(() => {});
	}
}

export function isSoftwareRenderer(rendererString) {
	return /swiftshader|llvmpipe|software/i.test(rendererString ?? "");
}
