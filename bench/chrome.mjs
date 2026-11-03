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
	// Chromium picks two graphics backends independently: ANGLE for OpenGL and
	// Dawn for WebGPU. Pointing only ANGLE at Vulkan leaves WebGPU on SwiftShader
	// while the renderer string happily reports the GPU - which is exactly how a
	// full day was lost here.
	//
	// `VulkanFromANGLE` is the flag that actually moves Dawn; without it
	// `maxComputeWorkgroupStorageSize` stays at SwiftShader's 32768 and every
	// kernel runs on the CPU. With it, this machine reports 49152 and a 4.29 GB
	// buffer cap. Verified by sweeping ten flag combinations and reading those
	// limits, which are a far more reliable signal than the renderer string.
	return [
		...base,
		"--enable-features=Vulkan,VulkanFromANGLE",
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
	const env = {};

	const a = path.join(HERE, ".cache", "chrome-deps", "usr", "lib", "x86_64-linux-gnu");
	const b = path.join(HERE, ".cache", "chrome-deps", "lib", "x86_64-linux-gnu");
	const parts = [a, b].filter((p) => fs.existsSync(p));
	if (parts.length) {
		const existing = process.env.LD_LIBRARY_PATH ? `:${process.env.LD_LIBRARY_PATH}` : "";
		env.LD_LIBRARY_PATH = parts.join(":") + existing;
	}

	// NVK refuses to probe Pascal unless this opt-in is set. It is only read by
	// Mesa's NVK, so it is harmless (and ignored) with the proprietary driver.
	// Set unconditionally: an earlier version returned early when the Chrome
	// dependency bundle was absent, silently dropping it and reporting a software
	// renderer on a working GPU.
	env.NVK_I_WANT_A_BROKEN_VULKAN_DRIVER = "1";

	// Prefer the proprietary NVIDIA ICD when present. Mesa's nouveau/lvp ICDs
	// also exist on this box, and the loader would otherwise be free to pick the
	// software one. Do not filter to nouveau specifically - that pinned the
	// harness to a driver that is no longer in use.
	const nvidiaIcd = "/usr/share/vulkan/icd.d/nvidia_icd.json";
	if (fs.existsSync(nvidiaIcd)) {
		env.VK_ICD_FILENAMES = nvidiaIcd;
	}

	return env;
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
