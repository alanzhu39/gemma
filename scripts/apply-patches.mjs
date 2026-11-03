#!/usr/bin/env node
/**
 * Applies (or verifies) every patch in patches/ against node_modules.
 *
 *   node scripts/apply-patches.mjs           # apply all patches
 *   node scripts/apply-patches.mjs --check   # verify they apply; change nothing
 *
 * Runs from `postinstall` so a clean `npm install` reproduces the patched tree.
 * See patches/README.md for how to create a patch.
 */
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const PATCH_DIR = path.join(ROOT, "patches");
const checkOnly = process.argv.includes("--check");

if (!fs.existsSync(PATCH_DIR)) {
	console.log("[patches] no patches/ directory; nothing to do");
	process.exit(0);
}

const patches = fs
	.readdirSync(PATCH_DIR)
	.filter((f) => f.endsWith(".patch"))
	.sort();

if (patches.length === 0) {
	console.log("[patches] no patches to apply");
	process.exit(0);
}

let failed = 0;
for (const name of patches) {
	const file = path.join(PATCH_DIR, name);
	const args = ["-p1", "--batch", "--forward"];
	if (checkOnly) args.push("--dry-run");
	args.push("-i", file);

	try {
		execFileSync("patch", args, { cwd: ROOT, stdio: "pipe" });
		console.log(`[patches] ${checkOnly ? "ok    " : "applied"} ${name}`);
	} catch (e) {
		// `--forward` makes an already-applied patch exit non-zero. Distinguish
		// that from a real failure so re-running postinstall stays quiet.
		const out = `${e.stdout ?? ""}${e.stderr ?? ""}`;
		if (/Reversed \(or previously applied\) patch detected|previously applied/i.test(out)) {
			console.log(`[patches] skipped ${name} (already applied)`);
			continue;
		}
		failed++;
		console.error(`[patches] FAILED ${name}`);
		console.error(out.trim().slice(0, 2000));
	}
}

if (failed > 0) {
	console.error(`\n[patches] ${failed} patch(es) failed.`);
	console.error("The pinned dependency version probably changed.");
	console.error("See patches/README.md to rebase the patch.");
	process.exit(1);
}
