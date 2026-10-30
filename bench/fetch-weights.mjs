#!/usr/bin/env node
/**
 * Downloads the FP16 weights into bench/.cache so benchmark runs never hit the
 * network. Idempotent: skips the download when the file is already complete.
 */
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CACHE = path.join(HERE, ".cache");
const OUT = path.join(CACHE, "model.safetensors");
const URL_ = "https://huggingface.co/alanzhu39/gemma-3-270m-it-f16/resolve/main/model.safetensors";
const EXPECTED_BYTES = 512 * 1024 * 1024; // ~512 MB

await fsp.mkdir(CACHE, { recursive: true });

if (fs.existsSync(OUT) && fs.statSync(OUT).size >= EXPECTED_BYTES * 0.95) {
	console.log(`weights already present: ${OUT}`);
	process.exit(0);
}

console.log(`downloading ${URL_}`);
const res = await fetch(URL_);
if (!res.ok) throw new Error(`download failed: ${res.status}`);
const total = Number(res.headers.get("content-length") ?? 0);
const tmp = `${OUT}.partial`;
const file = fs.createWriteStream(tmp);
let seen = 0;
for await (const chunk of res.body) {
	seen += chunk.length;
	file.write(chunk);
	if (total) {
		process.stdout.write(`\r${((seen / total) * 100).toFixed(1)}% (${(seen / 1e6).toFixed(0)} MB)`);
	}
}
await new Promise((r) => file.end(r));
await fsp.rename(tmp, OUT);
console.log(`\nwrote ${OUT}`);
