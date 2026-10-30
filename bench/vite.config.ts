import { defineConfig } from "vite";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const WEIGHTS = path.join(HERE, ".cache", "model.safetensors");

/**
 * Vite config for the benchmark harness only.
 *
 * Deliberately does NOT load the SvelteKit plugin: the benchmark imports the
 * engine directly (plain TypeScript), and SvelteKit's dev server restricts
 * requests to its own app directories, which would block /bench/*.
 */
export default defineConfig({
	root: ROOT,
	appType: "spa",
	logLevel: "error",
	server: {
		port: 0,
		strictPort: false,
		fs: { strict: false, allow: [ROOT] },
	},
	optimizeDeps: {
		// Large, already-ESM dependencies: prebundling them just slows startup.
		exclude: ["@jax-js/jax", "@jax-js/loaders"],
	},
	plugins: [
		{
			// Stream the model weights from bench/.cache. Registered through a
			// plugin so it sits inside Vite's middleware stack; adding it after
			// `server.listen()` is too late and Vite's SPA fallback answers first.
			name: "bench-serve-weights",
			configureServer(server) {
				server.middlewares.use("/weights/model.safetensors", (req, res) => {
					if (!fs.existsSync(WEIGHTS)) {
						res.statusCode = 404;
						res.end(`weights missing at ${WEIGHTS}; run: npm run bench:fetch`);
						return;
					}
					const { size } = fs.statSync(WEIGHTS);
					res.setHeader("content-type", "application/octet-stream");
					res.setHeader("content-length", size);
					fs.createReadStream(WEIGHTS).pipe(res);
				});
			},
		},
	],
});
