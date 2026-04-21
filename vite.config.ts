import tailwindcss from "@tailwindcss/vite";
import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	css: {
		preprocessorOptions: {
			scss: {
				additionalData: '@use "/src/styles/variables.scss" as *;',
			},
		},
	},
});
