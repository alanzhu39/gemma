<script lang="ts">
	import { runInference } from "$lib/gemma3/inference";
	import { fromSafetensors, type Gemma3 } from "$lib/gemma3/model";
	import { AutoTokenizer, PreTrainedTokenizer, TextStreamer } from "@huggingface/transformers";
	import { defaultDevice, init, setDebug } from "@jax-js/jax";
	import { cachedFetch, safetensors, type FetchProgress } from "@jax-js/loaders";
	import DownloadToast, { type DownloadState } from "$lib/DownloadToast.svelte";
	import { pipeline } from "@huggingface/transformers";

	let model = $state<Gemma3 | null>(null);
	let tokenizer = $state<PreTrainedTokenizer | null>(null);

	let measurements = $state<number[]>([]);

	let downloadState = $state<DownloadState>({
		visible: false,
		loadedBytes: 0,
	});

	async function getModel(): Promise<Gemma3> {
		if (model === null) {
			const weightsUrl =
				"https://huggingface.co/alanzhu39/gemma-3-270m-it-f16/resolve/main/model.safetensors";
			const data = await cachedFetch(weightsUrl, {}, (progress: FetchProgress) => {
				downloadState.visible = true;
				downloadState.loadedBytes = progress.loadedBytes;
				downloadState.totalBytes = progress.totalBytes;
			});
			downloadState.visible = false;
			const file = safetensors.parse(data);
			model = fromSafetensors(file);
		}
		return model;
	}

	async function getTokenizer(): Promise<PreTrainedTokenizer> {
		if (tokenizer === null) {
			tokenizer = await AutoTokenizer.from_pretrained("alanzhu39/gemma-3-270m-it-f16");
		}
		return tokenizer;
	}

	async function runBenchmark() {
		setDebug(1);

		const devices = await init("webgpu");
		if (!devices.includes("webgpu")) {
			alert("WebGPU required but not available!");
			return;
		}

		defaultDevice("webgpu");

		const model = await getModel();
		const tokenizer = await getTokenizer();

		const start = performance.now();

		const context = "Here is a Python function to find the nth Fibonacci number:";
		const steps = 20;
		const output = runInference(model, tokenizer, context, steps);

		const elapsed = performance.now() - start;
		const tps = output.length / (elapsed / 1000);
		console.log(output.length, elapsed);
		measurements.push(tps);
	}

	async function benchmarkTransformersJs() {
		console.log("starting transformers.js benchmark");

		// Create a text generation pipeline
		const generator = await pipeline("text-generation", "onnx-community/gemma-3-270m-it-ONNX", {
			dtype: "fp32",
		});

		// Define the list of messages
		const messages = [
			{ role: "system", content: "You are a helpful assistant." },
			{ role: "user", content: "Write a poem about machine learning." },
		];

		// Generate a response
		const start = performance.now();
		console.log(`Start: ${start}`);
		const output = await generator(messages, {
			max_new_tokens: 512,
			do_sample: false,
			streamer: new TextStreamer(generator.tokenizer, {
				skip_prompt: true,
				skip_special_tokens: true,
				// callback_function: (text) => { /* Optional callback function */ },
			}),
		});
		const elapsed = performance.now() - start;
		console.log(`Elapsed: ${elapsed}`);
		const message = output[0].generated_text.at(-1)?.content;
		const tps = message ? message.length / (elapsed / 1000) : 0;
		measurements.push(tps);
		console.log(message);
	}
</script>

<h1>Performance</h1>

<div>
	<button onclick={runBenchmark} style="cursor: pointer;">Benchmark</button>

	<button onclick={benchmarkTransformersJs} style="cursor: pointer;">
		Benchmark Transformers.js
	</button>
</div>

<div>
	Measurements
	{#each measurements.toReversed() as tps, i (i)}
		<div>{tps.toFixed(2)} tok/s</div>
	{/each}
</div>

<DownloadToast {...downloadState} />

<style lang="scss">
	button {
		border: 1px solid black;
		border-radius: 5px;
		padding: 5px;
	}
</style>
