<script lang="ts">
	import { runInference } from "$lib/gemma3/inference";
	import { fromSafetensors, type Gemma3 } from "$lib/gemma3/model";
	import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";
	import { defaultDevice, init } from "@jax-js/jax";
	import { cachedFetch, safetensors, type FetchProgress } from "@jax-js/loaders";
	import DownloadToast, { type DownloadState } from "$lib/DownloadToast.svelte";

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
		const steps = 100;
		const output = runInference(model, tokenizer, context, steps);

		const elapsed = performance.now() - start;
		const tps = output.length / (elapsed / 1000);
		measurements.push(tps);
	}
</script>

<h1>Performance</h1>
<button onclick={runBenchmark} style="cursor: pointer;">Benchmark</button>

<div>
	Measurements
	{#each measurements as tps, i (i)}
		<div>{tps.toFixed(2)} tok/s</div>
	{/each}
</div>

<DownloadToast {...downloadState} />
