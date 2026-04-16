<script lang="ts">
	import { runInference } from "$lib/gemma3/inference";
	import {
		fromSafetensors,
		runAttention,
		runGemmaTextScaledWordEmbedding,
		runRMSNorm,
		type Gemma3,
	} from "$lib/gemma3/model";
	import { AutoTokenizer } from "@huggingface/transformers";
	import { numpy as np, defaultDevice, init, tree } from "@jax-js/jax";
	import { cachedFetch, safetensors } from "@jax-js/loaders";

	let downloadProgress = $state<number | null>(null);

	async function run() {
		const devices = await init("webgpu");
		if (!devices.includes("webgpu")) {
			alert("WebGPU required but not available!");
			return;
		}
		defaultDevice("webgpu");

		console.log("Running...");

		const weightsUrl =
			"https://huggingface.co/alanzhu39/gemma-3-270m-it-f16/resolve/main/model.safetensors";
		const data = await cachedFetch(weightsUrl, {}, (progress) => {
			downloadProgress = progress.totalBytes
				? Math.round((progress.loadedBytes / progress.totalBytes) * 100)
				: null;
		});
		const file = safetensors.parse(data);
		const weights: Gemma3 = fromSafetensors(file);
		const text = "Plants create energy through a process known as";

		console.log("Loaded weights");

		const tokenizer = await AutoTokenizer.from_pretrained("alanzhu39/gemma-3-270m-it-f16");

		await runInference(weights, tokenizer, text);

		tree.dispose(weights);
	}

	async function test() {
		const devices = await init("webgpu");
		if (!devices.includes("webgpu")) {
			alert("WebGPU required but not available!");
			return;
		}
		defaultDevice("webgpu");

		console.log("Running...");

		const weightsUrl =
			"https://huggingface.co/alanzhu39/gemma-3-270m-it-f16/resolve/main/model.safetensors";
		const data = await cachedFetch(weightsUrl, {}, (progress) => {
			downloadProgress = progress.totalBytes
				? Math.round((progress.loadedBytes / progress.totalBytes) * 100)
				: null;
		});
		const file = safetensors.parse(data);
		const weights: Gemma3 = fromSafetensors(file);
		const text = "Plants create energy through a process known as";

		console.log("Loaded weights");

		const tokenizer = await AutoTokenizer.from_pretrained("alanzhu39/gemma-3-270m-it-f16");
		const tokens = tokenizer.encode(text);

		console.log("Tokenized text");

		// Embed tokens
		const tokensAr = np.array(tokens, { dtype: np.uint32 });

		let x = runGemmaTextScaledWordEmbedding(weights.tokenEmbed.ref, tokensAr);

		let residual = x.ref;
		x = runRMSNorm(tree.ref(weights.layers[0].inputLayernorm), x);
		x = runAttention(tree.ref(weights.layers[0].selfAttn), x, true);
		// x = runRMSNorm(tree.ref(weights.layers[0].postAttentionLayernorm), x);
		// x = np.clip(x.add(residual), -65504.0, 65504.0);

		// residual = x.ref;
		// x = runRMSNorm(preFeedforwardLayernorm, x);
		// x = runMLP(mlp, x);
		// x = runRMSNorm(postFeedforwardLayernorm, x);
		// x = np.clip(x.add(residual), -65504.0, 65504.0);

		console.log(x.js());

		tree.dispose(weights);
	}
</script>

<h1>Welcome to Gemma3 inference!</h1>
<button onclick={run}>Run</button>
{#if downloadProgress != null}
	<div>Download progress: {downloadProgress}%</div>
{/if}

<button onclick={test}>Test</button>
