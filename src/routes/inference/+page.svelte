<script lang="ts">
	import { runInference } from "$lib/gemma3/inference";
	import { fromSafetensors, type Gemma3 } from "$lib/gemma3/model";
	import { AutoTokenizer } from "@huggingface/transformers";
	import { defaultDevice, init, tree } from "@jax-js/jax";
	import { cachedFetch, safetensors } from "@jax-js/loaders";
	import AttentionByToken from "./AttentionByToken.svelte";
	import Layers from "./Layers.svelte";
	import LayerView from "./layer-view/LayerView.svelte";
	import LayerPlaceholder from "./layer-view/LayerPlaceholder.svelte";
	import { setInferenceContext, type InferenceContext } from "./context";

	// Font configuration — swap these to change the UI typeface
	const fonts = {
		sans: "'Plus Jakarta Sans', sans-serif",
		mono: "'IBM Plex Mono', monospace",
	};

	const NUM_LAYERS = 18;
	let selectedLayer = $state<number | null>(null);
	function setSelectedLayer(layer: number) {
		if (layer === selectedLayer) {
			selectedLayer = null;
		} else {
			selectedLayer = layer;
		}
	}

	const inferenceContext = $state<InferenceContext>({
		tokens: [],
		attentionWeights: [],
	});
	const tokens = $derived(inferenceContext.tokens);
	setInferenceContext(inferenceContext);

	let context = $state("The capital of France is");
	let downloadProgress = $state<number | null>(null);
	let isRunning = $state(false);
	let isReady = $state(false);

	// Placeholder data shapes for visualization (replaced by real data when run)
	let attentionWeights = $state<number[][]>([]);
	let activations = $state<number[][]>([]);

	async function run() {
		isRunning = true;
		const devices = await init("webgpu");
		if (!devices.includes("webgpu")) {
			alert("WebGPU required but not available!");
			isRunning = false;
			return;
		}
		defaultDevice("webgpu");

		const weightsUrl =
			"https://huggingface.co/alanzhu39/gemma-3-270m-it-f16/resolve/main/model.safetensors";
		const data = await cachedFetch(weightsUrl, {}, (progress) => {
			downloadProgress = progress.totalBytes
				? Math.round((progress.loadedBytes / progress.totalBytes) * 100)
				: null;
		});
		const file = safetensors.parse(data);
		const weights: Gemma3 = fromSafetensors(file);
		const tokenizer = await AutoTokenizer.from_pretrained("alanzhu39/gemma-3-270m-it-f16");

		await runInference(weights, tokenizer, context, 1);

		tree.dispose(weights);
		isRunning = false;
		isReady = true;
	}

	// Demo placeholder tokens and data for layout preview
	const demoTokens = ["The", "▁capital", "▁of", "▁France", "▁is", "▁Paris"];
	const demoAttention = demoTokens.map((_, i) =>
		demoTokens.map((_, j) => (j <= i ? Math.random() : 0)),
	);
	const demoActivations = Array.from({ length: 18 }, () =>
		Array.from({ length: 32 }, () => Math.random() * 2 - 1),
	);

	$effect(() => {
		if (!isReady && tokens.length === 0) {
			inferenceContext.tokens = demoTokens;
			attentionWeights = demoAttention;
			activations = demoActivations;
		}
	});
</script>

<svelte:head>
	<link
		href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
		rel="stylesheet"
	/>
</svelte:head>

<div
	class="flex h-full flex-col overflow-hidden"
	style="background: #faf9f7; font-family: {fonts.sans};"
>
	<!-- Layers -->
	<Layers numLayers={NUM_LAYERS} {setSelectedLayer} />

	<!-- Main visualization area -->
	{#if selectedLayer !== null}
		<LayerView {selectedLayer} />
	{:else}
		<LayerPlaceholder />
	{/if}

	<!-- Context -->
	<div
		style="border-top: 1.5px solid rgb(232, 227, 221); background: rgb(255, 255, 255); padding: 12px 20px; flex-shrink: 0;"
	>
		<div style="display: flex; gap: 8px; align-items: center; margin-bottom: 10px;">
			<span
				style="font-size: 13px; font-weight: 600; color: rgb(28, 25, 23); font-family: &quot;Plus Jakarta Sans&quot;, sans-serif;"
			>
				Context
			</span>
		</div>
		<textarea
			bind:value={context}
			rows={5}
			style="
        width: 100%; resize: none; border: 1.5px solid rgb(232, 227, 221); border-radius: 7px; padding: 8px 12px;
        font-size: 13px; font-family: &quot;IBM Plex Mono&quot;, monospace; color: rgb(28, 25, 23); background: rgb(250, 249, 247);
        outline: none; line-height: 1.6; transition: border-color 0.15s; margin-bottom: 8px;
      "
			placeholder="Enter a prompt…"
		></textarea>
		<div style="display: flex; gap: 8px; align-items: center; margin-bottom: 10px;">
			<AttentionByToken />
			<div style="margin-left: auto; display: flex; gap: 7px;">
				<button
					onclick={run}
					disabled={isRunning}
					class="w-full rounded-lg px-4 py-2 text-sm font-semibold transition-all"
					style="
              padding: 5px 16px; border-radius: 6px; font-size: 12px; font-family: &quot;Plus Jakarta Sans&quot;, sans-serif;
              font-weight: 600; cursor: pointer; background: oklch(0.55 0.14 32); border: 1.5px solid oklch(0.55 0.14 32);
              color: rgb(255, 255, 255); box-shadow: oklch(0.55 0.14 32 / 0.2) 0px 2px 8px; transition: 0.12s;
            "
				>
					{#if isRunning}
						<!-- TODO: download modal -->
						<span style="font-family: {fonts.mono};">running…</span>
					{:else}
						Generate next →
					{/if}
				</button>
			</div>
		</div>
	</div>
</div>
