<script lang="ts">
	import { generateOnce } from "$lib/gemma3/inference";
	import { fromSafetensors, type Gemma3 } from "$lib/gemma3/model";
	import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";
	import { defaultDevice, init } from "@jax-js/jax";
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
	setInferenceContext(inferenceContext);

	let context = $state("The capital of France is");
	let isRunning = $state(false);
	$inspect(isRunning);

	let model = $state<Gemma3 | null>(null);
	let tokenizer = $state<PreTrainedTokenizer | null>(null);

	async function getModel(): Promise<Gemma3> {
		if (model === null) {
			const weightsUrl =
				"https://huggingface.co/alanzhu39/gemma-3-270m-it-f16/resolve/main/model.safetensors";
			const data = await cachedFetch(weightsUrl, {});
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

	async function run() {
		isRunning = true;
		const devices = await init("webgpu");
		if (!devices.includes("webgpu")) {
			alert("WebGPU required but not available!");
			return;
		}

		defaultDevice("webgpu");

		const model = await getModel();
		const tokenizer = await getTokenizer();

		const tokensAr = generateOnce(model, tokenizer, context);
		inferenceContext.tokens = tokensAr.flatMap((token) => tokenizer.decode([token]));
		context = tokenizer.decode(tokensAr, { skip_special_tokens: true });
		isRunning = false;
	}
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
		<!-- TODO: animate new tokens -->
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
					// TODO: disabled color
					class="w-full rounded-lg px-4 py-2 text-sm font-semibold transition-all"
					style="
              padding: 5px 16px; border-radius: 6px; font-size: 12px; font-family: &quot;Plus Jakarta Sans&quot;, sans-serif;
              font-weight: 600; cursor: pointer; background: oklch(0.55 0.14 32); border: 1.5px solid oklch(0.55 0.14 32);
              color: rgb(255, 255, 255); box-shadow: oklch(0.55 0.14 32 / 0.2) 0px 2px 8px; transition: 0.12s;
            "
				>
					{#if isRunning}
						Running...
					{:else}
						Generate next →
					{/if}
				</button>
			</div>
		</div>
	</div>
</div>
