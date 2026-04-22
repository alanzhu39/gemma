<script lang="ts">
	import { generateOnce } from "$lib/gemma3/inference";
	import { fromSafetensors, NUM_LAYERS, type Gemma3 } from "$lib/gemma3/model";
	import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";
	import { defaultDevice, init } from "@jax-js/jax";
	import { cachedFetch, safetensors } from "@jax-js/loaders";
	import AttentionByToken from "./AttentionByToken.svelte";
	import Layers from "./Layers.svelte";
	import LayerView from "./layer-view/LayerView.svelte";
	import LayerPlaceholder from "./layer-view/LayerPlaceholder.svelte";
	import { setInferenceContext, type InferenceContext } from "./context";

	const inferenceContext = $state<InferenceContext>({
		tokens: [],
		attentionWeights: [],
		selectedLayer: null,
		selectedHead: 0,
	});
	setInferenceContext(inferenceContext);
	const selectedLayer = $derived(inferenceContext.selectedLayer);
	const tokens = $derived(inferenceContext.tokens);

	function setSelectedLayer(layer: number) {
		if (layer === selectedLayer) {
			inferenceContext.selectedLayer = null;
		} else {
			inferenceContext.selectedLayer = layer;
			inferenceContext.selectedHead = 0;
		}
	}

	let context = $state("Plants create energy through a process known as");
	let isRunning = $state(false);

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

		const [tokensAr, attentionWeights] = generateOnce(model, tokenizer, context);
		context = tokenizer.decode(tokensAr, { skip_special_tokens: true });
		isRunning = false;
		inferenceContext.tokens = tokensAr.flatMap((token) => tokenizer.decode([token]));
		inferenceContext.attentionWeights = attentionWeights;
	}
</script>

<svelte:head>
	<link
		href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
		rel="stylesheet"
	/>
</svelte:head>

<div class="container">
	<!-- Layers -->
	<Layers numLayers={NUM_LAYERS} {setSelectedLayer} />

	<!-- Main visualization area -->
	{#if selectedLayer !== null}
		<LayerView />
	{:else}
		<LayerPlaceholder />
	{/if}

	<!-- Context -->
	<div class="context">
		<div class="header">
			<span class="title">Context</span>
			{#if tokens.length > 1}
				<span class="subtitle"> · {tokens.length} tokens</span>
			{/if}
		</div>
		<!-- TODO: animate new tokens -->
		<textarea bind:value={context} rows={5} class="text-input" placeholder="Enter a prompt…"
		></textarea>
		<div class="actions">
			<AttentionByToken />
			<div class="generate">
				<button onclick={run} disabled={isRunning} class="run-button">
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

<style lang="scss">
	.container {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-width: 100%;
		overflow: hidden;
		background: $background-gray;
		font-family: $font-sans;

		.context {
			border-top: 1.5px solid $border-gray;
			background: $background-white;
			padding: 12px 20px;
			flex-shrink: 0;

			.header {
				display: flex;
				gap: 8px;
				align-items: center;
				margin-bottom: 10px;

				.title {
					font-size: 13px;
					font-weight: 600;
					color: $text-black;
					font-family: $font-sans;
				}

				.subtitle {
					font-size: 11px;
					color: $text-gray;
					font-weight: 400;
					font-family: $font-mono;
				}
			}

			.text-input {
				width: 100%;
				resize: none;
				border: 1.5px solid $border-gray;
				border-radius: 7px;
				padding: 8px 12px;
				font-size: 13px;
				font-family: $font-mono;
				color: $text-black;
				background: $background-gray;
				outline: none;
				line-height: 1.6;
				transition: border-color 0.15s;
				margin-bottom: 8px;
			}

			.actions {
				display: flex;
				gap: 8px;
				align-items: center;

				.generate {
					margin-left: auto;
					display: flex;
					gap: 7px;
					padding: 10px 0;

					.run-button {
						@include transition-all;

						width: 132px;
						border-radius: 8px;
						padding: 8px 16px;
						font-size: 14px;
						line-height: 1.42;
						font-weight: 600;
						padding: 5px 16px;
						border-radius: 6px;
						font-size: 12px;
						font-family: $font-sans;
						font-weight: 600;
						cursor: pointer;
						background: $accent-terra;
						border: 1.5px solid $accent-terra;
						color: $text-white;
						transition: 0.12s;

						&:disabled {
							opacity: 0.5;
							cursor: not-allowed;
							box-shadow: none;
						}
					}
				}
			}
		}
	}
</style>
