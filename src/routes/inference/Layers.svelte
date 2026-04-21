<script lang="ts">
	import { isSlidingAttention } from "$lib/gemma3/model";
	import { getInferenceContext } from "./context";

	type Props = {
		numLayers: number;
		setSelectedLayer: (layer: number) => void;
	};

	let { numLayers, setSelectedLayer }: Props = $props();

	const inferenceContext = getInferenceContext();
	const selectedLayer = $derived(inferenceContext.selectedLayer);
</script>

<div class="container">
	<span class="title">LAYERS</span>
	<div class="layers">
		<div class="embed">embed</div>
		<div class="arrow-1">→</div>
		{#each Array.from({ length: numLayers }, (_, i) => i) as layer (layer)}
			{#if layer !== 0}
				<div class="arrow-2">›</div>
			{/if}
			<!-- TODO: layer info modal -->
			<button
				onclick={() => setSelectedLayer(layer)}
				class={`layer ${isSlidingAttention(layer) ? "swa" : "gqa"} ${layer === selectedLayer ? "selected" : ""}`}
			>
				<span class="layer-num">
					L{layer}
				</span>
				<span class="layer-type">
					{isSlidingAttention(layer) ? "SWA" : "GQA"}
				</span>
			</button>
		{/each}
		<div class="arrow-1">→</div>
		<div class="lm-head">lm_head</div>
	</div>
</div>

<style lang="scss">
	.container {
		padding: 10px 20px 8px;
		border-bottom: 1.5px solid $border-gray;
		background: $background-white;
		flex-shrink: 0;
		min-width: 100%;

		.title {
			font-size: 10px;
			color: $text-gray;
			font-family: $font-mono;
			margin-bottom: 6px;
		}

		.layers {
			display: flex;
			align-items: center;
			gap: 4px;
			overflow-x: auto;
			padding-bottom: 2px;

			.embed {
				padding: 5px 9px;
				border-radius: 6px;
				background: $background-teal;
				border: 1.5px solid $border-teal;
				flex-shrink: 0;
				font-size: 9px;
				color: $text-teal;
				font-family: $font-mono;
				text-align: center;
			}

			.arrow-1 {
				color: $text-light-gray;
				font-size: 14px;
				flex-shrink: 0;
				padding: 0px 2px;
			}

			.arrow-2 {
				color: $text-light-gray;
				font-size: 11px;
				flex-shrink: 0;
			}

			.layer {
				padding: 5px 8px;
				border-radius: 6px;
				cursor: pointer;
				user-select: none;
				background: $background-white;
				border: 1.5px solid $border-gray;
				color: $text-gray;
				display: flex;
				flex-direction: column;
				align-items: center;
				gap: 2px;
				transition: 0.12s;
				box-shadow: none;

				&.swa {
					&:hover {
						background: $background-red;
					}

					&.selected {
						color: $text-red;
						border: 1.5px solid $border-red;
						background: $background-red;
					}
				}

				&.gqa {
					&:hover {
						background: $background-blue;
					}

					&.selected {
						color: $text-blue;
						border: 1.5px solid $border-blue;
						background: $background-blue;
					}
				}

				.layer-num {
					font-size: 10px;
					font-family: $font-mono;
				}

				.layer-type {
					font-size: 9px;
					font-family: $font-mono;
					font-weight: 600;
				}
			}

			.lm-head {
				padding: 5px 9px;
				border-radius: 6px;
				background: $background-red;
				border: 1.5px solid $border-red;
				flex-shrink: 0;
				font-size: 9px;
				color: $text-red;
				font-family: $font-mono;
				text-align: center;
			}
		}
	}
</style>
