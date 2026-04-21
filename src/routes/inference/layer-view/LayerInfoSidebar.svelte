<script lang="ts">
	import { isSlidingAttention, NUM_HEADS } from "$lib/gemma3/model";
	import { getInferenceContext } from "../context";

	const inferenceContext = getInferenceContext();
	const selectedLayer = $derived(inferenceContext.selectedLayer);
	const isSWA = $derived(isSlidingAttention(selectedLayer!));

	// TODO: head selection
	const selectedHead = 0;
</script>

<div class="container">
	<div class="header">
		<span class="title">
			Layer {selectedLayer}
		</span>
		<span class={`label ${isSWA ? "swa" : "gqa"}`}>
			{#if isSWA}
				SWA
			{:else}
				GQA
			{/if}
		</span>
	</div>

	{#snippet info(label: string, value: string)}
		<div class="info">
			<span class="label">{label}</span>
			<span class="value">{value}</span>
		</div>
	{/snippet}
	{@render info("Q heads", "4")}
	{@render info("KV heads", "1")}
	{@render info("Head dim", "256")}
	{@render info("MLP dim", "2,048")}
	{@render info("Norm", "RMSNorm")}
	{@render info("Context", isSWA ? "sliding window" : "full")}
	{#if isSWA}
		{@render info("Sliding window size", "512")}
	{/if}

	<div class="heads-container">
		<div class="title">ATTENTION HEAD</div>
		<div class="heads">
			{#each { length: NUM_HEADS }, i}
				<button class={`head-button ${selectedHead === i ? "selected" : ""}`}>
					{i}
				</button>
			{/each}
		</div>
	</div>
</div>

<style lang="scss">
	.container {
		width: 220px;
		border-right: 1.5px solid $border-gray;
		padding: 16px;
		overflow: auto;
		flex-shrink: 0;
		background: $background-white;

		.header {
			display: flex;
			align-items: center;
			gap: 8px;
			margin-bottom: 14px;

			.title {
				font-size: 15px;
				font-weight: 600;
				color: $text-black;
				font-family: $font-sans;
			}

			.label {
				padding: 2px 7px;
				border-radius: 4px;
				font-size: 10px;
				font-family: $font-mono;
				font-weight: 600;

				&.swa {
					color: $text-red;
					border: 1px solid $border-red;
					background: $background-red;
				}

				&.gqa {
					color: $text-blue;
					border: 1px solid $border-blue;
					background: $background-blue;
				}
			}
		}

		.info {
			display: flex;
			justify-content: space-between;
			padding: 5px 0px;
			border-bottom: 1px solid $border-gray;
			font-size: 11px;
			font-family: $font-mono;

			.label {
				color: $text-gray;
			}

			.value {
				color: $text-black;
			}
		}

		.heads-container {
			margin-top: 14px;
			margin-bottom: 8px;

			.title {
				font-size: 10px;
				color: rgb(160, 152, 144);
				font-family: $font-mono;
				margin-bottom: 6px;
			}

			.heads {
				display: flex;
				flex-wrap: wrap;
				gap: 4px;

				.head-button {
					width: 28px;
					height: 26px;
					border-radius: 5px;
					font-size: 11px;
					padding: 0px;
					font-family: $font-mono;
					cursor: pointer;
					background: $background-gray;
					border: 1.5px solid $border-gray;
					color: $text-dark-gray;
					font-weight: 400;
					transition: 0.1s;

					&.selected {
						background: $background-red;
						border: 1.5px solid $border-red;
						color: $text-red;
						font-weight: 600;
					}
				}
			}
		}
	}
</style>
