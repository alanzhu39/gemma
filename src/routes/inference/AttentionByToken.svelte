<script lang="ts">
	import { NUM_HEADS, NUM_LAYERS } from "$lib/gemma3/model";
	import { stringify } from "$lib/util";
	import { getInferenceContext } from "./context";

	const inferenceContext = getInferenceContext();
	const tokens = $derived(inferenceContext.tokens);

	// Calculate averaged weights of last token over all layers and heads
	const averagedWeights: number[] = $derived.by(() => {
		if (tokens.length <= 1) {
			return [];
		}
		const lastTokenIndex = tokens.length - 3;
		const avg: number[] = [];
		for (let l = 0; l < NUM_LAYERS; l++) {
			for (let h = 0; h < NUM_HEADS; h++) {
				for (let i = 0; i < lastTokenIndex; i++) {
					if (l === 0 && h === 0) {
						avg.push(0);
					}
					avg[i] +=
						inferenceContext.attentionWeights[l][h][lastTokenIndex][i] / (NUM_LAYERS * NUM_HEADS);
				}
			}
		}
		return avg;
	});

	function activation(value: number): number {
		return Math.round(value * 100);
	}
</script>

<div class="container">
	<div class="tokens">
		{#snippet tokenSnippet(weight: number, text: string)}
			<div class="token">
				{#if weight > 1}
					<div class="weight">{weight}%</div>
				{/if}
				<div class="text" style="--token-opacity: {weight}%;">{text}</div>
			</div>
		{/snippet}

		{#each tokens as token, i (i)}
			{@render tokenSnippet(activation(averagedWeights[i]), stringify(token))}
		{/each}
	</div>
</div>

<style lang="scss">
	.container {
		display: flex;
		flex-direction: row-reverse;
		overflow-x: auto;
		padding-bottom: 10px;

		.tokens {
			flex: 1;
			display: flex;
			gap: 4px;
			align-items: flex-end;
		}

		.token {
			display: flex;
			flex-direction: column;
			align-items: center;
			gap: 1px;

			.weight {
				font-size: 8px;
				color: $accent-terra;
				font-family: $font-mono;
			}

			.text {
				padding: 3px 7px;
				border-radius: 5px;
				font-size: 12px;
				font-family: $font-mono;
				background: rgba($accent-terra, var(--token-opacity));
				border: 1.5px solid $border-gray;
				color: $text-dark-gray;
				font-weight: 400;
				transition: 0.3s;
			}
		}
	}
</style>
