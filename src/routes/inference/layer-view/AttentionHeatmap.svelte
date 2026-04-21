<script lang="ts">
	import { stringify } from "$lib/util";
	import { getInferenceContext } from "../context";

	const inferenceContext = getInferenceContext();
	const tokens = $derived(inferenceContext.tokens);
	const selectedLayer = $derived(inferenceContext.selectedLayer);
	const selectedHead = $derived(inferenceContext.selectedHead);

	let zoom = $state(1.0);
	function zoomIn() {
		zoom = Math.min(zoom + 0.1, 2);
	}
	function zoomOut() {
		zoom = Math.max(zoom - 0.1, 0.2);
	}

	// Must have selected layer
	const attentionWeights = $derived(
		inferenceContext.attentionWeights[selectedLayer!][selectedHead],
	);

	const [minWeight, maxWeight] = $derived.by(() => {
		let minWeight = Infinity;
		let maxWeight = 0;
		const n = attentionWeights.length;
		const m = attentionWeights[0].length;
		for (let i = 0; i < n; i++) {
			for (let j = 0; j < m; j++) {
				if (attentionWeights[i][j] < minWeight && attentionWeights[i][j] > 0) {
					minWeight = attentionWeights[i][j];
				}
				if (attentionWeights[i][j] > maxWeight) {
					maxWeight = attentionWeights[i][j];
				}
			}
		}
		return [minWeight, maxWeight];
	});

	function cellOpacity(value: number): number {
		return Math.round(((value - minWeight) / (maxWeight - minWeight)) * 100);
	}
</script>

<div class="container">
	<div class="header">
		<div>
			<span class="title">Attention weights</span>
			<span class="subtitle"> · causal mask applied</span>
		</div>
	</div>
	<div class="text">
		Each row shows how much a token attends to each previous token in the columns.
	</div>

	<!-- Heatmap grid -->
	<div class="grid-container">
		{#if tokens.length > 0}
			<div class="grid" style="zoom: {zoom};">
				<!-- Column labels -->
				<div class="column-labels">
					{#each tokens.slice(0, -1) as token, i (i)}
						<div class="label" title={token}>
							{stringify(token)}
						</div>
					{/each}
				</div>

				<!-- Rows with row label -->
				{#each tokens.slice(1) as rowToken, row (row)}
					<div class="rows">
						<div class="label" title={rowToken}>
							{rowToken.length > 10 ? rowToken.slice(0, 10) + "…" : stringify(rowToken)}
						</div>
						{#each tokens.slice(0, -1) as colToken, col (col)}
							{@const weight = attentionWeights[row]?.[col] ?? 0}
							<div
								class={`cell ${weight <= 0 ? "empty" : ""}`}
								style={`--cell-opacity: ${cellOpacity(weight)}%`}
								title="{stringify(rowToken)} → {stringify(colToken)}: {weight.toFixed(3)}"
							></div>
						{/each}
					</div>
				{/each}
			</div>
		{:else}
			<div class="placeholder">Run inference to see attention patterns</div>
		{/if}
	</div>

	<div class="zoom-controls">
		Zoom:
		<button onclick={zoomOut} disabled={zoom <= 0.3}>−</button>
		<span>{Math.round(zoom * 100)}%</span>
		<button onclick={zoomIn} disabled={zoom >= 2}>+</button>
	</div>
</div>

<style lang="scss">
	.container {
		flex: 1 1 0%;
		padding: 16px 20px;
		overflow: auto;
		display: flex;
		flex-direction: column;
		gap: 10px;

		.header {
			display: flex;
			align-items: center;
			justify-content: space-between;
		}

		.zoom-controls {
			display: flex;
			align-items: center;
			gap: 4px;

			font-size: 11px;
			color: $text-gray;
			font-family: $font-mono;
			min-width: 32px;
			text-align: center;

			button {
				width: 22px;
				height: 22px;
				padding: 0;
				border: 1px solid $border-gray;
				border-radius: 4px;
				background: none;
				cursor: pointer;
				font-size: 14px;
				line-height: 1;
				color: $text-gray;
				display: flex;
				align-items: center;
				justify-content: center;

				&:hover:not(:disabled) {
					background: $border-gray;
				}

				&:disabled {
					opacity: 0.3;
					cursor: default;
				}
			}
		}

		.title {
			font-size: 13px;
			font-weight: 600;
			color: $text-black;
			font-family: $font-sans;
			margin-bottom: 2px;
		}

		.subtitle {
			font-size: 11px;
			color: $text-gray;
			font-weight: 400;
			margin-left: 6px;
			font-family: $font-mono;
		}

		.text {
			font-size: 11px;
			color: $text-gray;
			margin-bottom: 12px;
		}

		.grid-container {
			flex: 1;
			overflow: auto;
			margin-bottom: 5px;

			.grid {
				display: inline-block;

				.column-labels {
					display: flex;
					gap: 2px;
					padding-left: 79px;
					margin-bottom: 2px;
					align-items: center;

					.label {
						overflow: hidden;
						text-overflow: ellipsis;
						white-space: nowrap;
						text-align: left;
						width: 36px;
						font-family: monospace;
						font-size: 10px;
						color: $text-dark-gray;
						writing-mode: vertical-rl;
						transform: rotate(180deg);
						height: 52px;
						line-height: 1;
						padding-top: 4px;
					}
				}

				.rows {
					margin-bottom: 2px;
					display: flex;
					align-items: center;
					gap: 2px;

					.label {
						flex-shrink: 0;
						padding-right: 6px;
						text-align: right;
						width: 64px;
						font-family: monospace;
						font-size: 10px;
						color: $text-dark-gray;
					}

					.cell {
						border-radius: 4px;
						width: 36px;
						height: 36px;
						background: rgba($accent-terra, var(--cell-opacity));
						flex-shrink: 0;

						&.empty {
							border: 1px solid $border-gray;
						}
					}
				}
			}
		}

		.placeholder {
			display: flex;
			height: 100%;
			align-items: center;
			justify-content: center;
			color: $text-gray;
			font-family: monospace;
			font-size: 12px;
		}
	}
</style>
