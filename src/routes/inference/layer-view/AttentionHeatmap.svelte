<script lang="ts">
	import { stringify } from "$lib/util";
	import { getInferenceContext } from "../context";
	import AttentionByHead from "./AttentionByHead.svelte";

	const inferenceContext = getInferenceContext();
	const tokens = $derived(inferenceContext.tokens);
	const attentionWeights = $derived(inferenceContext.attentionWeights);

	function cellColor(value: number): string {
		const clamped = Math.max(0, Math.min(1, value));
		const alpha = Math.round(clamped * 100);
		return `oklch(0.55 0.14 32 / ${alpha}%)`;
	}
</script>

<div
	style="flex: 1 1 0%; padding: 16px 20px; overflow: auto; display: flex; flex-direction: column; gap: 10px;"
>
	<div>
		<div
			style="font-size: 13px; font-weight: 600; color: rgb(28, 25, 23); font-family: &quot;Plus Jakarta Sans&quot;, sans-serif; margin-bottom: 2px;"
		>
			Attention weights
			<span
				style="font-size: 11px; color: rgb(160, 152, 144); font-weight: 400; margin-left: 8px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
			>
				8×8 · causal mask applied
			</span>
		</div>
	</div>
	<div style="font-size: 11px; color: rgb(160, 152, 144); margin-bottom: 12px;">
		Each row shows how much a token attends to earlier tokens. Click another layer to compare.
	</div>

	<!-- Heatmap grid -->
	<div class="flex-1 overflow-auto p-5">
		{#if tokens.length > 0}
			<div class="inline-block">
				<!-- Column labels -->
				<div class="flex" style="padding-left: 64px; margin-bottom: 2px;">
					{#each tokens as token, i (i)}
						<div
							class="truncate text-center"
							style="
										width: 36px;
										font-family: monospace;
										font-size: 10px;
										color: #a09890;
										writing-mode: vertical-rl;
										transform: rotate(180deg);
										height: 52px;
										line-height: 1;
										padding-top: 4px;
									"
							title={token}
						>
							{stringify(token)}
						</div>
					{/each}
				</div>

				<!-- Rows with row label -->
				{#each tokens as rowToken, row (row)}
					<div class="mb-0.5 flex items-center">
						<div
							class="shrink-0 pr-2 text-right"
							style="width: 64px; font-family: monospace; font-size: 10px; color: #6b6560;"
							title={rowToken}
						>
							{rowToken.length > 15 ? rowToken.slice(0, 14) + "…" : stringify(rowToken)}
						</div>
						{#each tokens as colToken, col (col)}
							<div
								class="mr-0.5 rounded-sm"
								style="
											width: 36px;
											height: 36px;
											background: {cellColor(attentionWeights[row]?.[col] ?? 0)};
											flex-shrink: 0;
										"
								title="{rowToken} → {colToken}: {(attentionWeights[row]?.[col] ?? 0).toFixed(3)}"
							></div>
						{/each}
					</div>
				{/each}
			</div>
		{:else}
			<div
				class="flex h-full items-center justify-center"
				style="color: #a09890; font-family: monospace; font-size: 12px;"
			>
				Run inference to see attention patterns
			</div>
		{/if}
	</div>

	<!-- Activations per head -->
	<AttentionByHead />
</div>
