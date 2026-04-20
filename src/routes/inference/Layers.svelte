<script lang="ts">
	import { isSlidingAttention } from "$lib/gemma3/model";

	type Props = {
		numLayers: number;
		setSelectedLayer: (layer: number) => void;
	};

	let { numLayers, setSelectedLayer }: Props = $props();
</script>

<div
	style="padding: 10px 20px 8px; border-bottom: 1.5px solid rgb(232, 227, 221); background: rgb(255, 255, 255); flex-shrink: 0;"
>
	<span
		style="font-size: 10px; color: rgb(160, 152, 144); font-family: &quot;IBM Plex Mono&quot;, monospace; margin-bottom: 6px;"
	>
		LAYERS
	</span>
	<div style="display: flex; align-items: center; gap: 4px; overflow-x: auto; padding-bottom: 2px;">
		<div
			class="shrink-0 rounded-md px-3 py-1 text-xs"
			style="
          padding: 5px 9px; border-radius: 6px; background: oklch(0.96 0.05 195); border: 1.5px solid oklch(0.78 0.1 195); flex-shrink: 0;
          font-size: 9px; color: oklch(0.5 0.12 195); font-family: &quot;IBM Plex Mono&quot;, monospace; text-align: center;
        "
		>
			embed
		</div>
		<div style="color: rgb(232, 227, 221); font-size: 14px; flex-shrink: 0; padding: 0px 2px;">
			→
		</div>
		{#each Array.from({ length: numLayers }, (_, i) => i) as layer (layer)}
			{#if layer !== 0}
				<div style="color: rgb(232, 227, 221); font-size: 11px; flex-shrink: 0;">›</div>
			{/if}
			<!-- TODO: layer info modal -->
			<button
				onclick={() => setSelectedLayer(layer)}
				style="
            padding: 5px 8px; border-radius: 6px; cursor: pointer;
            user-select: none; background: rgb(255, 255, 255);
            border: 1.5px solid rgb(232, 227, 221); display: flex;
            flex-direction: column; align-items: center; gap: 2px; transition: 0.12s;
            box-shadow: none;
          "
			>
				<span
					style="font-size: 10px; font-family: &quot;IBM Plex Mono&quot;, monospace; color: rgb(160, 152, 144);"
				>
					L{layer}
				</span>
				<span
					style="font-size: 9px; font-family: &quot;IBM Plex Mono&quot;, monospace; font-weight: 600; color: rgb(160, 152, 144);"
				>
					{isSlidingAttention(layer) ? "SWA" : "GQA"}
				</span>
			</button>
		{/each}
		<div style="color: rgb(232, 227, 221); font-size: 14px; flex-shrink: 0; padding: 0px 2px;">
			→
		</div>
		<div
			style="
          padding: 5px 9px; border-radius: 6px; background: oklch(0.96 0.04 32); border: 1.5px solid oklch(0.82 0.1 32); flex-shrink: 0;
          font-size: 9px; color: oklch(0.55 0.14 32); font-family: &quot;IBM Plex Mono&quot;, monospace; text-align: center;
        "
		>
			lm_head
		</div>
	</div>
</div>
