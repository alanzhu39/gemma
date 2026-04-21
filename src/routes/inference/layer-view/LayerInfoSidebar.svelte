<script lang="ts">
	import { isSlidingAttention, NUM_HEADS } from "$lib/gemma3/model";
	import { getInferenceContext } from "../context";

	const inferenceContext = getInferenceContext();
	const selectedLayer = $derived(inferenceContext.selectedLayer);
	const isSWA = $derived(isSlidingAttention(selectedLayer!));
</script>

<div
	style="width: 220px; border-right: 1.5px solid rgb(232, 227, 221); padding: 16px; overflow: auto; flex-shrink: 0; background: rgb(255, 255, 255);"
>
	<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
		<span
			style="font-size: 15px; font-weight: 600; color: rgb(28, 25, 23); font-family: &quot;Plus Jakarta Sans&quot;, sans-serif;"
		>
			Layer {selectedLayer}
		</span>
		<span
			style="padding: 2px 7px; border-radius: 4px; font-size: 10px; font-family: &quot;IBM Plex Mono&quot;, monospace; font-weight: 600; background: oklch(0.96 0.04 32); color: oklch(0.55 0.14 32); border: 1px solid oklch(0.82 0.1 32);"
		>
			{#if isSWA}
				SWA
			{:else}
				GQA
			{/if}
		</span>
	</div>
	<div
		style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
	>
		<span style="color: rgb(160, 152, 144);">Q heads</span>
		<span style="color: rgb(28, 25, 23);">4</span>
	</div>
	<div
		style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
	>
		<span style="color: rgb(160, 152, 144);">KV heads</span>
		<span style="color: rgb(28, 25, 23);">1</span>
	</div>
	<div
		style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
	>
		<span style="color: rgb(160, 152, 144);">Head dim</span>
		<span style="color: rgb(28, 25, 23);">256</span>
	</div>
	<div
		style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
	>
		<span style="color: rgb(160, 152, 144);">MLP dim</span>
		<span style="color: rgb(28, 25, 23);">2,048</span>
	</div>
	<div
		style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
	>
		<span style="color: rgb(160, 152, 144);">Norm</span>
		<span style="color: rgb(28, 25, 23);">RMSNorm</span>
	</div>
	<div
		style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
	>
		<span style="color: rgb(160, 152, 144);">Context</span>
		<span style="color: rgb(28, 25, 23);">{isSWA ? "sliding window" : "full"}</span>
	</div>
	{#if isSWA}
		<div
			style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
		>
			<span style="color: rgb(160, 152, 144);">Sliding window size</span>
			<span style="color: rgb(28, 25, 23);">512</span>
		</div>
	{/if}
	<div style="margin-top: 14px; margin-bottom: 8px;">
		<div
			style="font-size: 10px; color: rgb(160, 152, 144); font-family: &quot;IBM Plex Mono&quot;, monospace; margin-bottom: 6px;"
		>
			ATTENTION HEAD
		</div>
		<div style="display: flex; flex-wrap: wrap; gap: 4px;">
			{#each { length: NUM_HEADS }, i}
				<button
					// style="width: 28px; height: 26px; border-radius: 5px; font-size: 11px; padding: 0px; font-family: &quot;IBM Plex Mono&quot;, monospace; cursor: pointer; background: oklch(0.96 0.04 32); border: 1.5px solid oklch(0.82 0.1 32); color: oklch(0.55 0.14 32); font-weight: 600; transition: 0.1s;"
					style="width: 28px; height: 26px; border-radius: 5px; font-size: 11px; padding: 0px; font-family: &quot;IBM Plex Mono&quot;, monospace; cursor: pointer; background: rgb(250, 249, 247); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.1s;"
				>
					{i}
				</button>
			{/each}
		</div>
	</div>
</div>
