<script lang="ts">
	type Props = {
		selectedLayer: number;
	};

	let { selectedLayer }: Props = $props();
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
			GQA
		</span>
	</div>
	<div
		style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
	>
		<span style="color: rgb(160, 152, 144);">Q heads</span><span style="color: rgb(28, 25, 23);"
			>8</span
		>
	</div>
	<div
		style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
	>
		<span style="color: rgb(160, 152, 144);">KV heads</span><span style="color: rgb(28, 25, 23);"
			>4</span
		>
	</div>
	<div
		style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
	>
		<span style="color: rgb(160, 152, 144);">Head dim</span><span style="color: rgb(28, 25, 23);"
			>256</span
		>
	</div>
	<div
		style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
	>
		<span style="color: rgb(160, 152, 144);">FFN dim</span><span style="color: rgb(28, 25, 23);"
			>16,384</span
		>
	</div>
	<div
		style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
	>
		<span style="color: rgb(160, 152, 144);">Context</span><span style="color: rgb(28, 25, 23);"
			>full</span
		>
	</div>
	<div
		style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
	>
		<span style="color: rgb(160, 152, 144);">Norm</span><span style="color: rgb(28, 25, 23);"
			>RMSNorm</span
		>
	</div>
	<div style="margin-top: 14px; margin-bottom: 8px;">
		<div
			style="font-size: 10px; color: rgb(160, 152, 144); font-family: &quot;IBM Plex Mono&quot;, monospace; margin-bottom: 6px;"
		>
			ATTENTION HEAD
		</div>
		<div style="display: flex; flex-wrap: wrap; gap: 4px;">
			<button
				style="width: 28px; height: 26px; border-radius: 5px; font-size: 11px; padding: 0px; font-family: &quot;IBM Plex Mono&quot;, monospace; cursor: pointer; background: oklch(0.96 0.04 32); border: 1.5px solid oklch(0.82 0.1 32); color: oklch(0.55 0.14 32); font-weight: 600; transition: 0.1s;"
				>0</button
			><button
				style="width: 28px; height: 26px; border-radius: 5px; font-size: 11px; padding: 0px; font-family: &quot;IBM Plex Mono&quot;, monospace; cursor: pointer; background: rgb(250, 249, 247); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.1s;"
				>1</button
			><button
				style="width: 28px; height: 26px; border-radius: 5px; font-size: 11px; padding: 0px; font-family: &quot;IBM Plex Mono&quot;, monospace; cursor: pointer; background: rgb(250, 249, 247); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.1s;"
				>2</button
			><button
				style="width: 28px; height: 26px; border-radius: 5px; font-size: 11px; padding: 0px; font-family: &quot;IBM Plex Mono&quot;, monospace; cursor: pointer; background: rgb(250, 249, 247); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.1s;"
				>3</button
			><button
				style="width: 28px; height: 26px; border-radius: 5px; font-size: 11px; padding: 0px; font-family: &quot;IBM Plex Mono&quot;, monospace; cursor: pointer; background: rgb(250, 249, 247); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.1s;"
				>4</button
			><button
				style="width: 28px; height: 26px; border-radius: 5px; font-size: 11px; padding: 0px; font-family: &quot;IBM Plex Mono&quot;, monospace; cursor: pointer; background: rgb(250, 249, 247); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.1s;"
				>5</button
			><button
				style="width: 28px; height: 26px; border-radius: 5px; font-size: 11px; padding: 0px; font-family: &quot;IBM Plex Mono&quot;, monospace; cursor: pointer; background: rgb(250, 249, 247); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.1s;"
				>6</button
			><button
				style="width: 28px; height: 26px; border-radius: 5px; font-size: 11px; padding: 0px; font-family: &quot;IBM Plex Mono&quot;, monospace; cursor: pointer; background: rgb(250, 249, 247); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.1s;"
				>7</button
			>
		</div>
	</div>
</div>
