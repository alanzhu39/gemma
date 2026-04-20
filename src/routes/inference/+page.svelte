<script lang="ts">
	import { runInference } from "$lib/gemma3/inference";
	import { fromSafetensors, isSlidingAttention, type Gemma3 } from "$lib/gemma3/model";
	import { AutoTokenizer } from "@huggingface/transformers";
	import { defaultDevice, init, tree } from "@jax-js/jax";
	import { cachedFetch, safetensors } from "@jax-js/loaders";

	// Font configuration — swap these to change the UI typeface
	const fonts = {
		sans: "'Plus Jakarta Sans', sans-serif",
		mono: "'IBM Plex Mono', monospace",
	};

	let context = $state("The capital of France is");
	let maxTokens = $state(20);
	let selectedLayer = $state<number | null>(null);
	let numLayers = $state(18);
	let downloadProgress = $state<number | null>(null);
	let isRunning = $state(false);
	let isReady = $state(false);

	// Placeholder data shapes for visualization (replaced by real data when run)
	let tokens = $state<string[]>([]);
	let attentionWeights = $state<number[][]>([]);
	let activations = $state<number[][]>([]);

	async function run() {
		isRunning = true;
		const devices = await init("webgpu");
		if (!devices.includes("webgpu")) {
			alert("WebGPU required but not available!");
			isRunning = false;
			return;
		}
		defaultDevice("webgpu");

		const weightsUrl =
			"https://huggingface.co/alanzhu39/gemma-3-270m-it-f16/resolve/main/model.safetensors";
		const data = await cachedFetch(weightsUrl, {}, (progress) => {
			downloadProgress = progress.totalBytes
				? Math.round((progress.loadedBytes / progress.totalBytes) * 100)
				: null;
		});
		const file = safetensors.parse(data);
		const weights: Gemma3 = fromSafetensors(file);
		const tokenizer = await AutoTokenizer.from_pretrained("alanzhu39/gemma-3-270m-it-f16");

		await runInference(weights, tokenizer, context, maxTokens);

		tree.dispose(weights);
		isRunning = false;
		isReady = true;
	}

	function cellColor(value: number): string {
		const clamped = Math.max(0, Math.min(1, value));
		const alpha = Math.round(clamped * 100);
		return `oklch(0.55 0.14 32 / ${alpha}%)`;
	}

	function activationColor(value: number): string {
		if (value >= 0) {
			const alpha = Math.round(Math.min(value, 1) * 100);
			return `oklch(0.50 0.12 195 / ${alpha}%)`;
		}
		const alpha = Math.round(Math.min(-value, 1) * 100);
		return `oklch(0.55 0.14 32 / ${alpha}%)`;
	}

	// Demo placeholder tokens and data for layout preview
	const demoTokens = ["The", "▁capital", "▁of", "▁France", "▁is", "▁Paris"];
	const demoAttention = demoTokens.map((_, i) =>
		demoTokens.map((_, j) => (j <= i ? Math.random() : 0)),
	);
	const demoActivations = Array.from({ length: 18 }, () =>
		Array.from({ length: 32 }, () => Math.random() * 2 - 1),
	);

	$effect(() => {
		if (!isReady && tokens.length === 0) {
			tokens = demoTokens;
			attentionWeights = demoAttention;
			activations = demoActivations;
		}
	});
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
	<div
		style="padding: 10px 20px 8px; border-bottom: 1.5px solid rgb(232, 227, 221); background: rgb(255, 255, 255); flex-shrink: 0;"
	>
		<span
			class="mb-2 block text-xs font-semibold tracking-widest uppercase"
			style="color: #a09890; font-family: {fonts.mono};"
		>
			Layers
		</span>
		<div
			style="display: flex; align-items: center; gap: 4px; overflow-x: auto; padding-bottom: 2px;"
		>
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
					onclick={() => (selectedLayer = layer)}
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

	<!-- Main visualization area -->
	<div style="flex: 1 1 0%; display: flex; overflow: hidden; min-height: 0px;">
		{#if selectedLayer !== null}
			<!-- Layer info sidebar -->
			<div
				style="width: 220px; border-right: 1.5px solid rgb(232, 227, 221); padding: 16px; overflow: auto; flex-shrink: 0; background: rgb(255, 255, 255);"
			>
				<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
					<span
						style="font-size: 15px; font-weight: 600; color: rgb(28, 25, 23); font-family: &quot;Plus Jakarta Sans&quot;, sans-serif;"
						>Layer 10</span
					><span
						style="padding: 2px 7px; border-radius: 4px; font-size: 10px; font-family: &quot;IBM Plex Mono&quot;, monospace; font-weight: 600; background: oklch(0.96 0.04 32); color: oklch(0.55 0.14 32); border: 1px solid oklch(0.82 0.1 32);"
						>GQA</span
					>
				</div>
				<div
					style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
				>
					<span style="color: rgb(160, 152, 144);">Q heads</span><span
						style="color: rgb(28, 25, 23);">8</span
					>
				</div>
				<div
					style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
				>
					<span style="color: rgb(160, 152, 144);">KV heads</span><span
						style="color: rgb(28, 25, 23);">4</span
					>
				</div>
				<div
					style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
				>
					<span style="color: rgb(160, 152, 144);">Head dim</span><span
						style="color: rgb(28, 25, 23);">256</span
					>
				</div>
				<div
					style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
				>
					<span style="color: rgb(160, 152, 144);">FFN dim</span><span
						style="color: rgb(28, 25, 23);">16,384</span
					>
				</div>
				<div
					style="display: flex; justify-content: space-between; padding: 5px 0px; border-bottom: 1px solid rgb(232, 227, 221); font-size: 11px; font-family: &quot;IBM Plex Mono&quot;, monospace;"
				>
					<span style="color: rgb(160, 152, 144);">Context</span><span
						style="color: rgb(28, 25, 23);">full</span
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
			<!-- Attention weights -->
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
										font-family: {fonts.mono};
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
										{token}
									</div>
								{/each}
							</div>

							<!-- Rows with row label -->
							{#each tokens as rowToken, row (row)}
								<div class="mb-0.5 flex items-center">
									<div
										class="shrink-0 pr-2 text-right"
										style="width: 64px; font-family: {fonts.mono}; font-size: 10px; color: #6b6560;"
										title={rowToken}
									>
										{rowToken.length > 7 ? rowToken.slice(0, 6) + "…" : rowToken}
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
											title="{rowToken} → {colToken}: {(attentionWeights[row]?.[col] ?? 0).toFixed(
												3,
											)}"
										></div>
									{/each}
								</div>
							{/each}
						</div>
					{:else}
						<div
							class="flex h-full items-center justify-center"
							style="color: #a09890; font-family: {fonts.mono}; font-size: 12px;"
						>
							Run inference to see attention patterns
						</div>
					{/if}
				</div>

				<!-- Activations per head -->
				<div style="margin-top: 8px;">
					<div
						style="font-size: 10px; color: rgb(160, 152, 144); font-family: &quot;IBM Plex Mono&quot;, monospace; margin-bottom: 8px;"
					>
						ALL HEADS — last token's pattern
					</div>
					<div style="display: flex; gap: 6px; flex-wrap: wrap;">
						<div style="cursor: pointer;">
							<div
								style="font-size: 9px; color: oklch(0.55 0.14 32); font-family: &quot;IBM Plex Mono&quot;, monospace; text-align: center; margin-bottom: 2px; font-weight: 600;"
							>
								H0
							</div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.646 0.106 32); opacity: 1;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.949 0.01 32); opacity: 1;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.894 0.027 32); opacity: 1;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.703 0.088 32); opacity: 1;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.551 0.137 32); opacity: 1;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.915 0.021 32); opacity: 1;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.795 0.059 32); opacity: 1;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.54 0.14 32); opacity: 1;"
							></div>
						</div>
						<div style="cursor: pointer;">
							<div
								style="font-size: 9px; color: rgb(160, 152, 144); font-family: &quot;IBM Plex Mono&quot;, monospace; text-align: center; margin-bottom: 2px; font-weight: 400;"
							>
								H1
							</div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.961 0.006 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.884 0.03 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.961 0.006 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.91 0.022 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.962 0.006 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.972 0.003 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.772 0.066 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.54 0.14 32); opacity: 0.5;"
							></div>
						</div>
						<div style="cursor: pointer;">
							<div
								style="font-size: 9px; color: rgb(160, 152, 144); font-family: &quot;IBM Plex Mono&quot;, monospace; text-align: center; margin-bottom: 2px; font-weight: 400;"
							>
								H2
							</div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.54 0.14 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.693 0.091 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.9 0.025 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.634 0.11 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.695 0.091 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.558 0.134 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.75 0.073 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.548 0.137 32); opacity: 0.5;"
							></div>
						</div>
						<div style="cursor: pointer;">
							<div
								style="font-size: 9px; color: rgb(160, 152, 144); font-family: &quot;IBM Plex Mono&quot;, monospace; text-align: center; margin-bottom: 2px; font-weight: 400;"
							>
								H3
							</div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.954 0.008 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.935 0.014 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.962 0.006 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.891 0.028 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.968 0.004 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.925 0.017 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.967 0.004 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.54 0.14 32); opacity: 0.5;"
							></div>
						</div>
						<div style="cursor: pointer;">
							<div
								style="font-size: 9px; color: rgb(160, 152, 144); font-family: &quot;IBM Plex Mono&quot;, monospace; text-align: center; margin-bottom: 2px; font-weight: 400;"
							>
								H4
							</div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.951 0.009 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.846 0.042 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.905 0.024 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.54 0.14 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.788 0.061 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.842 0.044 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.691 0.092 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.548 0.137 32); opacity: 0.5;"
							></div>
						</div>
						<div style="cursor: pointer;">
							<div
								style="font-size: 9px; color: rgb(160, 152, 144); font-family: &quot;IBM Plex Mono&quot;, monospace; text-align: center; margin-bottom: 2px; font-weight: 400;"
							>
								H5
							</div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.945 0.011 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.959 0.007 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.963 0.005 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.974 0.002 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.972 0.003 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.962 0.006 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.964 0.005 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.54 0.14 32); opacity: 0.5;"
							></div>
						</div>
						<div style="cursor: pointer;">
							<div
								style="font-size: 9px; color: rgb(160, 152, 144); font-family: &quot;IBM Plex Mono&quot;, monospace; text-align: center; margin-bottom: 2px; font-weight: 400;"
							>
								H6
							</div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.94 0.013 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.917 0.02 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.907 0.023 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.952 0.009 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.848 0.042 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.934 0.015 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.608 0.118 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.54 0.14 32); opacity: 0.5;"
							></div>
						</div>
						<div style="cursor: pointer;">
							<div
								style="font-size: 9px; color: rgb(160, 152, 144); font-family: &quot;IBM Plex Mono&quot;, monospace; text-align: center; margin-bottom: 2px; font-weight: 400;"
							>
								H7
							</div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.933 0.015 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.97 0.003 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.964 0.005 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.973 0.002 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.975 0.002 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.974 0.002 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.959 0.007 32); opacity: 0.5;"
							></div>
							<div
								style="width: 14px; height: 8px; margin-bottom: 1px; border-radius: 1px; background: oklch(0.54 0.14 32); opacity: 0.5;"
							></div>
						</div>
					</div>
				</div>
			</div>
		{:else}
			<div
				style="flex: 1 1 0%; display: flex; align-items: center; justify-content: center; flex-direction: column; gap: 10px;"
			>
				<div style="font-size: 28px; color: rgb(232, 227, 221);">↑</div>
				<div
					style="font-size: 14px; color: rgb(160, 152, 144); font-family: &quot;Plus Jakarta Sans&quot;, sans-serif;"
				>
					Select a layer to inspect its attention weights
				</div>
				<div
					style="font-size: 12px; color: rgb(160, 152, 144); font-family: &quot;IBM Plex Mono&quot;, monospace;"
				>
					GQA = grouped-query attention · SWA = sliding window attention
				</div>
			</div>
		{/if}
	</div>

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
			<div style="display: flex; flex-wrap: wrap; gap: 4px; align-items: flex-end;">
				<div style="display: flex; flex-direction: column; align-items: center; gap: 1px;">
					<div
						style="padding: 3px 7px; border-radius: 5px; font-size: 12px; font-family: &quot;IBM Plex Mono&quot;, monospace; background: oklch(0.908 0.023 32); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.3s;"
					>
						The
					</div>
				</div>
				<div style="display: flex; flex-direction: column; align-items: center; gap: 1px;">
					<div
						style="padding: 3px 7px; border-radius: 5px; font-size: 12px; font-family: &quot;IBM Plex Mono&quot;, monospace; background: oklch(0.923 0.018 32); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.3s;"
					>
						transformer
					</div>
				</div>
				<div style="display: flex; flex-direction: column; align-items: center; gap: 1px;">
					<div
						style="padding: 3px 7px; border-radius: 5px; font-size: 12px; font-family: &quot;IBM Plex Mono&quot;, monospace; background: oklch(0.946 0.011 32); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.3s;"
					>
						model
					</div>
				</div>
				<div style="display: flex; flex-direction: column; align-items: center; gap: 1px;">
					<div
						style="font-size: 8px; color: oklch(0.55 0.14 32); font-family: &quot;IBM Plex Mono&quot;, monospace;"
					>
						10%
					</div>
					<div
						style="padding: 3px 7px; border-radius: 5px; font-size: 12px; font-family: &quot;IBM Plex Mono&quot;, monospace; background: oklch(0.887 0.03 32); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.3s;"
					>
						reads
					</div>
				</div>
				<div style="display: flex; flex-direction: column; align-items: center; gap: 1px;">
					<div
						style="padding: 3px 7px; border-radius: 5px; font-size: 12px; font-family: &quot;IBM Plex Mono&quot;, monospace; background: oklch(0.906 0.023 32); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.3s;"
					>
						tokens
					</div>
				</div>
				<div style="display: flex; flex-direction: column; align-items: center; gap: 1px;">
					<div
						style="padding: 3px 7px; border-radius: 5px; font-size: 12px; font-family: &quot;IBM Plex Mono&quot;, monospace; background: oklch(0.927 0.017 32); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.3s;"
					>
						one
					</div>
				</div>
				<div style="display: flex; flex-direction: column; align-items: center; gap: 1px;">
					<div
						style="font-size: 8px; color: oklch(0.55 0.14 32); font-family: &quot;IBM Plex Mono&quot;, monospace;"
					>
						12%
					</div>
					<div
						style="padding: 3px 7px; border-radius: 5px; font-size: 12px; font-family: &quot;IBM Plex Mono&quot;, monospace; background: oklch(0.863 0.037 32); border: 1.5px solid rgb(232, 227, 221); color: rgb(107, 101, 96); font-weight: 400; transition: 0.3s;"
					>
						by
					</div>
				</div>
				<div style="display: flex; flex-direction: column; align-items: center; gap: 1px;">
					<div
						style="font-size: 8px; color: oklch(0.55 0.14 32); font-family: &quot;IBM Plex Mono&quot;, monospace;"
					>
						47%
					</div>
					<div
						style="padding: 3px 7px; border-radius: 5px; font-size: 12px; font-family: &quot;IBM Plex Mono&quot;, monospace; background: oklch(0.54 0.14 32); border: 1.5px solid oklch(0.82 0.1 32); color: oklch(0.55 0.14 32); font-weight: 600; transition: 0.3s;"
					>
						one
					</div>
				</div>
			</div>
			<div style="margin-left: auto; display: flex; gap: 7px;">
				<button
					onclick={run}
					disabled={isRunning}
					class="w-full rounded-lg px-4 py-2 text-sm font-semibold transition-all"
					style="
              padding: 5px 16px; border-radius: 6px; font-size: 12px; font-family: &quot;Plus Jakarta Sans&quot;, sans-serif;
              font-weight: 600; cursor: pointer; background: oklch(0.55 0.14 32); border: 1.5px solid oklch(0.55 0.14 32);
              color: rgb(255, 255, 255); box-shadow: oklch(0.55 0.14 32 / 0.2) 0px 2px 8px; transition: 0.12s;
            "
				>
					{#if isRunning}
						<!-- TODO: download modal -->
						<span style="font-family: {fonts.mono};">running…</span>
					{:else}
						Generate next →
					{/if}
				</button>
			</div>
		</div>
	</div>
</div>
