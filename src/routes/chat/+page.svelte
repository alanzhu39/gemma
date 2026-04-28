<script lang="ts">
	import { SAMPLING_DEFAULTS, streamGenerate } from "$lib/gemma3/inference";
	import {
		createGemma3State,
		fromSafetensors,
		MAX_CONTEXT_LEN,
		type Gemma3,
		type Gemma3State,
	} from "$lib/gemma3/model";
	import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";
	import { defaultDevice, init, tree } from "@jax-js/jax";
	import { cachedFetch, safetensors, type FetchProgress } from "@jax-js/loaders";
	import { marked } from "marked";
	import DownloadToast, { type DownloadState } from "$lib/DownloadToast.svelte";

	type Message = {
		role: "user" | "assistant";
		content: string;
		tokens?: number;
		tps?: number;
		latencyMs?: number;
	};

	let messages = $state<Message[]>([]);
	let inputVal = $state("");
	let isStreaming = $state(false);
	let isLoadingModel = $state(false);
	let contextTokens = $state(0);

	let samplingState = $state({ ...SAMPLING_DEFAULTS });
	const { temperature, topK, topP } = $derived(samplingState);

	let model = $state<Gemma3 | null>(null);
	let tokenizer = $state<PreTrainedTokenizer | null>(null);
	let modelState = $state<Gemma3State | null>(null);

	let chatEndEl = $state<HTMLDivElement | null>(null);
	let scrollTrigger = $state(0);

	let downloadState = $state<DownloadState>({
		visible: false,
		loadedBytes: 0,
	});

	$effect(() => {
		// eslint-disable-next-line @typescript-eslint/no-unused-expressions -- detect scrollTrigger reactively
		scrollTrigger;
		chatEndEl?.scrollIntoView({ block: "end" });
	});

	async function getModel(): Promise<Gemma3> {
		if (model) return model;
		const weightsUrl =
			"https://huggingface.co/alanzhu39/gemma-3-270m-it-f16/resolve/main/model.safetensors";
		const data = await cachedFetch(weightsUrl, {}, (progress: FetchProgress) => {
			downloadState.visible = true;
			downloadState.loadedBytes = progress.loadedBytes;
			downloadState.totalBytes = progress.totalBytes;
		});
		downloadState.visible = false;
		const file = safetensors.parse(data);
		model = fromSafetensors(file);
		return model;
	}

	async function getTokenizer(): Promise<PreTrainedTokenizer> {
		if (tokenizer) return tokenizer;
		const chatTemplateUrl =
			"https://huggingface.co/alanzhu39/gemma-3-270m-it-f16/resolve/main/chat_template.jinja";
		const chatTemplate = await fetch(chatTemplateUrl).then((r) => r.text());
		tokenizer = await AutoTokenizer.from_pretrained("alanzhu39/gemma-3-270m-it-f16");
		tokenizer.chat_template = chatTemplate;
		return tokenizer;
	}

	async function send() {
		if (!inputVal.trim() || isStreaming) return;

		const userContent = inputVal.trim();
		inputVal = "";

		messages.push({ role: "user", content: userContent });
		scrollTrigger++;

		isStreaming = true;
		isLoadingModel = !model || !tokenizer;

		try {
			const devices = await init("webgpu");
			if (!devices.includes("webgpu")) {
				alert("WebGPU required but not available!");
				return;
			}
			defaultDevice("webgpu");

			const [model, tokenizer] = await Promise.all([getModel(), getTokenizer()]);
			isLoadingModel = false;

			if (modelState === null) {
				modelState = createGemma3State(model);
			}

			const userTokens = tokenizer.encode(userContent).length;
			messages[messages.length - 1].tokens = userTokens;
			contextTokens += userTokens;

			const chatHistory = messages.map((msg) => ({ role: msg.role, content: msg.content }));
			messages.push({ role: "assistant", content: "" });
			scrollTrigger++;

			let tokenCount = 0;
			const startTime = performance.now();

			for await (const { token, state: finalState } of streamGenerate(
				model,
				tokenizer,
				modelState,
				chatHistory,
				{
					temperature,
					topK,
					topP,
				},
			)) {
				if (token !== undefined) {
					messages[messages.length - 1].content += token;
					tokenCount++;
					contextTokens++;
					scrollTrigger++;
				}
				if (finalState !== undefined) {
					modelState = finalState;
				}
			}

			const elapsed = performance.now() - startTime;
			const last = messages[messages.length - 1];
			last.tokens = tokenCount;
			last.tps = parseFloat((tokenCount / (elapsed / 1000)).toFixed(1));
			last.latencyMs = parseFloat((elapsed / tokenCount).toFixed(0));
			scrollTrigger++;
		} finally {
			isStreaming = false;
			isLoadingModel = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			if (!isStreaming) send();
		}
	}

	function clearChat() {
		if (!isStreaming) {
			messages = [];
			contextTokens = 0;
			if (modelState !== null) {
				tree.dispose(modelState);
				modelState = null;
			}
		}
	}
</script>

<DownloadToast {...downloadState} />

<div class="layout">
	<!-- Sidebar -->
	<aside class="sidebar">
		<div class="sidebar-section">
			<div class="section-label">MODEL</div>
			<div class="model-badge">
				<div class="model-name">gemma-3-270m-it</div>
				<div class="model-sub">f16 · webgpu</div>
			</div>
		</div>

		<div class="sidebar-section">
			<div class="section-label">STATUS</div>
			<div class={`status-pill ${isLoadingModel ? "loading" : model ? "ready" : "idle"}`}>
				{isLoadingModel ? "● loading…" : model ? "● ready" : "● idle"}
			</div>
		</div>

		<div class="sidebar-section">
			<div class="section-label">CONTEXT</div>
			<div class="stat-row">
				<span class="stat-value">{contextTokens.toLocaleString()}</span>
				<span class="stat-unit">tok used</span>
			</div>
			<div class="stat-row">
				<span class="stat-value">{MAX_CONTEXT_LEN.toLocaleString()}</span>
				<span class="stat-unit">tok max</span>
			</div>
		</div>

		<div class="sidebar-section">
			<span class="section-label" style="font-size:11px">SAMPLING SETTINGS</span>
			<div class="sampling-body">
				{#snippet samplingParam(
					label: string,
					key: keyof typeof samplingState,
					range: {
						min: number;
						max: number;
						step: number;
					},
				)}
					<div class="sampling-param">
						<label class="param-label" for={key}>
							<span>{label}</span>
							<span class="param-value"
								>{key === "topK" ? samplingState[key] : samplingState[key].toFixed(2)}</span
							>
						</label>
						<input id={key} type="range" {...range} bind:value={samplingState[key]} />
					</div>
				{/snippet}

				{@render samplingParam("temperature", "temperature", {
					min: 0.01,
					max: 2.0,
					step: 0.01,
				})}
				{@render samplingParam("top-k", "topK", { min: 1, max: 100, step: 1 })}
				{@render samplingParam("top-p", "topP", { min: 0.01, max: 1, step: 0.01 })}
			</div>
		</div>

		<button class="clear-btn" onclick={clearChat} disabled={isStreaming || messages.length === 0}>
			clear chat
		</button>
	</aside>

	<!-- Chat area -->
	<div class="chat-area">
		<!-- Messages -->
		<div class="messages">
			{#if messages.length === 0}
				<div class="empty-state">
					<div class="empty-title">gemma-3-270m-it</div>
					<div class="empty-sub">running entirely in your browser via WebGPU</div>
				</div>
			{/if}

			{#each messages as msg, i (i)}
				<div class={`message ${msg.role}`}>
					<div class={`avatar ${msg.role}`}>
						{msg.role === "user" ? "U" : "G"}
					</div>
					<div class="message-body">
						<div class="message-meta">
							<span class={`role-label ${msg.role}`}>
								{msg.role === "user" ? "user" : "gemma-3-270m-it"}
							</span>
							{#if msg.tokens}
								<span class="stat">{msg.tokens} tok</span>
							{/if}
							{#if msg.tps}
								<span class="stat">{msg.tps} tok/s</span>
							{/if}
							{#if msg.latencyMs}
								<span class="stat">{msg.latencyMs} ms/tok</span>
							{/if}
							{#if isStreaming && i === messages.length - 1 && msg.role === "assistant"}
								<span class="generating">● generating</span>
							{/if}
						</div>
						{#if msg.role === "user"}
							<div class="user-bubble">{msg.content}</div>
						{:else}
							<div class="prose">
								<!-- eslint-disable-next-line svelte/no-at-html-tags Chat messages should be safe -->
								{@html marked.parse(msg.content)}
								{#if isStreaming && i === messages.length - 1}
									<span class="cursor"></span>
								{/if}
							</div>
						{/if}
					</div>
				</div>
			{/each}

			<div bind:this={chatEndEl}></div>
		</div>

		<!-- Input bar -->
		<div class="input-bar">
			<textarea
				bind:value={inputVal}
				onkeydown={handleKeydown}
				placeholder="Message gemma-3-270m-it… (Enter to send)"
				rows={1}
				// disabled={isStreaming}
			></textarea>
			<button class="send-btn" onclick={send} disabled={isStreaming || !inputVal.trim()}>
				{isStreaming ? "●●●" : "SEND →"}
			</button>
		</div>
		<div class="input-hint">Enter to send · Shift+Enter for newline</div>
	</div>
</div>

<style lang="scss">
	.layout {
		display: flex;
		height: 100%;
		overflow: hidden;
		background: $background-gray;
		font-family: $font-sans;
	}

	/* ── Sidebar ── */
	.sidebar {
		width: 200px;
		flex-shrink: 0;
		border-right: 1px solid $border-gray;
		background: $background-white;
		padding: 16px;
		display: flex;
		flex-direction: column;
		gap: 20px;
		overflow-y: auto;
	}

	.section-label {
		font-size: 9px;
		font-family: $font-mono;
		color: $text-gray;
		letter-spacing: 0.06em;
		margin-bottom: 6px;
	}

	.model-badge {
		padding: 8px 10px;
		background: $background-green;
		border: 1px solid $border-green;
		border-radius: 7px;

		.model-name {
			font-size: 11px;
			font-family: $font-mono;
			color: $accent-green;
			font-weight: 600;
			margin-bottom: 2px;
		}

		.model-sub {
			font-size: 9px;
			font-family: $font-mono;
			color: $text-gray;
		}
	}

	.status-pill {
		font-size: 11px;
		font-family: $font-mono;
		padding: 4px 8px;
		border-radius: 5px;

		&.idle {
			color: $text-gray;
			background: $background-gray;
			border: 1px solid $border-gray;
		}

		&.loading {
			color: $text-blue;
			background: $background-blue;
			border: 1px solid $border-blue;
			animation: pulse 1s ease-in-out infinite;
		}

		&.ready {
			color: $accent-green;
			background: $background-green;
			border: 1px solid $border-green;
		}
	}

	.stat-row {
		display: flex;
		align-items: baseline;
		gap: 5px;
		margin-bottom: 3px;

		.stat-value {
			font-size: 12px;
			font-family: $font-mono;
			color: $text-dark-gray;
			font-weight: 600;
		}

		.stat-unit {
			font-size: 9px;
			font-family: $font-mono;
			color: $text-gray;
		}
	}

	.clear-btn {
		@include transition-all;

		margin-top: auto;
		padding: 6px 12px;
		border-radius: 6px;
		border: 1px solid $border-gray;
		background: $background-white;
		color: $text-dark-gray;
		font-size: 11px;
		font-family: $font-mono;
		cursor: pointer;

		&:hover:not(:disabled) {
			border-color: $text-red;
			color: $text-red;
		}

		&:disabled {
			opacity: 0.4;
			cursor: default;
		}
	}

	/* ── Sampling settings ── */
	.sampling-body {
		margin-top: 14px;
		display: flex;
		flex-direction: column;
		gap: 20px;
	}

	.sampling-param {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.param-label {
		display: flex;
		justify-content: space-between;
		font-size: 10px;
		font-family: $font-mono;
		color: $text-gray;

		.param-value {
			color: $accent-terra;
			font-weight: 600;
		}
	}

	input[type="range"] {
		-webkit-appearance: none;
		appearance: none;
		width: 100%;
		height: 3px;
		border-radius: 2px;
		background: $border-gray;
		outline: none;
		cursor: pointer;

		&::-webkit-slider-thumb {
			-webkit-appearance: none;
			appearance: none;
			width: 12px;
			height: 12px;
			border-radius: 50%;
			background: $accent-terra;
			cursor: pointer;
		}

		&::-moz-range-thumb {
			width: 12px;
			height: 12px;
			border-radius: 50%;
			background: $accent-terra;
			cursor: pointer;
			border: none;
		}
	}

	/* ── Chat area ── */
	.chat-area {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.messages {
		flex: 1;
		overflow-y: auto;
		padding: 24px 28px;
		display: flex;
		flex-direction: column;
		gap: 0;
	}

	.empty-state {
		margin: auto;
		text-align: center;

		.empty-title {
			font-size: 14px;
			font-family: $font-mono;
			color: $accent-green;
			font-weight: 600;
			margin-bottom: 6px;
		}

		.empty-sub {
			font-size: 12px;
			color: $text-gray;
		}
	}

	/* ── Message ── */
	.message {
		display: flex;
		gap: 12px;
		padding: 16px 0;
		border-bottom: 1px solid $border-gray;

		&.user {
			flex-direction: row-reverse;
		}

		&:last-of-type {
			border-bottom: none;
		}
	}

	.avatar {
		width: 28px;
		height: 28px;
		border-radius: 6px;
		flex-shrink: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 10px;
		font-family: $font-mono;

		&.user {
			background: $background-light-blue;
			border: 1px solid $border-light-blue;
			color: $text-dark-blue;
		}

		&.assistant {
			background: $background-green;
			border: 1px solid $border-green;
			color: $accent-green;
		}
	}

	.message-body {
		flex: 1;
		max-width: 100%;

		.user & {
			max-width: 70%;
		}
	}

	.message-meta {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 6px;
	}

	.role-label {
		font-size: 11px;
		font-family: $font-mono;

		&.user {
			color: $text-dark-blue;
		}

		&.assistant {
			color: $accent-green;
		}
	}

	.stat {
		font-size: 9px;
		font-family: $font-mono;
		color: $text-gray;
	}

	.generating {
		font-size: 9px;
		font-family: $font-mono;
		color: $accent-green;
		animation: pulse 1s ease-in-out infinite;
	}

	.user-bubble {
		background: $background-light-blue;
		border: 1px solid $border-light-blue;
		border-radius: 8px;
		padding: 10px 14px;
		font-size: 13px;
		font-family: $font-mono;
		color: $text-black;
		line-height: 1.6;
		white-space: pre-wrap;
	}

	/* ── Markdown prose ── */
	.prose {
		font-size: 13px;
		line-height: 1.7;
		color: $text-black;
		font-family: $font-sans;

		:global(p) {
			margin: 0 0 10px;

			&:last-child {
				margin-bottom: 0;
			}
		}

		:global(strong) {
			font-weight: 600;
			color: $text-black;
		}

		:global(em) {
			font-style: italic;
		}

		:global(code) {
			font-family: $font-mono;
			font-size: 12px;
			background: $background-gray;
			border: 1px solid $border-gray;
			border-radius: 3px;
			padding: 1px 5px;
			color: $accent-green;
		}

		:global(pre) {
			background: $background-gray;
			border: 1px solid $border-gray;
			border-radius: 6px;
			padding: 10px 14px;
			overflow-x: auto;
			margin: 8px 0;

			:global(code) {
				background: none;
				border: none;
				padding: 0;
				font-size: 12px;
				color: $text-dark-gray;
			}
		}

		:global(ul),
		:global(ol) {
			padding-left: 20px;
			margin: 6px 0;
		}

		:global(li) {
			margin-bottom: 3px;
		}

		:global(h1),
		:global(h2),
		:global(h3) {
			font-weight: 700;
			margin: 12px 0 6px;
			color: $text-black;
		}

		:global(h1) {
			font-size: 15px;
		}
		:global(h2) {
			font-size: 14px;
		}
		:global(h3) {
			font-size: 13px;
		}
	}

	.cursor {
		display: inline-block;
		width: 7px;
		height: 13px;
		background: $accent-green;
		border-radius: 1px;
		margin-left: 2px;
		vertical-align: text-bottom;
		animation: blink 0.7s step-end infinite;
	}

	/* ── Input bar ── */
	.input-bar {
		display: flex;
		gap: 10px;
		align-items: flex-end;
		padding: 12px 20px 8px;
		border-top: 1.5px solid $border-gray;
		background: $background-white;

		textarea {
			flex: 1;
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
			max-height: 160px;
			overflow-y: auto;
			transition: border-color 0.15s;

			&:focus {
				border-color: $accent-green;
			}

			&:disabled {
				opacity: 0.5;
			}
		}
	}

	.send-btn {
		@include transition-all;

		padding: 8px 16px;
		border-radius: 6px;
		font-size: 11px;
		font-family: $font-mono;
		font-weight: 600;
		cursor: pointer;
		background: $accent-terra;
		border: 1.5px solid $accent-terra;
		color: $text-white;
		white-space: nowrap;
		flex-shrink: 0;

		&:hover:not(:disabled) {
			opacity: 0.85;
		}

		&:disabled {
			opacity: 0.45;
			cursor: default;
		}
	}

	.input-hint {
		padding: 0 20px 10px;
		font-size: 10px;
		font-family: $font-mono;
		color: $text-gray;
		background: $background-white;
	}

	/* ── Animations ── */
	@keyframes blink {
		0%,
		100% {
			opacity: 1;
		}
		50% {
			opacity: 0;
		}
	}

	@keyframes pulse {
		0%,
		100% {
			opacity: 1;
		}
		50% {
			opacity: 0.5;
		}
	}
</style>
