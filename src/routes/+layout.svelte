<script lang="ts">
	import "./layout.css";
	import favicon from "$lib/assets/favicon.svg";
	import { resolve } from "$app/paths";
	import { page } from "$app/state";

	const TABS = [
		{
			id: "inference",
			label: "Inference",
			href: "/inference",
		},
		{
			id: "performance",
			label: "Performance",
			href: "/performance",
		},
		{ id: "chat", label: "Chat", href: "/chat" },
	] as const;

	let { children } = $props();

	let activeTab = $derived(TABS.find((t) => page.url.pathname.startsWith(t.href))?.id ?? "");
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<link
		href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
		rel="stylesheet"
	/>
</svelte:head>

<!-- Header bar -->
<header class="header">
	<!-- Brand -->
	<div class="brand">
		<span class="title">
			<a href={resolve("/")}> ◈ gemma </a>
		</span>
		<span class="webgpu"> WebGPU </span>
		<span class="status">● ready</span>
	</div>

	<!-- Tabs -->
	<nav class="tabs">
		{#each TABS as tab (tab.id)}
			{@const isActive = tab.id === activeTab}
			<a href={resolve(tab.href)} class={`tab ${isActive ? "active" : ""}`}>
				<span class={`label ${isActive ? "active" : ""}`}>
					{tab.label}
				</span>
			</a>
		{/each}
	</nav>
</header>

<!-- Page content -->
<main class="main">
	{@render children()}
</main>

<style lang="scss">
	:global(*, *::before, *::after) {
		box-sizing: border-box;
	}

	:global(body) {
		margin: 0;
		padding: 0;
		width: 100dvw;
		height: 100dvh;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		background: $background-gray;
		color: $text-black;
	}

	.header {
		display: flex;
		flex-shrink: 0;
		height: 44px;
		border-bottom: 1px solid $border-gray;
		background: $background-white;
		font-family: $font-mono;
		font-size: 11px;

		.brand {
			display: flex;
			gap: 10px;
			align-items: center;
			padding-inline: 16px;
			text-decoration-line: none;
			border-right: 1px solid $border-gray;

			.title {
				font-weight: 700;
				color: $accent-terra;
				font-size: 13px;
				letter-spacing: -0.01em;
				font-family: $font-sans;
			}

			.webgpu {
				border-radius: 4px;
				padding: 2px 6px;
				background: $background-teal;
				color: $text-teal;
				border: 1px solid $border-teal;
				font-size: 10px;
			}

			.status {
				color: $accent-green;
				font-size: 10px;
			}
		}

		.tabs {
			display: flex;
			align-items: stretch;
			justify-content: center;
			padding-left: 12px;

			.tab {
				@include transition-all;

				display: flex;
				align-items: center;
				gap: 6px;
				padding-inline: 20px;
				text-decoration-line: none;
				border-bottom: 2px solid transparent;

				&.active {
					border-bottom: 2px solid $accent-terra;
				}

				.label {
					font-size: 12px;
					font-family: $font-sans;
					font-weight: 600;
					color: $text-gray;

					&.active {
						color: $accent-terra;
					}
				}
			}
		}
	}

	.main {
		flex: 1;
		overflow: hidden;
	}
</style>
