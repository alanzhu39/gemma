<script lang="ts">
	import "./layout.css";
	import favicon from "$lib/assets/favicon.svg";
	import { resolve } from "$app/paths";
	import { page } from "$app/stores";

	// Font configuration — swap to change the global UI typeface
	const fonts = {
		sans: "'Plus Jakarta Sans', sans-serif",
		mono: "'IBM Plex Mono', monospace",
	};

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

	let activeTab = $derived(TABS.find((t) => $page.url.pathname.startsWith(t.href))?.id ?? "");
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<link
		href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
		rel="stylesheet"
	/>
</svelte:head>

<!-- Header bar -->
<header
	class="flex shrink-0 border-b"
	style="
		height: 44px;
		background: #ffffff;
		border-color: #e8e3dd;
		font-family: {fonts.mono};
		font-size: 11px;
	"
>
	<!-- Brand -->
	<div
		class="flex shrink-0 items-center gap-2.5 border-r px-4 no-underline"
		style="border-color: #e8e3dd;"
	>
		<span
			style="
				font-weight: 700;
				color: oklch(0.55 0.14 32);
				font-size: 13px;
				letter-spacing: -0.01em;
				font-family: {fonts.sans};
			"
		>
			<a href={resolve("/")}> ◈ gemma </a>
		</span>
		<span
			class="rounded px-1.5 py-0.5"
			style="
				background: oklch(0.96 0.05 195);
				color: oklch(0.50 0.12 195);
				border: 1px solid oklch(0.78 0.10 195);
				font-size: 10px;
			"
		>
			WebGPU
		</span>
		<span style="color: oklch(0.50 0.14 142); font-size: 10px;">● ready</span>
	</div>

	<!-- Tabs -->
	<nav class="flex items-stretch justify-center pl-3">
		{#each TABS as tab (tab.id)}
			{@const isActive = tab.id === activeTab}
			<a
				href={resolve(tab.href)}
				class="flex items-center gap-1.5 px-5 no-underline transition-all"
				style="
					border-bottom: 2px solid {isActive ? 'oklch(0.55 0.14 32)' : 'transparent'};
				"
			>
				<span
					style="
						font-size: 12px;
						font-family: {fonts.sans};
						font-weight: 600;
						color: {isActive ? 'oklch(0.55 0.14 32)' : '#a09890'};
					"
				>
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
		background: #faf9f7;
		color: #1c1917;
	}

	.main {
		flex: 1;
		overflow: hidden;
	}
</style>
