<script lang="ts">
	import "./layout.css";
	import favicon from "$lib/assets/favicon.svg";
	import { resolve } from "$app/paths";

	let { children } = $props();
</script>

<svelte:head><link rel="icon" href={favicon} /></svelte:head>
<header class="flex items-center justify-between px-3 py-2">
	<h1>
		<a href={resolve("/")}>gemma</a>
	</h1>
	<nav>
		<a href={resolve("/inference")}>inference</a>
		<a href={resolve("/performance")}>performance</a>
		<a href={resolve("/chat")}>chat</a>
		<a href="https://github.com/alanzhu39/gemma">github</a>
	</nav>
</header>
<!-- TODO: Nav menu: home, inference, performance, chat -->
{@render children()}

<style>
	:global(body) {
		margin: 0;
		padding: 0;

		width: 100dvw;
		height: 100dvh;
	}
</style>
