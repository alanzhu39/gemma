<script lang="ts">
	import AttentionHeatmap from "./AttentionHeatmap.svelte";
	import LayerInfoSidebar from "./LayerInfoSidebar.svelte";
</script>

<div class="container">
	<!-- Layer info sidebar -->
	<LayerInfoSidebar />
	<!-- Attention weights -->
	<AttentionHeatmap />
</div>

<style lang="scss">
	.container {
		flex: 1 1 0%;
		display: flex;
		overflow: hidden;
		min-height: 0px;
	}
</style>
