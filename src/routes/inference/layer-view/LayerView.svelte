<script lang="ts">
	import AttentionHeatmap from "./AttentionHeatmap.svelte";
	import LayerInfoSidebar from "./LayerInfoSidebar.svelte";

	type Props = {
		selectedLayer: number;
	};

	let { selectedLayer }: Props = $props();
</script>

<div style="flex: 1 1 0%; display: flex; overflow: hidden; min-height: 0px;">
	<!-- Layer info sidebar -->
	<LayerInfoSidebar {selectedLayer} />
	<!-- Attention weights -->
	<AttentionHeatmap />
</div>
