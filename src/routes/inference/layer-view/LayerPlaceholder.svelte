<div class="container">
	<div class="placeholder">
		<div class="up-arrow">↑</div>
		<div class="title">Select a layer to inspect its attention weights</div>
		<div class="subtitle">GQA = grouped-query attention · SWA = sliding window attention</div>
	</div>
</div>

<style lang="scss">
	.container {
		flex: 1 1 0%;
		display: flex;
		overflow: hidden;
		min-height: 0px;
		min-width: 100%;

		.placeholder {
			flex: 1 1 0%;
			display: flex;
			align-items: center;
			justify-content: center;
			flex-direction: column;
			gap: 10px;

			.up-arrow {
				font-size: 28px;
				color: $text-light-gray;
			}

			.title {
				font-size: 14px;
				color: $text-gray;
				font-family: $font-sans;
			}

			.subtitle {
				font-size: 12px;
				color: $text-gray;
				font-family: $font-mono;
			}
		}
	}
</style>
