<script lang="ts">
	export type DownloadState = {
		visible: boolean;
		loadedBytes: number;
		totalBytes?: number;
	};

	let { loadedBytes, totalBytes, visible }: DownloadState = $props();

	function formatMB(bytes: number) {
		return (bytes / 1024 / 1024).toFixed(0);
	}

	const progress = $derived(totalBytes ? (loadedBytes / totalBytes) * 100 : null);
</script>

{#if visible}
	<div class="toast">
		<div class="toast-header">
			<span class="icon">↓</span>
			<span class="title">Downloading model weights</span>
		</div>
		<div class="progress-track">
			{#if progress !== null}
				<div class="progress-fill" style="width: {progress}%"></div>
			{:else}
				<div class="progress-indeterminate"></div>
			{/if}
		</div>
		<div class="toast-sub">
			{#if progress !== null}
				{formatMB(loadedBytes)} MB / {formatMB(totalBytes!)} MB · {progress.toFixed(0)}%
			{:else}
				{formatMB(loadedBytes)} MB downloaded…
			{/if}
		</div>
	</div>
{/if}

<style lang="scss">
	.toast {
		position: fixed;
		bottom: 20px;
		right: 20px;
		z-index: 1000;
		background: $background-white;
		border: 1.5px solid $border-gray;
		border-radius: 10px;
		padding: 12px 16px;
		min-width: 260px;
		box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
		font-family: $font-mono;

		.toast-header {
			display: flex;
			align-items: center;
			gap: 7px;
			margin-bottom: 9px;

			.icon {
				font-size: 13px;
				color: $accent-terra;
			}

			.title {
				font-size: 12px;
				font-weight: 600;
				color: $text-black;
				font-family: $font-sans;
			}
		}

		.progress-track {
			height: 5px;
			background: $background-gray;
			border-radius: 3px;
			overflow: hidden;
			margin-bottom: 7px;

			.progress-fill {
				height: 100%;
				background: $accent-terra;
				border-radius: 3px;
				transition: width 0.2s ease;
			}

			.progress-indeterminate {
				height: 100%;
				width: 40%;
				background: $accent-terra;
				border-radius: 3px;
				animation: slide 1.2s ease-in-out infinite;
			}
		}

		.toast-sub {
			font-size: 10px;
			color: $text-gray;
		}
	}

	@keyframes slide {
		0% { transform: translateX(-100%); }
		100% { transform: translateX(350%); }
	}
</style>
