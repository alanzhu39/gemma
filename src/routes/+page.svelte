<script lang="ts">
	import { resolve } from "$app/paths";
</script>

<div class="home">
	<div class="hero">
		<h1 class="headline">Gemma 3 in your browser</h1>
		<p class="subhead">
			Full inference for
			<a href="https://huggingface.co/google/gemma-3-270m-it" class="link" target="_blank"
				>Gemma 3 270M-Instruct</a
			>
			(FP16), running entirely on-device via WebGPU. Optimized with KV-caching and JIT kernel fusion powered
			by
			<a href="https://jax-js.com/" class="link" target="_blank">jax-js</a>.
		</p>
		<a href="https://github.com/alanzhu39/gemma" class="github-link" target="_blank">
			<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
				<path
					d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.573C20.565 21.795 24 17.295 24 12c0-6.63-5.37-12-12-12z"
				/>
			</svg>
			View source on GitHub
		</a>
	</div>

	<div class="cards">
		<a href={resolve("/inference")} class="card">
			<div class="card-label">Inference</div>
			<div class="card-title">Explore the model</div>
			<p class="card-desc">
				Run a forward pass and inspect attention weights layer by layer. Select any attention head
				to see how the model attends across tokens.
			</p>
			<span class="card-link">Open →</span>
		</a>

		<a href={resolve("/chat")} class="card">
			<div class="card-label">Chat</div>
			<div class="card-title">Talk to the model</div>
			<p class="card-desc">
				Interact with Gemma 3 through a chat interface. Responses stream token-by-token, generated
				entirely on your machine.
			</p>
			<span class="card-link">Open →</span>
		</a>
	</div>
</div>

<style lang="scss">
	.home {
		height: 100%;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 48px 24px;
		background: $background-gray;
		font-family: $font-sans;
	}

	.hero {
		text-align: center;
		max-width: 560px;
		margin-bottom: 45px;

		.headline {
			font-size: 36px;
			font-weight: 700;
			color: $text-black;
			letter-spacing: -0.02em;
			margin: 0 0 16px;
			line-height: 1.15;
		}

		.subhead {
			font-size: 15px;
			color: $text-dark-gray;
			line-height: 1.65;
			margin: 0;
		}

		.link {
			color: $accent-terra;
			text-decoration: none;
			font-weight: 500;

			&:hover {
				text-decoration: underline;
			}
		}

		.github-link {
			@include transition-all;

			display: inline-flex;
			align-items: center;
			gap: 6px;
			margin-top: 20px;
			font-family: $font-mono;
			font-size: 11px;
			color: $text-gray;
			text-decoration: none;

			&:hover {
				color: $text-black;
			}
		}
	}

	.cards {
		display: flex;
		gap: 16px;
		flex-wrap: wrap;
		justify-content: center;
		max-width: 680px;
		width: 100%;
	}

	.card {
		@include transition-all;

		flex: 1;
		min-width: 240px;
		max-width: 300px;
		background: $background-white;
		border: 1.5px solid $border-gray;
		border-radius: 12px;
		padding: 20px 22px;
		text-decoration: none;
		display: flex;
		flex-direction: column;
		gap: 6px;

		&:hover {
			border-color: $accent-terra;
			box-shadow: 0 4px 16px oklch(0.55 0.14 32 / 0.08);
		}

		.card-label {
			font-family: $font-mono;
			font-size: 10px;
			font-weight: 500;
			color: $accent-terra;
			text-transform: uppercase;
			letter-spacing: 0.06em;
		}

		.card-title {
			font-size: 16px;
			font-weight: 700;
			color: $text-black;
			letter-spacing: -0.01em;
		}

		.card-desc {
			font-size: 13px;
			color: $text-dark-gray;
			line-height: 1.6;
			margin: 4px 0 8px;
			flex: 1;
		}

		.card-link {
			font-size: 12px;
			font-weight: 600;
			color: $accent-terra;
			font-family: $font-mono;
		}
	}
</style>
