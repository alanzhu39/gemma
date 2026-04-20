<script lang="ts">
	function activationColor(value: number): string {
		if (value >= 0) {
			const alpha = Math.round(Math.min(value, 1) * 100);
			return `oklch(0.50 0.12 195 / ${alpha}%)`;
		}
		const alpha = Math.round(Math.min(-value, 1) * 100);
		return `oklch(0.55 0.14 32 / ${alpha}%)`;
	}
</script>

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
