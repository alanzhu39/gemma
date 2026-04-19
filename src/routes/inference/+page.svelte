<script lang="ts">
	import { runInference } from "$lib/gemma3/inference";
	import { fromSafetensors, type Gemma3 } from "$lib/gemma3/model";
	import { AutoTokenizer } from "@huggingface/transformers";
	import { defaultDevice, init, tree } from "@jax-js/jax";
	import { cachedFetch, safetensors } from "@jax-js/loaders";

	let downloadProgress = $state<number | null>(null);

	let context = "";

	async function run() {
		const devices = await init("webgpu");
		if (!devices.includes("webgpu")) {
			alert("WebGPU required but not available!");
			return;
		}
		defaultDevice("webgpu");

		console.log("Running...");

		const weightsUrl =
			"https://huggingface.co/alanzhu39/gemma-3-270m-it-f16/resolve/main/model.safetensors";
		const data = await cachedFetch(weightsUrl, {}, (progress) => {
			downloadProgress = progress.totalBytes
				? Math.round((progress.loadedBytes / progress.totalBytes) * 100)
				: null;
		});
		const file = safetensors.parse(data);
		const weights: Gemma3 = fromSafetensors(file);
		// const text = "Plants create energy through a process known as";
		// const text = "The capital of France is";
		const text = `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean nulla arcu, mollis eget elit ut, sodales rhoncus magna. Proin eleifend dolor non magna rhoncus, non finibus enim lobortis. Phasellus fermentum, quam vel laoreet auctor, mi felis viverra arcu, in volutpat felis lorem eget lacus. Nullam congue hendrerit dignissim. Fusce ut elit consectetur, facilisis dui in, convallis turpis. Nam quam massa, blandit eget dolor a, dignissim tincidunt est. Cras ultricies iaculis arcu, quis semper velit volutpat laoreet.
Phasellus condimentum, tellus ac malesuada vehicula, risus ante ullamcorper dui, quis imperdiet dolor leo nec erat. Integer quis tellus eu diam sodales ornare consectetur nec augue. Nullam sagittis efficitur ligula. Nunc ac auctor eros. Etiam luctus placerat cursus. Praesent vitae quam imperdiet lectus scelerisque sollicitudin. Vestibulum id turpis nec lorem semper scelerisque vel sit amet erat. Maecenas maximus vel nulla eget mattis. Aenean vel tellus a odio laoreet luctus. Sed ut egestas ex. Pellentesque feugiat at augue in pellentesque.
In hac habitasse platea dictumst. Duis aliquet mollis consequat. Duis elit augue, aliquet in ultrices congue, placerat sit amet tortor. Quisque in odio ut tellus posuere suscipit. Cras laoreet rutrum consequat. Pellentesque venenatis mi dolor, a luctus augue sagittis sit amet. In rutrum dui risus, sed pulvinar purus aliquet in. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Nunc fermentum ex eu massa fringilla congue sit amet laoreet massa. Suspendisse leo magna, pellentesque fringilla dui at, dictum pulvinar massa. Nunc non elit feugiat, auctor sem vel, varius quam. Curabitur tincidunt, purus vel placerat accumsan, ligula ligula dignissim odio, eget ullamcorper nibh justo a justo. Ut eu pretium tellus.
Morbi at erat eget nisi sodales cursus. Integer ultricies efficitur efficitur. Maecenas eget commodo arcu, a rhoncus nibh. Nullam quis purus nec tellus feugiat porttitor ut ut nibh. Aliquam aliquam enim aliquet erat sagittis varius. Ut dictum tellus sit amet facilisis feugiat. Nulla eget rhoncus dui. Integer molestie orci nec neque rhoncus, ut gravida enim venenatis. Nulla vehicula, mauris in viverra elementum, mi nibh efficitur leo, vel accumsan diam libero id mauris. Nunc non ligula quis dolor vulputate dapibus. Nam tincidunt ipsum at libero euismod pharetra.
Fusce molestie lorem sit amet justo accumsan, sit amet tempor ligula semper. Etiam in dolor sit amet nunc egestas ullamcorper. Nullam sit amet malesuada turpis. Fusce aliquam tristique erat sed auctor. Aenean non velit turpis. Aliquam quis euismod mi. Integer lacus erat, vehicula ac bibendum sit amet, eleifend blandit odio. Praesent sagittis sodales semper. Aenean sit amet posuere enim. Aenean est tortor, accumsan nec ante nec, lacinia blandit purus.`;

		console.log("Loaded weights");

		const tokenizer = await AutoTokenizer.from_pretrained("alanzhu39/gemma-3-270m-it-f16");

		await runInference(weights, tokenizer, text, 10);

		tree.dispose(weights);
	}
</script>

<h1>Inference</h1>
<!-- TODO: model architecture -->
<!-- TODO: layer activations -->
<!-- TODO: text input context -->
<div class="flex flex-col p-3">
	<h2>Context</h2>
	<input bind:value={context} type="text" class="border-2 border-r-2 border-black" />
	<button onclick={run}>Run</button>
	{#if downloadProgress != null}
		<div>Download progress: {downloadProgress}%</div>
	{/if}
</div>
