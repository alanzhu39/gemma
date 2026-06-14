<script lang="ts">
	import { runDouble } from "$lib/webgpu/double";
	import { runIndexing } from "$lib/webgpu/indexing";
	import { defaultDevice, init, jit, setDebug } from "@jax-js/jax";
	import { nn, numpy as np } from "@jax-js/jax";

	async function runTest() {
		setDebug(3);

		const devices = await init("webgpu");
		if (!devices.includes("webgpu")) {
			alert("WebGPU required but not available!");
			return;
		}

		defaultDevice("webgpu");

		const S = 2;
		const d_embed = 10;
		const d_hidden = 5;

		function foo(x: np.Array): np.Array {
			// x: [S, d_embed]
			const qProj = np.ones([d_hidden, d_embed]); // [d_hidden, d_embed]
			const kProj = np.ones([d_hidden, d_embed]); // [d_hidden, d_embed]
			const vProj = np.ones([d_hidden, d_embed]); // [d_hidden, d_embed]

			const q = np.dot(x.ref, qProj.transpose()); // [S, d_hidden]
			const k = np.dot(x.ref, kProj.transpose()); // [S, d_hidden]
			const v = np.dot(x, vProj.transpose());

			const s = nn.softmax(np.einsum("qd,kd->qk", q, k)); // [S, S]
			const a = np.einsum("qk,vd->qd", s, v); // [S, d_hidden]

			return a;
		}

		const fooJit = jit(foo);

		const x = np.ones([S, d_embed]);

		const y = fooJit(x);

		console.log(y.shape);
		console.log(y.js());
	}
</script>

<h1>Testing different WGSL kernels</h1>

<button onclick={runTest}>Test JIT</button>

<button onclick={runDouble}>Test Double</button>

<button onclick={runIndexing}>Test Indexing</button>

<style lang="scss">
	button {
		border: 1px solid black;
		border-radius: 5px;
		padding: 5px;
		cursor: pointer;
	}
</style>
