import { defaultDevice, init, jit, numpy as np, setDebug } from "@jax-js/jax";

export async function runMatmulJaxJs() {
	setDebug(3);

	const devices = await init("webgpu");
	if (!devices.includes("webgpu")) {
		alert("WebGPU required but not available!");
		return;
	}

	defaultDevice("webgpu");

	function foo(x: np.Array, y: np.Array): np.Array {
		return np.dot(x, y);
	}

	const fooJit = jit(foo);

	const x = np.ones([10, 2000]);
	const y = np.ones([2000, 30]);

	const res = fooJit(x, y);

	console.log(res.shape);
	console.log(res.js());
}
