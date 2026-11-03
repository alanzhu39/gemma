/**
 * Proves the gate's WebGPU error detector actually fires.
 *
 * The gate now listens for `uncapturederror` on every device and fails the run
 * if any arrive. A detector nobody has seen fire is decoration, so this triggers
 * the exact class of bug the listener exists for: a buffer destroyed before its
 * command buffer is submitted.
 *
 * Expected when the listener is installed: at least one captured error
 * mentioning "destroyed". The computation itself returns 0 rather than throwing,
 * which is why the error channel is the only way to notice.
 */
export async function probe() {
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) throw new Error("no adapter");
	const dev = await adapter.requestDevice();

	const caught: string[] = [];
	dev.addEventListener("uncapturederror", (e) => {
		caught.push(String((e as GPUUncapturedErrorEvent).error.message));
	});

	const buf = dev.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
	const st = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
	const mod = dev.createShaderModule({
		code: `
		@group(0) @binding(0) var<storage, read_write> b: array<f32>;
		@compute @workgroup_size(64)
		fn main(@builtin(global_invocation_id) g: vec3<u32>) {
			if (g.x < 64u) { b[g.x] = b[g.x] + 1.0; }
		}`,
	});
	const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
	const bg = dev.createBindGroup({
		layout: pipe.getBindGroupLayout(0),
		entries: [{ binding: 0, resource: { buffer: buf } }],
	});
	const enc = dev.createCommandEncoder();
	const pass = enc.beginComputePass();
	pass.setPipeline(pipe);
	pass.setBindGroup(0, bg);
	pass.dispatchWorkgroups(1);
	pass.end();
	enc.copyBufferToBuffer(buf, 0, st, 0, 4);
	const cb = enc.finish();

	buf.destroy(); // the bug: destroy before the submit that references it
	dev.queue.submit([cb]);
	await new Promise((r) => setTimeout(r, 300));

	const value = await (async () => {
		try {
			await st.mapAsync(GPUMapMode.READ);
			const v = new Float32Array(st.getMappedRange())[0];
			st.unmap();
			return v;
		} catch {
			return NaN;
		}
	})();

	return {
		captured: caught.length,
		first: caught[0]?.slice(0, 80) ?? null,
		value,
		// A silent zero rather than an exception is the whole point.
		silentWrongAnswer: value === 0,
	};
}
