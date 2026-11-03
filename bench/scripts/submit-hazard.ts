/**
 * Is batching submits safe? It is not, and the failure is silent.
 *
 * jax-js's buffer lifetime is: encode -> submit -> caller decRef -> at zero refs,
 * buffer.destroy() immediately (webgpu-*.js:907, no pool, no deferral). That is
 * safe today because the submit happens first, so the destroy is queued behind
 * real work.
 *
 * If submits are batched, the order becomes: encode -> decRef -> destroy ->
 * ...later... submit. The command buffer then references a destroyed buffer.
 *
 * Measured directly on this adapter:
 *
 *   submit, no destroy (control)    value = 1    kernel ran
 *   destroy, then submit            value = 0    kernel did NOT run
 *   device lost                     no
 *   error raised                    no
 *
 * A silently zero result, not a crash and not a validation error. A batched
 * implementation would therefore produce wrong numbers that only the
 * correctness gate could catch, and only where they reach the output.
 *
 * Conclusion: submit batching also requires deferring buffer.destroy() until
 * after the batched submit. That is a lifetime change, not a 20-line edit.
 */
export async function probe() {
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) throw new Error("no adapter");
	const dev = await adapter.requestDevice();

	const make = () => {
		const buf = dev.createBuffer({
			size: 256,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		});
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
		const cb = (() => {
			const enc = dev.createCommandEncoder();
			const pass = enc.beginComputePass();
			pass.setPipeline(pipe);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(1);
			pass.end();
			enc.copyBufferToBuffer(buf, 0, st, 0, 4);
			return enc.finish();
		})();
		return { buf, st, cb };
	};

	const run = async (destroyFirst: boolean) => {
		const { buf, st, cb } = make();
		if (destroyFirst) buf.destroy();
		dev.queue.submit([cb]);
		await st.mapAsync(GPUMapMode.READ);
		const v = new Float32Array(st.getMappedRange())[0];
		st.unmap();
		buf.destroy();
		st.destroy();
		return v;
	};

	return {
		controlSubmitThenDestroy: await run(false),
		destroyThenSubmit: await run(true),
		expected: 1,
	};
}
