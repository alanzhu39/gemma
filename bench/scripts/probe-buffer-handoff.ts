export async function probe() {
	const out: Record<string, unknown> = {};
	const { init, defaultDevice, numpy: np, getWebGPUDevice } = await import("@jax-js/jax");
	await init("webgpu");
	defaultDevice("webgpu");
	const dev = getWebGPUDevice();
	out.deviceFeatures = [...dev.features];
	out.maxBufferSize = dev.limits.maxBufferSize;

	// Can a jax-js array hand out its buffer without dying?
	const a = np.zeros([8], { dtype: np.float32 });
	out.aRcBefore = a.refCount;
	const buf = a.ref.gpuBufferSync();
	out.aRcAfterBorrow = a.refCount;
	out.bufSize = buf.size;
	out.aStillUsable = a.shape.join(",");

	// Write through the raw buffer and read it back through jax.
	const enc = dev.createCommandEncoder();
	const stage = dev.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
	const mod = dev.createShaderModule({
		code: `@group(0) @binding(0) var<storage, read_write> y: array<f32>;
		@compute @workgroup_size(8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
			if (g.x < 8u) { y[g.x] = f32(g.x) * 3.0; }
		}`,
	});
	const pipe = await dev.createComputePipelineAsync({
		layout: "auto",
		compute: { module: mod, entryPoint: "main" },
	});
	const bg = dev.createBindGroup({
		layout: pipe.getBindGroupLayout(0),
		entries: [{ binding: 0, resource: { buffer: buf } }],
	});
	const pass = enc.beginComputePass();
	pass.setPipeline(pipe);
	pass.setBindGroup(0, bg);
	pass.dispatchWorkgroups(1);
	pass.end();
	enc.copyBufferToBuffer(buf, 0, stage, 0, 32);
	dev.queue.submit([enc.finish()]);
	await stage.mapAsync(GPUMapMode.READ);
	out.stageValues = Array.from(new Float32Array(stage.getMappedRange()));
	stage.unmap();

	// And critically: does jax-js see the values we wrote into its own buffer?
	out.jaxSees = Array.from((await a.data()) as Float32Array);
	out.bufDestroyed = false;

	// Simulate the real shape: a [1,640] row borrowed from a bigger latent.
	const lat = np.ones([4, 640], { dtype: np.float32 });
	const row = lat.ref.slice([-1]).flatten();
	out.rowContiguous = row.shape.join(",");
	const rowBuf = row.ref.gpuBufferSync();
	out.rowBufSize = rowBuf.size;
	out.rowRcAfter = row.refCount;

	// Try to build an Array over a buffer jax did not allocate.
	stage.destroy();
	return out;
}
