/**
 * The simplest possible measurement: a kernel that burns arithmetic for a known
 * number of iterations, with no memory traffic and no jax-js involved.
 *
 * If this is fast, the GPU computes fine and our GEMV timings are being limited
 * by something else (submission, copies, readback). If this is also slow, the
 * GPU is not executing our work at all despite ANGLE reporting the NVIDIA
 * adapter.
 */
export async function probe() {
	const adapter = await navigator.gpu.requestAdapter();
	const dev = await adapter!.requestDevice();

	const N = 1 << 16;
	const out = dev.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
	const stage = dev.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

	const mod = dev.createShaderModule({
		code: `
		@group(0) @binding(0) var<storage, read_write> out: array<f32>;
		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let i = gid.x;
			if (i >= arrayLength(&out)) { return; }
			var a = f32(i) * 1.0001;
			var b = f32(i) * 0.9999;
			for (var k = 0u; k < 8192u; k = k + 1u) {
				a = a * 1.000001 + b * 0.000001;
				b = b * 0.999999 + a * 0.000001;
			}
			out[i] = a + b;
		}`,
	});
	const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
	const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }] });
	const wg = Math.ceil(N / 256);

	const run = async () => {
		const enc = dev.createCommandEncoder();
		const pass = enc.beginComputePass();
		pass.setPipeline(pipe);
		pass.setBindGroup(0, bg);
		pass.dispatchWorkgroups(wg);
		pass.end();
		enc.copyBufferToBuffer(out, 0, stage, 0, 16);
		dev.queue.submit([enc.finish()]);
		await stage.mapAsync(GPUMapMode.READ);
		const v = new Float32Array(stage.getMappedRange()).slice(0, 4);
		stage.unmap();
		return v;
	};

	const v = await run();
	if (!v.every((x) => Number.isFinite(x))) throw new Error(`kernel produced ${v}`);

	await run();
	const t0 = performance.now();
	const REPS = 5;
	for (let i = 0; i < REPS; i++) await run();
	const ms = (performance.now() - t0) / REPS;

	// 8192 iterations * 4 flops * 2 statements per thread
	const flops = N * 8192 * 8;
	return {
		workgroups: wg,
		ms: +ms.toFixed(2),
		gflops: +(flops / 1e9 / (ms / 1000)).toFixed(1),
		sample: v[0],
	};
}
