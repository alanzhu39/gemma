/**
 * How fast is this adapter at a purely compute-bound kernel?
 *
 * Measured memory-bound ops: elementwise 9.4 GB/s, reductions 1.5 GB/s, against
 * a nominal ~192 GB/s. If compute is also crippled, the adapter is simply slow
 * (NVK on Pascal is explicitly labelled "broken" by Mesa). If compute is fast,
 * then only the memory path is degraded, which is a different story.
 */
export async function probe() {
	const adapter = await navigator.gpu.requestAdapter();
	const dev = await adapter!.requestDevice();
	const N = 64 * 1024;
	const out = dev.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
	const staging = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

	// Compute-bound: each thread does many FMAs on registers only.
	const mod = dev.createShaderModule({
		code: `
		@group(0) @binding(0) var<storage, read_write> out: array<f32>;
		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let i = gid.x;
			if (i >= arrayLength(&out)) { return; }
			var a = f32(i) * 1.0001;
			var b = f32(i) * 0.9999;
			for (var k = 0u; k < 4096u; k = k + 1u) {
				a = a * 1.000001 + b * 0.000001;
				b = b * 0.999999 + a * 0.000001;
			}
			out[i] = a + b;
		}`,
	});
	const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
	const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }] });

	const run = async () => {
		const enc = dev.createCommandEncoder();
		const pass = enc.beginComputePass();
		pass.setPipeline(pipe);
		pass.setBindGroup(0, bg);
		pass.dispatchWorkgroups(Math.ceil(N / 256));
		pass.end();
		enc.copyBufferToBuffer(out, 0, staging, 0, 4);
		dev.queue.submit([enc.finish()]);
		await staging.mapAsync(GPUMapMode.READ);
		staging.unmap();
	};

	await run();
	const t0 = performance.now();
	await run();
	const ms = performance.now() - t0;

	// 4096 iterations x 4 flops x 2 statements x N threads
	const flops = N * 4096 * 8;
	out.destroy();
	staging.destroy();
	return { ms: +ms.toFixed(2), gflops: +(flops / 1e9 / (ms / 1000)).toFixed(2) };
}
