/**
 * Does batching submits help the real step, or only a synthetic loop?
 *
 * Measured cost of a submit call is ~42 us in a tight loop, but jax-js defers
 * dispatch to a sync point (`PendingExecute.submit()`), so the submits happen
 * while the driver may already be working. A tight-loop measurement cannot see
 * that; it puts every submit next to the previous one with no GPU work between.
 *
 * This compares two shapes that both do real work:
 *  - "interleaved": submit, tiny kernel, submit, tiny kernel ... (like a step)
 *  - "batched":     all encoders, one submit at the end (what batching would do)
 *
 * If batching wins only when there is no work between submits, it will not help
 * a real step, which always has work between them.
 */
export async function probe() {
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) throw new Error("no adapter");
	const dev = await adapter.requestDevice({
		requiredLimits: {
			maxBufferSize: adapter.limits.maxBufferSize,
			maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
			maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
		},
	});

	const N = 256;
	const buf = dev.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
	const stage = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

	// A kernel with a little real work: ~4k FMAs per invocation, 1 workgroup.
	const mod = dev.createShaderModule({
		code: `
		@group(0) @binding(0) var<storage, read_write> b: array<f32>;
		@compute @workgroup_size(64)
		fn main(@builtin(global_invocation_id) g: vec3<u32>) {
			if (g.x >= 64u) { return; }
			var a = b[g.x] + 1.0;
			for (var k = 0u; k < 64u; k = k + 1u) { a = a * 1.0001 + 0.5; }
			b[g.x] = a;
		}`,
	});
	const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
	const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });

	const addPass = (enc: GPUCommandEncoder) => {
		const p = enc.beginComputePass();
		p.setPipeline(pipe);
		p.setBindGroup(0, bg);
		p.dispatchWorkgroups(1);
		p.end();
	};

	const timeIt = async (label: string, fn: () => Promise<void>) => {
		for (let i = 0; i < 3; i++) await fn();
		const t0 = performance.now();
		const reps = 10;
		for (let i = 0; i < reps; i++) await fn();
		return { label, ms: (performance.now() - t0) / reps };
	};

	const results: Record<string, number | string>[] = [];

	// Today's shape: one encoder + one submit per kernel, real work between.
	for (const n of [64, 469]) {
		const r = await timeIt(`interleaved x${n}`, async () => {
			for (let i = 0; i < n; i++) {
				const enc = dev.createCommandEncoder();
				addPass(enc);
				dev.queue.submit([enc.finish()]);
			}
			const enc = dev.createCommandEncoder();
			enc.copyBufferToBuffer(buf, 0, stage, 0, 4);
			dev.queue.submit([enc.finish()]);
			await stage.mapAsync(GPUMapMode.READ);
			stage.unmap();
		});
		results.push({ case: r.label, ms: +r.ms.toFixed(2), usPerKernel: +((r.ms * 1000) / n).toFixed(1) });
	}

	// Batching candidate: collect all encoders, one submit at the end.
	for (const n of [64, 469]) {
		const r = await timeIt(`batched x${n}`, async () => {
			const buffers: GPUCommandBuffer[] = [];
			for (let i = 0; i < n; i++) {
				const enc = dev.createCommandEncoder();
				addPass(enc);
				buffers.push(enc.finish());
			}
			const enc = dev.createCommandEncoder();
			enc.copyBufferToBuffer(buf, 0, stage, 0, 4);
			buffers.push(enc.finish());
			if (buffers.length > 32) {
				// WebGPU allows many buffers per submit, but chunk conservatively.
				for (let i = 0; i < buffers.length; i += 32) dev.queue.submit(buffers.slice(i, i + 32));
			} else {
				dev.queue.submit(buffers);
			}
			await stage.mapAsync(GPUMapMode.READ);
			stage.unmap();
		});
		results.push({ case: r.label, ms: +r.ms.toFixed(2), usPerKernel: +((r.ms * 1000) / n).toFixed(1) });
	}

	buf.destroy();
	stage.destroy();
	return results;
}
