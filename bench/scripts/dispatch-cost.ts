/**
 * What does one dispatch actually cost, on the host and on the GPU?
 *
 * The orchestrator derived 86-174 us/dispatch by subtracting traced GPU time
 * from wall clock, and had to give bounds because the GPU side is distorted by
 * idle clocks. This measures it directly instead, which removes the bounds.
 *
 * Three cases, all ending in a readback so work is genuinely complete:
 *  - N empty dispatches: what one dispatch costs when it does nothing.
 *  - N dispatches of a trivial kernel: the same, with a shader that actually runs.
 *  - 1 dispatch doing N times the work: proves the cost scales with dispatch
 *    count rather than with work.
 *
 * If dispatch is expensive relative to work, launching fewer, larger kernels is
 * the win - which is the whole argument for fusing the 218 norm dispatches.
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

	const buf = dev.createBuffer({ size: 1024 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
	const stage = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

	// Trivial kernel: one workgroup, one element. Real but negligible work.
	const mod = dev.createShaderModule({
		code: `
		@group(0) @binding(0) var<storage, read_write> b: array<f32>;
		@compute @workgroup_size(64)
		fn main(@builtin(global_invocation_id) g: vec3<u32>) {
			if (g.x < 64u) { b[g.x] = b[g.x] + 1.0; }
		}`,
	});
	const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
	const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });

	const timeIt = async (label: string, fn: () => Promise<void>, reps = 5) => {
		for (let i = 0; i < 3; i++) await fn();
		const t0 = performance.now();
		for (let i = 0; i < reps; i++) await fn();
		return { label, ms: (performance.now() - t0) / reps };
	};

	const results: Record<string, number | string>[] = [];

	for (const n of [1, 100, 1000]) {
		// N trivial dispatches in ONE command buffer, one submit, one readback.
		const r = await timeIt(`trivial x${n}`, async () => {
			const enc = dev.createCommandEncoder();
			for (let i = 0; i < n; i++) {
				const p = enc.beginComputePass();
				p.setPipeline(pipe);
				p.setBindGroup(0, bg);
				p.dispatchWorkgroups(1);
				p.end();
			}
			enc.copyBufferToBuffer(buf, 0, stage, 0, 4);
			dev.queue.submit([enc.finish()]);
			await stage.mapAsync(GPUMapMode.READ);
			stage.unmap();
		});
		results.push({ case: r.label, ms: +r.ms.toFixed(2), usPerDispatch: +((r.ms * 1000) / n).toFixed(1) });
	}

	// Same total work, one dispatch: each of n workgroups does one element.
	for (const n of [1, 100, 1000]) {
		const r = await timeIt(`batched x${n} (1 dispatch)`, async () => {
			const enc = dev.createCommandEncoder();
			const p = enc.beginComputePass();
			p.setPipeline(pipe);
			p.setBindGroup(0, bg);
			p.dispatchWorkgroups(n);
			p.end();
			enc.copyBufferToBuffer(buf, 0, stage, 0, 4);
			dev.queue.submit([enc.finish()]);
			await stage.mapAsync(GPUMapMode.READ);
			stage.unmap();
		});
		results.push({ case: r.label, ms: +r.ms.toFixed(2) });
	}

	buf.destroy();
	stage.destroy();
	return results;
}
