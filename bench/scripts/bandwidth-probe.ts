/**
 * What bandwidth can this card actually sustain, with everything done right?
 *
 * Rules learned the hard way, all enforced here:
 *  - request RAISED device limits: a bare requestDevice() gives 256 MB and
 *    16 KiB workgroup storage, and any larger working set silently does nothing;
 *  - keep dispatch count under 65535, which silently no-ops when exceeded;
 *  - write real data into the source buffer, so the driver cannot skip the work;
 *  - no two threads writing the same address;
 *  - force completion with a readback, so we time the GPU and not submission.
 *
 * Sustained load also matters for clocking: this card idles at 139 MHz and needs
 * continuous work to reach ~1650 MHz. Each measurement here runs repeatedly.
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

	const rows: Record<string, number | string>[] = [];

	const measure = async (label: string, mib: number) => {
		const bytes = mib * 1024 * 1024;
		const nel = Math.floor(bytes / 4);
		const src = dev.createBuffer({ size: nel * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
		const dst = dev.createBuffer({ size: nel * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
		const staging = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

		// Fill with real values in chunks.
		const chunk = new Float32Array(Math.min(nel, 1 << 22)).fill(1.5);
		for (let off = 0; off < nel; off += chunk.length) {
			dev.queue.writeBuffer(src, off * 4, chunk, 0, Math.min(chunk.length, nel - off));
		}

		const WG = 256;
		const per = 256;
		const mod = dev.createShaderModule({
			code: `
			@group(0) @binding(0) var<storage, read> src: array<f32>;
			@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
			const WG: u32 = ${WG}u;
			const PER: u32 = ${per}u;
			@compute @workgroup_size(256)
			fn main(@builtin(workgroup_id) wid: vec3<u32>,
			        @builtin(local_invocation_id) lid: vec3<u32>) {
				let stride = WG * PER;
				var i = wid.x * PER + lid.x;
				loop {
					if (i >= arrayLength(&src)) { break; }
					// Each invocation writes only its own slot: no contention.
					dst[i] = src[i] * 2.0;
					i = i + stride;
				}
			}`,
		});
		const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
		const bg = dev.createBindGroup({
			layout: pipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: src } },
				{ binding: 1, resource: { buffer: dst } },
			],
		});

		const run = async () => {
			const enc = dev.createCommandEncoder();
			const pass = enc.beginComputePass();
			pass.setPipeline(pipe);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(WG);
			pass.end();
			enc.copyBufferToBuffer(dst, 0, staging, 0, 4);
			dev.queue.submit([enc.finish()]);
			await staging.mapAsync(GPUMapMode.READ);
			staging.unmap();
		};

		// Warm up clocks and shaders with sustained work, not a single pass.
		for (let i = 0; i < 10; i++) await run();

		const t0 = performance.now();
		const N = 10;
		for (let i = 0; i < N; i++) await run();
		const ms = (performance.now() - t0) / N;

		// Reads `bytes`, writes `bytes`.
		rows.push({
			label,
			mib: +(nel * 4 / 2 ** 20).toFixed(0),
			ms: +ms.toFixed(2),
			gbps: +((nel * 4 * 2) / 1e9 / (ms / 1000)).toFixed(1),
		});
		src.destroy(); dst.destroy(); staging.destroy();
	};

	await measure("stream 256 MiB", 256);
	await measure("stream 640 MiB", 640);

	return rows;
}
