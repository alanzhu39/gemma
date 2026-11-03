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
 *
 * Two modes, because the distinction changes the number by ~2x:
 *  - "copy" reads N bytes and writes N bytes. Total traffic 2N.
 *  - "read" reads N bytes and writes one output word per workgroup. This is the
 *    right model for a matvec, which reads a weight matrix and emits a small
 *    output, and reads run faster when writes are not competing for bandwidth.
 *
 * Both verify their output against a CPU reference before reporting a time. An
 * unverified kernel can be silently elided by the driver, which is how earlier
 * versions of this probe "measured" thousands of GB/s.
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

	const measure = async (label: string, mib: number, mode: "copy" | "read") => {
		const bytes = mib * 1024 * 1024;
		const nel = Math.floor(bytes / 4);
		const src = dev.createBuffer({ size: nel * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
		// In read mode the output is one word per workgroup, so the buffer stays small.
		const outBytes = mode === "copy" ? nel * 4 : 256 * 4;
		const dst = dev.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
		const staging = dev.createBuffer({ size: outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

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
			const MODE: u32 = ${mode === "copy" ? 0 : 1}u;
			var<workgroup> part: array<f32, 256>;

			@compute @workgroup_size(256)
			fn main(@builtin(workgroup_id) wid: vec3<u32>,
			        @builtin(local_invocation_id) lid: vec3<u32>) {
				let stride = WG * PER;
				var i = wid.x * PER + lid.x;
				if (MODE == 0u) {
					// Copy: each invocation owns its own destination slot.
					loop {
						if (i >= arrayLength(&src)) { break; }
						dst[i] = src[i] * 2.0;
						i = i + stride;
					}
				} else {
					// Read-only: accumulate, then one reduction and one store per
					// workgroup. Reads the full buffer, writes 256 words total.
					var acc: f32 = 0.0;
					loop {
						if (i >= arrayLength(&src)) { break; }
						acc = acc + src[i];
						i = i + stride;
					}
					part[lid.x] = acc;
					workgroupBarrier();
					var step: u32 = 128u;
					loop {
						if (step == 0u) { break; }
						if (lid.x < step) { part[lid.x] = part[lid.x] + part[lid.x + step]; }
						workgroupBarrier();
						step = step / 2u;
					}
					if (lid.x == 0u) { dst[wid.x] = part[0]; }
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

		// Timed path: 4 bytes only, purely to force completion. Copying the whole
		// output to a mappable buffer inside the timed region added a second
		// device copy and a sync per iteration, which is why an earlier version of
		// this measured 8.7 GB/s for a copy and 131 GB/s for the same work.
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

		// Verify before timing: a kernel whose result is never checked can be
		// elided by the driver, which produced thousands-of-GB/s nonsense earlier.
		const verify = async () => {
			const enc = dev.createCommandEncoder();
			const pass = enc.beginComputePass();
			pass.setPipeline(pipe);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(WG);
			pass.end();
			enc.copyBufferToBuffer(dst, 0, staging, 0, mode === "copy" ? 4 : outBytes);
			dev.queue.submit([enc.finish()]);
			await staging.mapAsync(GPUMapMode.READ);
			const v = new Float32Array(staging.getMappedRange()).slice();
			staging.unmap();
			return v;
		};
		const got = await verify();
		if (mode === "copy") {
			if (Math.abs(got[0] - 3) > 1e-3) throw new Error(`${label}: expected 3, got ${got[0]}`);
		} else {
			// Every element is 1.5, split across WG parts, so the total is 1.5*nel.
			const total = got.reduce((a, b) => a + b, 0);
			const expect = 1.5 * nel;
			if (Math.abs(total - expect) / expect > 1e-3) {
				throw new Error(`${label}: sum ${total} != expected ${expect}`);
			}
		}

		// Warm up clocks and shaders with sustained work, not a single pass.
		for (let i = 0; i < 10; i++) await run();

		const t0 = performance.now();
		const N = 10;
		for (let i = 0; i < N; i++) await run();
		const ms = (performance.now() - t0) / N;

		// Copy moves 2N bytes; read mode moves N plus 1 KiB.
		const traffic = mode === "copy" ? nel * 4 * 2 : nel * 4;
		rows.push({
			label,
			mode,
			mib: +(nel * 4 / 2 ** 20).toFixed(0),
			ms: +ms.toFixed(2),
			gbps: +(traffic / 1e9 / (ms / 1000)).toFixed(1),
			verified: 1,
		});
		src.destroy(); dst.destroy(); staging.destroy();
	};

	await measure("copy 256 MiB", 256, "copy");
	await measure("copy 640 MiB", 640, "copy");
	await measure("read 640 MiB", 640, "read");

	return rows;
}
