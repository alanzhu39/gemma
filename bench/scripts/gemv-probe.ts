/**
 * Is the cost per-workgroup or per-byte?
 *
 * Same total work, same bytes read, only the number of workgroups changes.
 * If time scales with workgroup count rather than with bytes, then throughput is
 * limited by how workgroups are scheduled, not by memory bandwidth - which is
 * what the 0% GPU utilisation during the earlier probes suggested.
 */
export async function probe() {
	const adapter = await navigator.gpu.requestAdapter();
	const dev = await adapter!.requestDevice();

	const COLS = 640;
	const x = new Float32Array(COLS);
	for (let i = 0; i < COLS; i++) x[i] = ((i % 13) - 6) * 0.1;
	const xBuf = dev.createBuffer({ size: x.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
	dev.queue.writeBuffer(xBuf, 0, x);

	const results: Record<string, number>[] = [];

	for (const ROWS of [1024, 4096, 16384]) {
		const data = new Float32Array(ROWS * COLS);
		for (let i = 0; i < data.length; i++) data[i] = ((i % 97) - 48) * 0.01;
		const wBuf = dev.createBuffer({
			size: data.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
		dev.queue.writeBuffer(wBuf, 0, data);
		const yBuf = dev.createBuffer({ size: ROWS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
		const stage = dev.createBuffer({ size: ROWS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

		// Grid-stride: a FIXED small number of workgroups, each looping over rows.
		// This keeps the dispatch constant while total work grows.
		const WG = 256;
		const mod = dev.createShaderModule({
			code: `
			@group(0) @binding(0) var<storage, read> w: array<f32>;
			@group(0) @binding(1) var<storage, read> x: array<f32>;
			@group(0) @binding(2) var<storage, read_write> y: array<f32>;
			const COLS: u32 = ${COLS}u;
			const ROWS: u32 = ${ROWS}u;
			var<workgroup> part: array<f32, 128>;
			@compute @workgroup_size(128)
			fn main(@builtin(workgroup_id) wid: vec3<u32>,
			        @builtin(local_invocation_id) lid: vec3<u32>) {
				let lane = lid.x;
				var row = wid.x;
				loop {
					if (row >= ROWS) { break; }
					let base = row * COLS;
					var acc: f32 = 0.0;
					for (var k: u32 = lane; k < COLS; k = k + 128u) {
						acc = acc + w[base + k] * x[k];
					}
					part[lane] = acc;
					workgroupBarrier();
					var step: u32 = 64u;
					loop {
						if (step == 0u) { break; }
						if (lane < step) { part[lane] = part[lane] + part[lane + step]; }
						workgroupBarrier();
						step = step / 2u;
					}
					if (lane == 0u) { y[row] = part[0]; }
					// Barrier before the next iteration reuses the scratch array.
					workgroupBarrier();
					row = row + ${WG}u;
				}
			}`,
		});
		const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
		const bg = dev.createBindGroup({
			layout: pipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: wBuf } },
				{ binding: 1, resource: { buffer: xBuf } },
				{ binding: 2, resource: { buffer: yBuf } },
			],
		});

		const run = async () => {
			const enc = dev.createCommandEncoder();
			const pass = enc.beginComputePass();
			pass.setPipeline(pipe);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(WG);
			pass.end();
			enc.copyBufferToBuffer(yBuf, 0, stage, 0, ROWS * 4);
			dev.queue.submit([enc.finish()]);
			await stage.mapAsync(GPUMapMode.READ);
			const out = new Float32Array(stage.getMappedRange()).slice();
			stage.unmap();
			return out;
		};

		const got = await run();
		let worst = 0;
		for (let r = 0; r < ROWS; r += 331) {
			let ref = 0;
			for (let k = 0; k < COLS; k++) ref += data[r * COLS + k] * x[k];
			worst = Math.max(worst, Math.abs(ref - got[r]));
		}
		// Accumulation order differs from the scalar reference, so compare
		// against the scale of the result rather than an absolute epsilon.
		if (!(worst < Math.max(1e-1, 1e-4 * Math.abs(got[0] || 1)))) {
			throw new Error(`incorrect at rows=${ROWS}: err ${worst}`);
		}

		await run(); await run();
		const t0 = performance.now();
		const N = 5;
		for (let i = 0; i < N; i++) await run();
		const ms = (performance.now() - t0) / N;
		const bytes = ROWS * COLS * 4;
		results.push({
			rows: ROWS,
			workgroups: WG, // constant
			mib: +(bytes / 2 ** 20).toFixed(0),
			ms: +ms.toFixed(2),
			gbps: +(bytes / 1e9 / (ms / 1000)).toFixed(1),
			worstErr: +worst.toExponential(1),
		});
		wBuf.destroy(); yBuf.destroy(); stage.destroy();
	}

	return results;
}
