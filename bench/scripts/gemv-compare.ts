/**
 * Which GEMV access pattern actually wins?
 *
 * Three variants, same 640 MiB weight matrix, same data, all verified against a
 * CPU reference before timing. The point is to isolate access pattern from
 * everything else: no dispatch-count changes, no vectorisation, no tiling.
 *
 *  1. row-per-thread      - thread t owns output t, walks all 640 columns alone.
 *                           The "obvious" design. A warp touches 32 separate rows,
 *                           80 KiB apart.
 *  2. row-per-thread-vec4 - same, but loads vec4 along the row.
 *  3. warp-per-row        - 32 lanes cooperate on ONE row; a warp reads one
 *                           contiguous run. Reduction across lanes.
 *
 * A barrier inside non-uniform control flow is undefined behaviour and silently
 * skips the kernel, which once produced a 550 GB/s "result" (above the card's
 * 192 GB/s peak - the tell). Every early return here happens before any barrier.
 */
export async function probe() {
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) throw new Error("no adapter");
	const requiredFeatures: GPUFeatureName[] = [];
	if (adapter.features.has("subgroups" as GPUFeatureName)) {
		requiredFeatures.push("subgroups" as GPUFeatureName);
	}
	const dev = await adapter.requestDevice({
		requiredFeatures,
		requiredLimits: {
			maxBufferSize: adapter.limits.maxBufferSize,
			maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
			maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
		},
	});

	const COLS = 640;
	const ROWS = 65536; // 160 MiB f32; well past L2, dispatch stays under 65535
	const data = new Float32Array(ROWS * COLS);
	for (let i = 0; i < data.length; i++) data[i] = ((i % 97) - 48) * 0.01;
	const x = new Float32Array(COLS);
	for (let i = 0; i < COLS; i++) x[i] = ((i % 13) - 6) * 0.1;

	const mkBuf = (arr: Float32Array) => {
		const b = dev.createBuffer({
			size: arr.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
		dev.queue.writeBuffer(b, 0, arr);
		return b;
	};
	const wBuf = mkBuf(data);
	const xBuf = mkBuf(x);
	const yBuf = dev.createBuffer({ size: ROWS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
	const stage = dev.createBuffer({ size: ROWS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

	const bytes = ROWS * COLS * 4;

	const shaders: Record<string, { code: string; wg: number }> = {
		// 1. row-per-thread: one invocation per output row.
		"row-per-thread": {
			wg: Math.ceil(ROWS / 256),
			code: `
			@group(0) @binding(0) var<storage, read> w: array<f32>;
			@group(0) @binding(1) var<storage, read> x: array<f32>;
			@group(0) @binding(2) var<storage, read_write> y: array<f32>;
			const COLS: u32 = ${COLS}u;
			@compute @workgroup_size(256)
			fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
				let row = gid.x;
				if (row >= arrayLength(&y)) { return; }
				let base = row * COLS;
				var acc: f32 = 0.0;
				for (var k: u32 = 0u; k < COLS; k = k + 1u) {
					acc = acc + w[base + k] * x[k];
				}
				y[row] = acc;
			}`,
		},
		// 2. row-per-thread with vec4 loads along the row.
		"row-per-thread-vec4": {
			wg: Math.ceil(ROWS / 256),
			code: `
			@group(0) @binding(0) var<storage, read> w: array<vec4<f32>>;
			@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
			@group(0) @binding(2) var<storage, read_write> y: array<f32>;
			const COLS4: u32 = ${COLS / 4}u;
			@compute @workgroup_size(256)
			fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
				let row = gid.x;
				if (row >= arrayLength(&y)) { return; }
				let base = row * COLS4;
				var acc: vec4<f32> = vec4<f32>(0.0);
				for (var k: u32 = 0u; k < COLS4; k = k + 1u) {
					acc = acc + w[base + k] * x[k];
				}
				y[row] = acc.x + acc.y + acc.z + acc.w;
			}`,
		},
		// 3. warp-per-row: 32 lanes cover one row, reduced via shuffle.
		"warp-per-row": {
			wg: Math.ceil(ROWS / 8), // 256 threads = 8 warps = 8 rows per workgroup
			code: `
			enable subgroups;
			@group(0) @binding(0) var<storage, read> w: array<f32>;
			@group(0) @binding(1) var<storage, read> x: array<f32>;
			@group(0) @binding(2) var<storage, read_write> y: array<f32>;
			const COLS: u32 = ${COLS}u;
			const WARPS: u32 = 8u;
			@compute @workgroup_size(256)
			fn main(@builtin(workgroup_id) wid: vec3<u32>,
			        @builtin(local_invocation_index) li: u32) {
				let warp = li / 32u;
				let lane = li % 32u;
				let row = wid.x * WARPS + warp;
				var acc: f32 = 0.0;
				// No early return before a barrier; the loop simply does nothing
				// when row is out of range, and no barrier is used at all.
				if (row < arrayLength(&y)) {
					let base = row * COLS;
					for (var k: u32 = lane; k < COLS; k = k + 32u) {
						acc = acc + w[base + k] * x[k];
					}
				}
				// Butterfly reduction within the warp: no shared memory, no barrier.
				acc = acc + subgroupShuffleXor(acc, 16u);
				acc = acc + subgroupShuffleXor(acc, 8u);
				acc = acc + subgroupShuffleXor(acc, 4u);
				acc = acc + subgroupShuffleXor(acc, 2u);
				acc = acc + subgroupShuffleXor(acc, 1u);
				if (row < arrayLength(&y) && lane == 0u) {
					y[row] = acc;
				}
			}`,
		},
	};

	const results: Record<string, number | string>[] = [];

	for (const [name, { code, wg }] of Object.entries(shaders)) {
		let pipe: GPUComputePipeline;
		try {
			const mod = dev.createShaderModule({ code });
			pipe = await dev.createComputePipelineAsync({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
		} catch (e) {
			results.push({ variant: name, error: String(e).slice(0, 80) });
			continue;
		}
		const bg = dev.createBindGroup({
			layout: pipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: wBuf } },
				{ binding: 1, resource: { buffer: xBuf } },
				{ binding: 2, resource: { buffer: yBuf } },
			],
		});

		if (wg > 65535) {
			results.push({ variant: name, error: `dispatch ${wg} exceeds 65535` });
			continue;
		}

		const run = async () => {
			const enc = dev.createCommandEncoder();
			const pass = enc.beginComputePass();
			pass.setPipeline(pipe);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(wg);
			pass.end();
			enc.copyBufferToBuffer(yBuf, 0, stage, 0, ROWS * 4);
			dev.queue.submit([enc.finish()]);
			await stage.mapAsync(GPUMapMode.READ);
			const out = new Float32Array(stage.getMappedRange()).slice();
			stage.unmap();
			return out;
		};

		// Verify BEFORE timing. An unverified kernel can be elided, which is how
		// this exercise produced a 550 GB/s result that was never executing.
		const got = await run();
		let worst = 0;
		for (let r = 0; r < ROWS; r += 337) {
			let ref = 0;
			for (let k = 0; k < COLS; k++) ref += data[r * COLS + k] * x[k];
			worst = Math.max(worst, Math.abs(ref - got[r]));
		}
		if (!(worst < 1e-2)) {
			results.push({ variant: name, error: `incorrect, worst ${worst.toExponential(1)}` });
			continue;
		}

		for (let i = 0; i < 8; i++) await run();
		const t0 = performance.now();
		const N = 8;
		for (let i = 0; i < N; i++) await run();
		const ms = (performance.now() - t0) / N;

		const gbps = bytes / 1e9 / (ms / 1000);
		if (gbps > 192) {
			results.push({ variant: name, error: `impossible ${gbps.toFixed(0)} GB/s > 192 peak` });
			continue;
		}

		results.push({
			variant: name,
			ms: +ms.toFixed(2),
			gbps: +gbps.toFixed(1),
			workgroups: wg,
			worstErr: +worst.toExponential(1),
		});
	}

	wBuf.destroy(); xBuf.destroy(); yBuf.destroy(); stage.destroy();
	return { rows: ROWS, mib: +(bytes / 2 ** 20).toFixed(0), results };
}
