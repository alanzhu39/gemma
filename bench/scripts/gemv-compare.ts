/**
 * Which GEMV access pattern wins, and how the answer depends on working set.
 *
 * Three variants, identical data, all verified against a CPU reference before any
 * time is reported. The sweep matters as much as the variants: an earlier version
 * measured at 160 MiB and reported 50 GB/s while another measured 640 MiB and
 * reported 121 GB/s for what looked like the same kernel. Both were right. At
 * 160 MiB there are too few workgroups to hide latency, so the kernel is
 * latency-bound; the real workload is 640 MiB and is bandwidth-bound.
 *
 *  1. row-per-thread      - thread t owns output t and walks all columns alone.
 *                           A warp touches 32 rows, 2560 B apart.
 *  2. row-per-thread-vec4 - same, with vec4 loads along the row.
 *  3. warp-per-row        - 32 lanes share one row; reduced with subgroup shuffles.
 *
 * A barrier inside non-uniform control flow is undefined behaviour and silently
 * skips the kernel, which produced a 550 GB/s "result" once. The tell: anything
 * above the card's ~192 GB/s peak is a probe bug. No barriers are used here at
 * all, and results above peak are rejected rather than reported.
 */
export async function probe() {
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) throw new Error("no adapter");
	const requiredFeatures: GPUFeatureName[] = [];
	if (adapter.features.has("subgroups" as GPUFeatureName)) {
		requiredFeatures.push("subgroups" as GPUFeatureName);
	} else {
		throw new Error("adapter has no subgroups feature; warp-per-row needs it");
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
	const out: Record<string, number | string>[] = [];

	for (const ROWS of [65536, 131072, 262144]) {
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

		const variants: { name: string; wg: number; code: string }[] = [
			{
				name: "row-per-thread",
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
			{
				name: "row-per-thread-vec4",
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
			{
				name: "warp-per-row",
				wg: Math.ceil(ROWS / 8),
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
					// No early return before a barrier because there are no
					// barriers here at all; out-of-range rows simply compute 0.
					if (row < arrayLength(&y)) {
						let base = row * COLS;
						for (var k: u32 = lane; k < COLS; k = k + 32u) {
							acc = acc + w[base + k] * x[k];
						}
					}
					acc = acc + subgroupShuffleXor(acc, 16u);
					acc = acc + subgroupShuffleXor(acc, 8u);
					acc = acc + subgroupShuffleXor(acc, 4u);
					acc = acc + subgroupShuffleXor(acc, 2u);
					acc = acc + subgroupShuffleXor(acc, 1u);
					if (row < arrayLength(&y) && lane == 0u) { y[row] = acc; }
				}`,
			},
		];

		for (const v of variants) {
			if (v.wg > 65535) {
				out.push({ rows: ROWS, variant: v.name, error: `dispatch ${v.wg} > 65535` });
				continue;
			}
			let pipe: GPUComputePipeline;
			try {
				const mod = dev.createShaderModule({ code: v.code });
				pipe = await dev.createComputePipelineAsync({
					layout: "auto",
					compute: { module: mod, entryPoint: "main" },
				});
			} catch (e) {
				out.push({ rows: ROWS, variant: v.name, error: `shader: ${String(e).slice(0, 60)}` });
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

			const run = async () => {
				const enc = dev.createCommandEncoder();
				const pass = enc.beginComputePass();
				pass.setPipeline(pipe);
				pass.setBindGroup(0, bg);
				pass.dispatchWorkgroups(v.wg);
				pass.end();
				enc.copyBufferToBuffer(yBuf, 0, stage, 0, ROWS * 4);
				dev.queue.submit([enc.finish()]);
				await stage.mapAsync(GPUMapMode.READ);
				const arr = new Float32Array(stage.getMappedRange()).slice();
				stage.unmap();
				return arr;
			};

			// Verify before timing: an unverified kernel can be elided entirely.
			const got = await run();
			let worst = 0;
			let bad = 0;
			for (let r = 0; r < ROWS; r += 997) {
				let ref = 0;
				for (let k = 0; k < COLS; k++) ref += data[r * COLS + k] * x[k];
				const d = Math.abs(ref - got[r]);
				if (d > 1e-2) bad++;
				worst = Math.max(worst, d);
			}
			if (bad > 0) {
				out.push({ rows: ROWS, variant: v.name, error: `incorrect: ${bad} bad rows, worst ${worst.toExponential(1)}` });
				continue;
			}

			for (let i = 0; i < 8; i++) await run();
			const t0 = performance.now();
			const N = 8;
			for (let i = 0; i < N; i++) await run();
			const ms = (performance.now() - t0) / N;

			const bytes = ROWS * COLS * 4;
			const gbps = bytes / 1e9 / (ms / 1000);
			if (gbps > 192) {
				out.push({ rows: ROWS, variant: v.name, error: `impossible ${gbps.toFixed(0)} GB/s > 192 peak` });
				continue;
			}
			out.push({
				rows: ROWS,
				mib: +(bytes / 2 ** 20).toFixed(0),
				variant: v.name,
				workgroups: v.wg,
				ms: +ms.toFixed(2),
				gbps: +gbps.toFixed(1),
			});
		}

		wBuf.destroy();
		xBuf.destroy();
		yBuf.destroy();
		stage.destroy();
	}

	return out;
}
