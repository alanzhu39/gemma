import { defaultDevice, DType, init, nn, numpy as np } from "@jax-js/jax";

const N = 3000;
const d = 256;

async function getInputs() {
	const devices = await init("webgpu");
	if (!devices.includes("webgpu")) {
		alert("WebGPU required but not available!");
	}

	defaultDevice("webgpu");

	const a = np
		.arange(N * d)
		.astype(DType.Float32)
		.reshape([N, d]);

	const q = a.ref.add(1);
	const k = a.ref.add(2);
	const v = a.add(3);
	// TODO: mask, is-causal

	return {
		q,
		k,
		v,
	};
}

export async function runAttentionJaxJs() {
	console.log("Jax-js attention");
	const { q, k, v } = await getInputs();

	const out = nn.dotProductAttention(
		np.expandDims(q, -1),
		np.expandDims(k, -1),
		np.expandDims(v, -1),
	);
	console.log(out.shape);
	const start = performance.now();
	// console.log(await out.data());
	await out.data();
	console.log(`jax-js took ${performance.now() - start} ms`);
}

export async function runFlashAttention() {
	console.log("WebGPU attention");
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		alert("WebGPU required but not available!");
		return;
	}

	// Max workgroup size, this is the stand-in for SRAM size for us
	const maxWorkgroupStorage = adapter.limits.maxComputeWorkgroupStorageSize;
	const device = await adapter.requestDevice({
		requiredLimits: {
			maxComputeWorkgroupStorageSize: maxWorkgroupStorage,
		},
	});
	if (!device) {
		alert("WebGPU required but not available!");
		return;
	}

	const { q, k, v } = await getInputs();

	const dtypeBytes = 4; // f32 = 4 bytes
	const M = maxWorkgroupStorage / dtypeBytes;
	const [N, d] = q.shape;
	const Br = Math.min(d, Math.ceil(M / (4 * d))); // Number of query rows per tile
	const Tr = Math.ceil(N / Br); // Number of query tiles
	const Bc = Math.ceil(M / (4 * d)); // Number of kv column rows per tile
	const Tc = Math.ceil(N / Bc); // Number of kv tiles

	const workgroupSize = [Br, 1, 1];
	const dispatchSize = [Tr, 1, 1] as const;

	const module = device.createShaderModule({
		label: "FlashAttention module",
		code: /* wgsl */ `
			const d: u32 = ${d};
			const N: u32 = ${N};
			const Bc: u32 = ${Bc};
			const Tc: u32 = ${Tc};
			const Br: u32 = ${Br};
			const Tr: u32 = ${Tr};
			const neg_inf: f32 = bitcast<f32>(0xFF7FFFFFu); // technically, smallest f32

      @group(0) @binding(0) var<storage, read_write> o: array<array<f32, d>, N>;
      @group(0) @binding(1) var<storage, read_write> q: array<array<f32, d>, N>;
      @group(0) @binding(2) var<storage, read_write> k: array<array<f32, d>, N>;
      @group(0) @binding(3) var<storage, read_write> v: array<array<f32, d>, N>;

      var<workgroup> k_tile: array<array<f32, d>, Bc>;
      var<workgroup> v_tile: array<array<f32, d>, Bc>;

			var<private> q_tile: array<f32, d>;
			var<private> o_tile: array<f32, d>;
			var<private> s_tile: array<f32, Bc>;
			var<private> m: f32 = neg_inf;
			var<private> l: f32 = 0.0;

      @compute @workgroup_size(${workgroupSize})
      fn flashAttention(
        @builtin(global_invocation_id) global_invocation_id: vec3u,
        @builtin(local_invocation_id) local_invocation_id: vec3u,
        @builtin(workgroup_id) workgroup_id: vec3u,
        @builtin(num_workgroups) num_workgroups: vec3u,
        @builtin(local_invocation_index) local_invocation_index: u32
      ) {
				let i = global_invocation_id.x;

				// Load q_i
				q_tile = q[i];

				// for j in Tc
				for (var j: u32 = 0; j < Tc; j++) {
					// Cooperatively load k_tile, v_tile
					for (var c: u32 = 0; c < (Bc + Br - 1) / Br; c++) {
						let tile_index = c * Br + local_invocation_id.x;
						let arr_index = j * Bc + tile_index;
						if (arr_index < N && tile_index < Bc) {
							k_tile[tile_index] = k[arr_index];
							v_tile[tile_index] = v[arr_index];
						}
					}
					workgroupBarrier();

					// Compute scores
					var m_tile = neg_inf;
					for (var c: u32 = 0; c < Bc; c++) {
						var s: f32 = 0.0;
						for (var x: u32 = 0; x < d; x++) {
							s += q_tile[x] * k_tile[c][x];
						}
						s_tile[c] = s;
						m_tile = max(m_tile, s);
					}

					// Online softmax update
					let m_new = max(m, m_tile);
					let alpha = exp(m - m_new);
					let beta = exp(m_tile - m_new);

					for (var x: u32 = 0; x < d; x++) {
						o_tile[x] *= l * alpha;
					}

					var l_tile: f32 = 0;
          for (var c: u32 = 0; c < Bc; c++) {
            let p = exp(s_tile[c] - m_tile);
            l_tile += p;
						for (var x: u32 = 0; x < d; x++) {
              o_tile[x] += beta * p * v_tile[c][x];
						}
					}

					let l_new = alpha * l + beta * l_tile;
					for (var x = 0u; x < d; x++) {
						o_tile[x] /= l_new;
					}

					l = l_new;
					m = m_new;

					workgroupBarrier();
				}

				// Output o
				if (i < N) {
					o[i] = o_tile;
				}
      }
    `,
	});

	const pipeline = device.createComputePipeline({
		label: "FlashAttention pipeline",
		layout: "auto",
		compute: { module },
	});

	// Buffers
	const workUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
	const oWork = device.createBuffer({ size: N * d * 4, usage: workUsage });
	const qWork = device.createBuffer({ size: N * d * 4, usage: workUsage });
	const kWork = device.createBuffer({ size: N * d * 4, usage: workUsage });
	const vWork = device.createBuffer({ size: N * d * 4, usage: workUsage });
	device.queue.writeBuffer(qWork, 0, await q.data());
	device.queue.writeBuffer(kWork, 0, await k.data());
	device.queue.writeBuffer(vWork, 0, await v.data());

	const oRead = device.createBuffer({
		size: N * d * 4,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});

	// Bind groups
	const bindGroup = device.createBindGroup({
		label: "Buffers",
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: oWork },
			{ binding: 1, resource: qWork },
			{ binding: 2, resource: kWork },
			{ binding: 3, resource: vWork },
		],
	});

	// Command encoder
	const encoder = device.createCommandEncoder();

	const pass = encoder.beginComputePass();
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(...dispatchSize);
	pass.end();

	encoder.copyBufferToBuffer(oWork, 0, oRead, 0, oRead.size);

	const commandBuffer = encoder.finish();

	const start = performance.now();
	device.queue.submit([commandBuffer]);

	// Read results
	await oRead.mapAsync(GPUMapMode.READ);
	console.log(`WebGPU took ${performance.now() - start} ms`);

	const result = new Float32Array(oRead.getMappedRange().slice());

	oRead.destroy();

	// Log output
	console.log("WebGPU");
	console.log(result.length);
	// console.log(result);
}
