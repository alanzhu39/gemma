export async function runIndexing() {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter?.requestDevice();
	if (!device) {
		alert("WebGPU required but not available!");
		return;
	}

	const dispatchSize = [4, 3, 2] as const;
	const workgroupSize = [2, 3, 4];

	const reduceMul = (arr: readonly number[]) => arr.reduce((a, b) => a * b);

	const numThreadsPerWorkgroup = reduceMul(workgroupSize);

	// For each thread, we want to output:
	// - workgroup_id: vec3u
	// - global_invocation_id: vec3u
	// - local_invocation_id: vec3u
	// - local_invocation_index: u32
	const module = device.createShaderModule({
		label: "Our compute module",
		code: /* wgsl */ `
        @group(0) @binding(0) var<storage, read_write> workgroupResult: array<vec3u>;
        @group(0) @binding(1) var<storage, read_write> globalResult: array<vec3u>;
        @group(0) @binding(2) var<storage, read_write> localIdResult: array<vec3u>;
        @group(0) @binding(3) var<storage, read_write> localIndexResult: array<u32>;
				
				@compute @workgroup_size(${workgroupSize}) fn computeSomething(
          @builtin(workgroup_id) workgroup_id: vec3u,
					@builtin(global_invocation_id) global_invocation_id: vec3u,
					@builtin(local_invocation_id) local_invocation_id: vec3u,
					@builtin(local_invocation_index) local_invocation_index: u32,
          @builtin(num_workgroups) num_workgroups: vec3u
				) {
          let workgroup_index = workgroup_id.x
            + workgroup_id.y * num_workgroups.x
            + workgroup_id.z * num_workgroups.x * num_workgroups.y;
          let global_invocation_index =
            workgroup_index * ${numThreadsPerWorkgroup}
            + local_invocation_index;

          workgroupResult[global_invocation_index] = workgroup_id;
          globalResult[global_invocation_index] = global_invocation_id;
          localIdResult[global_invocation_index] = local_invocation_id;
          localIndexResult[global_invocation_index] = local_invocation_index;
				}
      `,
	});

	const pipeline = device.createComputePipeline({
		label: "Indexing pipeline",
		layout: "auto",
		compute: { module },
	});

	// Create buffers
	const numThreads = reduceMul(dispatchSize) * numThreadsPerWorkgroup;
	const vecSize = numThreads * 4 * 4; // vec3u (4 bytes overhead) * u32 (4 bytes)

	// Work buffers have:
	const workUsage =
		GPUBufferUsage.STORAGE | // 1. storage address_space
		GPUBufferUsage.COPY_SRC; // 2. copy source for copying to read buffers
	const workgroupWork = device.createBuffer({ size: vecSize, usage: workUsage });
	const globalWork = device.createBuffer({ size: vecSize, usage: workUsage });
	const localIdWork = device.createBuffer({ size: vecSize, usage: workUsage });
	const localIndexWork = device.createBuffer({ size: numThreads * 4, usage: workUsage });

	// To read we:
	const readUsage =
		GPUBufferUsage.COPY_DST | // 1. copy from work buffer to read buffer
		GPUBufferUsage.MAP_READ; // 2. read from mapped array
	const workgroupRead = device.createBuffer({ size: vecSize, usage: readUsage });
	const globalRead = device.createBuffer({ size: vecSize, usage: readUsage });
	const localIdRead = device.createBuffer({ size: vecSize, usage: readUsage });
	const localIndexRead = device.createBuffer({ size: numThreads * 4, usage: readUsage });

	const bindGroup = device.createBindGroup({
		label: "Work buffers",
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: workgroupWork },
			{ binding: 1, resource: globalWork },
			{ binding: 2, resource: localIdWork },
			{ binding: 3, resource: localIndexWork },
		],
	});

	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(...dispatchSize);
	pass.end();
	encoder.copyBufferToBuffer(workgroupWork, 0, workgroupRead, 0, workgroupRead.size);
	encoder.copyBufferToBuffer(globalWork, 0, globalRead, 0, globalRead.size);
	encoder.copyBufferToBuffer(localIdWork, 0, localIdRead, 0, localIdRead.size);
	encoder.copyBufferToBuffer(localIndexWork, 0, localIndexRead, 0, localIndexRead.size);

	const commandBuffer = encoder.finish();
	device.queue.submit([commandBuffer]);

	// Read results
	await Promise.all([
		workgroupRead.mapAsync(GPUMapMode.READ),
		globalRead.mapAsync(GPUMapMode.READ),
		localIdRead.mapAsync(GPUMapMode.READ),
		localIndexRead.mapAsync(GPUMapMode.READ),
	]);

	const workgroup = new Uint32Array(workgroupRead.getMappedRange().slice());
	const global = new Uint32Array(globalRead.getMappedRange().slice());
	const localId = new Uint32Array(localIdRead.getMappedRange().slice());
	const localIndex = new Uint32Array(localIndexRead.getMappedRange().slice());

	workgroupRead.destroy();
	globalRead.destroy();
	localIdRead.destroy();
	localIndexRead.destroy();

	const get3 = (arr: Uint32Array, i: number) => {
		const off = i * 4;
		return `${arr[off]}, ${arr[off + 1]}, ${arr[off + 2]}`;
	};

	for (let i = 0; i < numThreads; ++i) {
		if (i % numThreadsPerWorkgroup === 0) {
			console.log(`\
 -----------------------------------------------
 global                 global    local   local   dispatch: ${i / numThreadsPerWorkgroup}
 invoc.    workgroup    invoc.    invoc.  invoc.
 index     id           id        id      index
 -----------------------------------------------`);
		}
		console.log(
			` ${i.toString().padStart(3)}:      ${get3(workgroup, i)}      ${get3(global, i)}   ${get3(localId, i)}   ${localIndex[i]}`,
		);
	}
}
