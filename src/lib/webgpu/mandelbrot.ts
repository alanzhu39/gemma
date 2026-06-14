export async function runMandelbrot() {
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter?.requestDevice();
	if (!device) {
		alert("WebGPU required but not available!");
		return;
	}

	const [minX, maxX] = [-2, 0.5];
	const [minY, maxY] = [-1.25, 1.25];
	const [W, H] = [100, 50];
	const numIters = 256;
	const workgroupSize = [16, 16, 1];

	const module = device.createShaderModule({
		label: "Mandelbrot compute module",
		code: /* wgsl */ `
        const min_x: f32 = ${minX};
        const max_x: f32 = ${maxX};
        const min_y: f32 = ${minY};
        const max_y: f32 = ${maxY};
        const w: u32 = ${W};
        const h: u32 = ${H};

        @group(0) @binding(0) var<storage, read_write> result: array<u32>;
        @group(0) @binding(1) var<uniform> num_iters: u32;
				
				@compute @workgroup_size(${workgroupSize}) fn computeSomething(
					@builtin(global_invocation_id) global_invocation_id: vec3u,
          @builtin(num_workgroups) num_workgroups: vec3u
				) {
          let cr: f32 = min_x + (max_x - min_x) * (f32(global_invocation_id.x) / f32(w));
          let ci: f32 = min_y + (max_y - min_y) * (f32(global_invocation_id.y) / f32(h));

          var zr: f32 = 0;
          var zi: f32 = 0;
          var i: u32 = 0;
          while (i < num_iters && (zr * zr + zi * zi) < 4) {
            let zr2: f32 = zr * zr;
            let zi2: f32 = zi * zi;

            zi = 2 * zr * zi + ci;
            zr = zr2 - zi2 + cr;

            i++;
          }

          result[global_invocation_id.x + global_invocation_id.y * w] = i;
				}
      `,
	});

	const pipeline = device.createComputePipeline({
		label: "Mandelbrot pipeline",
		layout: "auto",
		compute: { module },
	});

	// Create buffers
	const size = W * H * 4; // One u32 per pixel

	// Work buffers have:
	const workUsage =
		GPUBufferUsage.STORAGE | // 1. storage address_space
		GPUBufferUsage.COPY_SRC; // 2. copy source for copying to read buffers
	const resultWork = device.createBuffer({ size, usage: workUsage });
	const numItersUniform = device.createBuffer({
		size: 4,
		usage: GPUBufferUsage.UNIFORM,
		mappedAtCreation: true,
	});
	new Uint32Array(numItersUniform.getMappedRange()).set([numIters]);
	numItersUniform.unmap();

	const bindGroup = device.createBindGroup({
		label: "Work buffers",
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: resultWork },
			{ binding: 1, resource: numItersUniform },
		],
	});

	// To read we:
	const readUsage =
		GPUBufferUsage.COPY_DST | // 1. copy from work buffer to read buffer
		GPUBufferUsage.MAP_READ; // 2. read from mapped array
	const resultRead = device.createBuffer({ size, usage: readUsage });

	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(Math.ceil(W / workgroupSize[0]), Math.ceil(H / workgroupSize[1]));
	pass.end();
	encoder.copyBufferToBuffer(resultWork, 0, resultRead, 0, resultRead.size);

	const commandBuffer = encoder.finish();
	device.queue.submit([commandBuffer]);

	// Read results
	await resultRead.mapAsync(GPUMapMode.READ);

	const result = new Uint32Array(resultRead.getMappedRange().slice());

	resultRead.destroy();

	// Log output
	const chars = " .:-=+*#%@";
	let output = "";
	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			const iter = result[y * W + x];
			const idx = Math.floor((iter / numIters) * (chars.length - 1));
			output += chars[idx];
		}
		output += "\n";
	}
	console.log(output);
}
