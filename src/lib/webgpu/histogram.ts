async function getImageDataFromUrl(url: string): Promise<ImageData> {
	return new Promise((resolve, reject) => {
		const img = new Image();

		// Enable CORS to prevent "tainted canvas" security errors
		img.crossOrigin = "Anonymous";

		img.onload = () => {
			// Create an offscreen canvas with matching dimensions
			const canvas = document.createElement("canvas");
			canvas.width = img.naturalWidth;
			canvas.height = img.naturalHeight;

			const ctx = canvas.getContext("2d");

			// Paint the image onto the canvas context
			ctx!.drawImage(img, 0, 0);

			// Extract the ImageData object
			const imageData = ctx!.getImageData(0, 0, canvas.width, canvas.height);
			resolve(imageData);
		};

		img.onerror = (err) => reject(err);
		img.src = url;
	});
}

// Returns a value from 0 to 1 for luminance.
// where r, g, b each go from 0 to 1.
function srgbLuminance(r: number, g: number, b: number): number {
	// from: https://www.w3.org/WAI/GL/wiki/Relative_luminance
	return r * 0.2126 + g * 0.7152 + b * 0.0722;
}

const numBins = 100;
const maxHeight = 30;

function printHistogram(bins: number[]) {
	const max = Math.max(...bins);
	let output = "";
	for (let i = maxHeight; i >= 0; i--) {
		for (let j = 0; j < numBins; j++) {
			if (bins[j] / max >= i / maxHeight) {
				output += "*";
			} else {
				output += " ";
			}
		}
		output += "\n";
	}
	console.log(output);
}

export async function runHistogramCpu() {
	console.log("CPU");

	const imgData = await getImageDataFromUrl("cat.jpg");
	const { width, height, data } = imgData;
	const bins = new Array(numBins).fill(0);

	const start = performance.now();
	for (let x = 0; x < width; x++) {
		for (let y = 0; y < height; y++) {
			const offset = (y * width + x) * 4;

			const r = data[offset + 0] / 255;
			const g = data[offset + 1] / 255;
			const b = data[offset + 2] / 255;
			const v = srgbLuminance(r, g, b);

			const bin = Math.min(numBins - 1, v * numBins) | 0;
			++bins[bin];
		}
	}
	console.log(`CPU took ${performance.now() - start} ms`);

	printHistogram(bins);
}

export async function runHistogram() {
	const imgData = await getImageDataFromUrl("cat.jpg");
	const { width, height, data } = imgData;

	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter?.requestDevice({
		requiredLimits: {
			maxStorageBufferBindingSize: 4 * width * height * 4,
		},
	});
	if (!device) {
		alert("WebGPU required but not available!");
		return;
	}

	const workgroupSize = [256, 1, 1];
	const dispatchSize = [Math.ceil(width / workgroupSize[0]), height] as const;
	const numChunks = dispatchSize[0] * dispatchSize[1];

	const luminanceModule = device.createShaderModule({
		label: "Per-pixel luminance",
		code: /* wgsl */ `
      const num_bins: u32 = ${numBins};
      const num_chunks: u32 = ${numChunks};
      const width: u32 = ${width};
      const height: u32 = ${height};

      var<workgroup> bins: array<atomic<u32>, num_bins>;
      @group(0) @binding(0) var<storage, read_write> chunks: array<array<u32, num_bins>, num_chunks>;
      @group(0) @binding(1) var<storage, read_write> img_data: array<u32>;

      const kSRGBLuminanceFactors = vec3f(0.2126, 0.7152, 0.0722);
      fn srgbLuminance(color: vec3f) -> f32 {
        return saturate(dot(color, kSRGBLuminanceFactors));
      }
      
      @compute @workgroup_size(${workgroupSize})
      fn luminance(
        @builtin(global_invocation_id) global_invocation_id: vec3u,
        @builtin(workgroup_id) workgroup_id: vec3u,
        @builtin(num_workgroups) num_workgroups: vec3u,
        @builtin(local_invocation_index) local_invocation_index: u32
      ) {
        let x = global_invocation_id.x;
        let y = global_invocation_id.y;
        if (x < width && y < height) {
          let offset = x + y * width;
          let packed: u32 = img_data[offset];
          const mask: u32 = 0xFFu;
          let r: f32 = f32(packed & mask) / 255f;
          let g: f32 = f32((packed >> 8u) & mask) / 255f;
          let b: f32 = f32((packed >> 16u) & mask) / 255f;
          let v: f32 = srgbLuminance(vec3f(r, g, b));
          let bin: u32 = min(num_bins - 1u, u32(v * f32(num_bins)));
          atomicAdd(&bins[bin], 1u);
        }

        workgroupBarrier();

        if (local_invocation_index < num_bins) {
          let chunk_index = workgroup_id.x + workgroup_id.y * num_workgroups.x;
          chunks[chunk_index][local_invocation_index] = atomicLoad(&bins[local_invocation_index]);
        }
      }
    `,
	});

	const luminancePipeline = device.createComputePipeline({
		label: "Luminance pipeline",
		layout: "auto",
		compute: { module: luminanceModule },
	});

	// Each workgroup processes all the bins, so workgroup size here should be num_bins...
	// We control the number of dispatches so that each workgroup sums two elements, striding correctly
	const sumModule = device.createShaderModule({
		label: "Sum bins across chunks",
		code: /* wgsl */ `
      const num_bins: u32 = ${numBins};
      const num_chunks: u32 = ${numChunks};

      @group(0) @binding(0) var<storage, read_write> chunks: array<array<u32, num_bins>, num_chunks>;
      @group(0) @binding(1) var<uniform> stride: u32;
      
      @compute @workgroup_size(num_bins, 1, 1)
      fn parallelSum(
        @builtin(workgroup_id) workgroup_id: vec3u,
        @builtin(local_invocation_index) local_invocation_index: u32
      ) {
        let chunk0 = workgroup_id.x * stride * 2;
        let chunk1 = chunk0 + stride;
        
        if (chunk1 < num_chunks) {
          chunks[chunk0][local_invocation_index] += chunks[chunk1][local_invocation_index];
        }
      }
    `,
	});

	const sumPipeline = device.createComputePipeline({
		label: "Sum pipeline",
		layout: "auto",
		compute: { module: sumModule },
	});

	// Work buffers
	const workUsage =
		GPUBufferUsage.STORAGE | // 1. storage address_space
		GPUBufferUsage.COPY_SRC; // 2. copy source for copying to read buffers

	const chunksWork = device.createBuffer({ size: 4 * numBins * numChunks, usage: workUsage });

	const imgDataWork = device.createBuffer({
		size: 4 * width * height * 4,
		usage: workUsage | GPUBufferUsage.COPY_DST,
	});
	device.queue.writeBuffer(imgDataWork, 0, data);

	// Read buffers
	const readUsage =
		GPUBufferUsage.COPY_DST | // 1. copy from work buffer to read buffer
		GPUBufferUsage.MAP_READ; // 2. read from mapped array
	const chunksRead = device.createBuffer({ size: chunksWork.size, usage: readUsage });

	// Bind groups
	const luminanceBindGroup = device.createBindGroup({
		label: "Work buffers",
		layout: luminancePipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: chunksWork },
			{ binding: 1, resource: imgDataWork },
		],
	});

	const sumBindGroups = [];
	const numSteps = Math.ceil(Math.log2(numChunks));
	for (let i = 0; i < numSteps; i++) {
		const strideBuffer = device.createBuffer({
			size: 4,
			usage: GPUBufferUsage.UNIFORM,
			mappedAtCreation: true,
		});
		new Uint32Array(strideBuffer.getMappedRange()).set([2 ** i]);
		strideBuffer.unmap();

		sumBindGroups.push(
			device.createBindGroup({
				label: `sum bind group ${i}`,
				layout: sumPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: chunksWork },
					{ binding: 1, resource: strideBuffer },
				],
			}),
		);
	}

	// Command encoder
	const encoder = device.createCommandEncoder();

	const pass = encoder.beginComputePass();
	pass.setPipeline(luminancePipeline);
	pass.setBindGroup(0, luminanceBindGroup);
	pass.dispatchWorkgroups(...dispatchSize);

	pass.setPipeline(sumPipeline);
	for (let i = 0; i < sumBindGroups.length; i++) {
		const stride = 2 ** i;
		pass.setBindGroup(0, sumBindGroups[i]);
		pass.dispatchWorkgroups(Math.ceil(numChunks / (stride * 2)));
	}
	pass.end();

	encoder.copyBufferToBuffer(chunksWork, 0, chunksRead, 0, chunksRead.size);

	const commandBuffer = encoder.finish();

	const start = performance.now();
	device.queue.submit([commandBuffer]);

	// Read results
	await chunksRead.mapAsync(GPUMapMode.READ);
	console.log(`WebGPU took ${performance.now() - start} ms`);

	const result = new Uint32Array(chunksRead.getMappedRange()).slice(0, numBins);

	chunksRead.destroy();

	// Log output
	console.log("WebGPU");
	printHistogram(Array.from(result));
}
