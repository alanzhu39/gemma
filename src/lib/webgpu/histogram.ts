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
	const imgData = await getImageDataFromUrl("cat.jpg");
	const { width, height, data } = imgData;
	const bins = new Array(numBins).fill(0);
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

	console.log("CPU");
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

	const workgroupSize = [1, 1, 1];

	// Naive method: one thread per pixel, sum to buckets with atomic add
	const module = device.createShaderModule({
		label: "Per-pixel luminance",
		code: /* wgsl */ `
      const num_bins: u32 = ${numBins};  

      @group(0) @binding(0) var<storage, read_write> bins: array<atomic<u32>>;
      @group(0) @binding(1) var<storage, read_write> img_data: array<u32>;
      @group(0) @binding(2) var<storage, read_write> debug: array<f32>;

      const kSRGBLuminanceFactors = vec3f(0.2126, 0.7152, 0.0722);
      fn srgbLuminance(color: vec3f) -> f32 {
        return saturate(dot(color, kSRGBLuminanceFactors));
      }
      
      @compute @workgroup_size(${workgroupSize})
      fn luminance(
        @builtin(global_invocation_id) global_invocation_id: vec3u,
        @builtin(num_workgroups) num_workgroups: vec3u
      ) {
        let offset = 4 * (global_invocation_id.x + global_invocation_id.y * num_workgroups.x);
        let r: f32 = f32(img_data[offset + 0]) / 255f;
        let g: f32 = f32(img_data[offset + 1]) / 255f;
        let b: f32 = f32(img_data[offset + 2]) / 255f;
        let v: f32 = srgbLuminance(vec3f(r, g, b));
        debug[offset] = f32(offset);
        let bin: u32 = min(num_bins - 1u, u32(v * f32(num_bins)));
        atomicAdd(&bins[bin], 1u);
      }
    `,
	});

	const pipeline = device.createComputePipeline({
		label: "Mandelbrot pipeline",
		layout: "auto",
		compute: { module },
	});

	// Create buffers
	const workUsage =
		GPUBufferUsage.STORAGE | // 1. storage address_space
		GPUBufferUsage.COPY_SRC; // 2. copy source for copying to read buffers
	const binsWork = device.createBuffer({ size: 4 * numBins, usage: workUsage });
	const imgDataWork = device.createBuffer({
		size: 4 * width * height * 4,
		usage: workUsage | GPUBufferUsage.COPY_DST,
	});
	const debugWork = device.createBuffer({
		size: 4 * width * height,
		usage: workUsage,
	});
	device.queue.writeBuffer(imgDataWork, 0, Uint32Array.from(data));

	const bindGroup = device.createBindGroup({
		label: "Work buffers",
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: binsWork },
			{ binding: 1, resource: imgDataWork },
			{ binding: 2, resource: debugWork },
		],
	});

	// To read we:
	const readUsage =
		GPUBufferUsage.COPY_DST | // 1. copy from work buffer to read buffer
		GPUBufferUsage.MAP_READ; // 2. read from mapped array
	const binsRead = device.createBuffer({ size: binsWork.size, usage: readUsage });
	const debugRead = device.createBuffer({ size: debugWork.size, usage: readUsage });

	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(width, height);
	pass.end();
	encoder.copyBufferToBuffer(binsWork, 0, binsRead, 0, binsRead.size);
	encoder.copyBufferToBuffer(debugWork, 0, debugRead, 0, debugRead.size);

	const commandBuffer = encoder.finish();
	device.queue.submit([commandBuffer]);

	// Read results
	await binsRead.mapAsync(GPUMapMode.READ);
	await debugRead.mapAsync(GPUMapMode.READ);

	const result = new Uint32Array(binsRead.getMappedRange().slice());
	const debug = new Float32Array(debugRead.getMappedRange().slice());
	console.log(debug);

	binsRead.destroy();
	debugRead.destroy();

	// Log output
	console.log("WebGPU");
	printHistogram(Array.from(result));
}
