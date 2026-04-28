# Gemma3 Inference

Gemma 3 270M-Instruct running entirely in your browser, on WebGPU.

**[Live demo →](https://gemma.alanzhu39.com)**

[images/screenshot.png]

## Overview

This project runs full FP16 inference for [Gemma 3 270M-Instruct](https://huggingface.co/google/gemma-3-270m-it) directly in the browser via WebGPU. The original Google weights are in BF16, which WebGPU doesn't support, so I [converted](https://github.com/alanzhu39/gemma/blob/main/scripts/convert_safetensors_dtype.py) the safetensors file to FP16 and host it in a separate HuggingFace [repo](https://huggingface.co/alanzhu39/gemma-3-270m-it-f16). For more info on the Gemma3 architecture, see [this blog post](https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/).

## Features

- **Attention visualization** — run a forward pass and inspect per-layer, per-head attention weights as a heatmap
- **Chat interface** — full instruct-format chat with streaming token generation

## Performance optimizations

- **KV cache** — past key/value pairs are cached so each generation step only processes the new token, not the full context
- **Ring buffer for sliding attention** — Gemma 3's local attention layers use a sliding window; the KV cache for these layers is managed with a ring buffer to avoid unneeded caching.

### JIT kernel fusion with jax-js

This project is built using [jax-js](https://jax-js.com/), a machine learning library and compiler for the web. It brings a JAX-like array programming API to the browser, compiling to WebGPU with built-in kernel fusion. It's super awesome, go check it out!

## Running locally

```sh
npm install
npm run dev
```

Requires a browser with WebGPU support (Chrome 113+ or Edge 113+). Safari support is partial.

## TODO

- [ ] Performance benchmarking: tok/s with vs. without JIT, memory usage with vs. without KV cache
- [ ] More performance optimizations
- [ ] More models (Gemma 4 E2B?)