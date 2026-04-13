import { safetensors } from "@jax-js/loaders";
import { readFileSync } from "node:fs";
import { fromSafetensors, type Gemma3 } from "./model";

// Run inference for model

// Load weights
const buf = readFileSync("weights/model.f16.safetensors");
const file = safetensors.parse(buf);
const weights: Gemma3 = fromSafetensors(file);

// Tokenize input

// Embed tokens
// Slice weights.tokenEmbed

// Decoder layers foward

// Final RMS norm

// Project back to token space
