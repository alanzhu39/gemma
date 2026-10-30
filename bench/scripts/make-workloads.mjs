#!/usr/bin/env node
/**
 * Generates bench/workloads.json: fixed, pre-tokenized prompts so the
 * benchmark harness never needs a tokenizer (or network) at run time.
 *
 * Run this only when changing workload definitions:
 *   node bench/scripts/make-workloads.mjs
 */
import { AutoTokenizer } from "@huggingface/transformers";
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const MODEL_ID = "alanzhu39/gemma-3-270m-it-f16";
const OUT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "workloads.json");

/**
 * Gemma 3 chat template. This repo publishes tokenizer_config.json without a
 * chat_template field, so we supply it explicitly rather than relying on the hub.
 */
const CHAT_TEMPLATE = `{{ bos_token }}
{%- for message in messages -%}
    {%- if message['role'] == 'assistant' -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\\n' + message['content'] | trim }}
    {{ '<end_of_turn>\\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\\n'}}
{%- endif -%}`;

// Realistic prose so prompts tokenize like real chat traffic rather than like
// degenerate repeated tokens.
const FILLER = `Machine learning systems that run in a browser have to cooperate with a graphics
driver, a scheduler, and a memory allocator that were designed for rendering
rather than for dense linear algebra. Every buffer that is allocated has to be
tracked, every dispatch has to be recorded into a command buffer, and every
readback forces a synchronisation point that drains work which may still be in
flight. When you profile such a system the first thing you notice is that the
cost of a single token is not the cost of the arithmetic; it is the cost of the
bookkeeping around it. Attention over a short context is cheap. The matrix
multiplications are small. What is expensive is launching hundreds of tiny
kernels and waiting for each of them in turn. A faster engine therefore comes
from doing fewer, larger pieces of work, and from keeping intermediate values
on the device instead of copying them back to the host to decide what to do
next.`;

const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
tokenizer.chat_template = CHAT_TEMPLATE;

/**
 * Build a chat-formatted prompt of exactly `n` tokens. The real template
 * suffix (end of turn + generation prompt) is preserved at the tail so the
 * model sees a well-formed conversation.
 */
function chatPrompt(n, seed) {
	const suffix = tokenizer.apply_chat_template(
		[{ role: "user", content: "PLACEHOLDER" }],
		{ tokenize: true, return_tensor: false, add_generation_prompt: true },
	).input_ids;
	// Drop the placeholder text tokens but keep bos/turn markers.
	const suffixTail = suffix.slice(-6);

	const bodyTokens = tokenizer.encode(
		`Request ${seed}. ${FILLER.repeat(Math.ceil(n / 40) + 2)}`,
	);
	const bodyLen = n - suffixTail.length;
	const ids = [...bodyTokens.slice(0, bodyLen), ...suffixTail];
	if (ids.length < n) throw new Error(`prompt too short: ${ids.length} < ${n}`);
	return ids.slice(0, n);
}

const workloads = {
	meta: {
		model: MODEL_ID,
		regenerate: "node bench/scripts/make-workloads.mjs",
		generatedAt: new Date().toISOString(),
	},
	prefill: [128, 512, 1024].map((n) => ({
		name: `prefill-${n}`,
		tokens: chatPrompt(n, n),
	})),
	// Decode workloads name the context they start from: the harness prefills
	// with `tokens`, then measures single-token steps.
	decode: [
		{ name: "decode-ctx128", contextLen: 128, steps: 32, tokens: chatPrompt(128, 11) },
		{ name: "decode-ctx1024", contextLen: 1024, steps: 32, tokens: chatPrompt(1024, 12) },
	],
	// B independent conversations: measures how the engine amortises work
	// across concurrent users (interleaved or batched steps).
	batch: [1, 2, 4].map((b) => ({
		name: `batch-${b}`,
		batchSize: b,
		tokens: Array.from({ length: b }, (_, i) => chatPrompt(128, 100 + i)),
	})),
};

await writeFile(OUT, JSON.stringify(workloads, null, "\t") + "\n");
console.log(`Wrote ${OUT}`);
for (const w of workloads.prefill) console.log(`  ${w.name}: ${w.tokens.length} tokens`);
for (const w of workloads.decode) console.log(`  ${w.name}: ${w.tokens.length} ctx, ${w.steps} steps`);
for (const w of workloads.batch) console.log(`  ${w.name}: [${w.tokens.map((t) => t.length).join(", ")}] tokens`);
