import types
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reference implementation of Gemma3-270M, for comparing and verifying our implementation against

text = "Plants create energy through a process known as"
# text = "The capital of France is"


def reference_bf16():
    tokenizer_path = "./tokenizer/"
    model_path = "./model_bf16/"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="sdpa",
    )
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)

    output = model.generate(**input_ids, cache_implementation="static", max_length=39)
    print(tokenizer.decode(output[0], skip_special_tokens=False))


def get_reference_f16():
    tokenizer_path = "./tokenizer/"
    model_path = "./model_f16/"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        # attn_implementation="sdpa",
    )

    # Patch forward pass with clamping on the residual additions, which
    # overflow float16 limits otherwise. Output is comparable to bf16 reference.
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = (residual + hidden_states).clamp(min=-65504, max=65504)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = (residual + hidden_states).clamp(min=-65504, max=65504)

        return hidden_states

    for layer in model.model.layers:
        layer.forward = types.MethodType(patched_forward, layer)

    return tokenizer, model


def reference_f16():
    tokenizer, model = get_reference_f16()

    input_ids = tokenizer(text, return_tensors="pt").to(model.device)

    output = model.generate(**input_ids, cache_implementation="static", max_length=39)
    print(tokenizer.decode(output[0], skip_special_tokens=False))


if __name__ == "__main__":
    # reference_bf16()
    reference_f16()
