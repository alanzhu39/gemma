import types
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reference implementation of Gemma3-270M, for comparing and verifying our implementation against

# text = "Plants create energy through a process known as"
# text = "The capital of France is"
text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean nulla arcu, mollis eget elit ut, sodales rhoncus magna. Proin eleifend dolor non magna rhoncus, non finibus enim lobortis. Phasellus fermentum, quam vel laoreet auctor, mi felis viverra arcu, in volutpat felis lorem eget lacus. Nullam congue hendrerit dignissim. Fusce ut elit consectetur, facilisis dui in, convallis turpis. Nam quam massa, blandit eget dolor a, dignissim tincidunt est. Cras ultricies iaculis arcu, quis semper velit volutpat laoreet.
Phasellus condimentum, tellus ac malesuada vehicula, risus ante ullamcorper dui, quis imperdiet dolor leo nec erat. Integer quis tellus eu diam sodales ornare consectetur nec augue. Nullam sagittis efficitur ligula. Nunc ac auctor eros. Etiam luctus placerat cursus. Praesent vitae quam imperdiet lectus scelerisque sollicitudin. Vestibulum id turpis nec lorem semper scelerisque vel sit amet erat. Maecenas maximus vel nulla eget mattis. Aenean vel tellus a odio laoreet luctus. Sed ut egestas ex. Pellentesque feugiat at augue in pellentesque.
In hac habitasse platea dictumst. Duis aliquet mollis consequat. Duis elit augue, aliquet in ultrices congue, placerat sit amet tortor. Quisque in odio ut tellus posuere suscipit. Cras laoreet rutrum consequat. Pellentesque venenatis mi dolor, a luctus augue sagittis sit amet. In rutrum dui risus, sed pulvinar purus aliquet in. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Nunc fermentum ex eu massa fringilla congue sit amet laoreet massa. Suspendisse leo magna, pellentesque fringilla dui at, dictum pulvinar massa. Nunc non elit feugiat, auctor sem vel, varius quam. Curabitur tincidunt, purus vel placerat accumsan, ligula ligula dignissim odio, eget ullamcorper nibh justo a justo. Ut eu pretium tellus.
Morbi at erat eget nisi sodales cursus. Integer ultricies efficitur efficitur. Maecenas eget commodo arcu, a rhoncus nibh. Nullam quis purus nec tellus feugiat porttitor ut ut nibh. Aliquam aliquam enim aliquet erat sagittis varius. Ut dictum tellus sit amet facilisis feugiat. Nulla eget rhoncus dui. Integer molestie orci nec neque rhoncus, ut gravida enim venenatis. Nulla vehicula, mauris in viverra elementum, mi nibh efficitur leo, vel accumsan diam libero id mauris. Nunc non ligula quis dolor vulputate dapibus. Nam tincidunt ipsum at libero euismod pharetra.
Fusce molestie lorem sit amet justo accumsan, sit amet tempor ligula semper. Etiam in dolor sit amet nunc egestas ullamcorper. Nullam sit amet malesuada turpis. Fusce aliquam tristique erat sed auctor. Aenean non velit turpis. Aliquam quis euismod mi. Integer lacus erat, vehicula ac bibendum sit amet, eleifend blandit odio. Praesent sagittis sodales semper. Aenean sit amet posuere enim. Aenean est tortor, accumsan nec ante nec, lacinia blandit purus."""


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

    output = model.generate(
        **input_ids,
        cache_implementation="static",
        max_length=(input_ids["input_ids"].shape[1] + 10),
    )
    print(tokenizer.decode(output[0], skip_special_tokens=False))


if __name__ == "__main__":
    # reference_bf16()
    reference_f16()
