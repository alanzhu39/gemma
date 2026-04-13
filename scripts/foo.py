import types
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open


def test_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")

    text = "Hello, world!"
    print("Tokens: ", tokenizer(text))


def test_weights():
    with (
        safe_open("weights/model.bf16.safetensors", framework="pt") as b,
        safe_open("weights/model.f16.safetensors", framework="pt") as f,
    ):
        # Get a specific tensor
        for key in b.keys():
            b_tensor = b.get_tensor(key).to(torch.float32)
            _f_tensor = f.get_tensor(key).to(torch.float32)

            mask = torch.abs(b_tensor) < 1e-5
            indices = torch.where(mask)
            if len(indices[0]) > 0:
                print(f"{key}: FAILED")
                print(indices)
                print(b_tensor[mask])

            # if not torch.allclose(b_tensor, f_tensor):
            #     print(f"{key}: FAILED")
            #     mask = ~torch.isclose(b_tensor, f_tensor)
            #     indices = torch.where(mask)

            #     print(indices)  # Outputs the index/indices
            #     print(b_tensor[mask], f_tensor[mask])  # Shows the violating values


def test_dtypes():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")

    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    for device in devices:
        for dtype in dtypes:
            try:
                x = torch.ones(4, 4, dtype=dtype, device=device)
                _y = x @ x  # matmul is a good stress test
                print(f"{device} + {dtype}: OK")
            except Exception as e:
                print(f"{device} + {dtype}: FAILED — {e}")


def test_overunderflow():
    tokenizer_path = "./tokenizer/"
    model_path = "./model_f16/"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # dtype=torch.float32,
        device_map="auto",
        # attn_implementation="eager",
    )

    # After conversion, check the weight distribution
    for name, param in model.named_parameters():
        if param.isinf().any():
            print(f"{name}: has inf (overflow during cast)")
        if param.isnan().any():
            print(f"{name}: has nan")
        # print(f"{name}: max={param.abs().max():.3f}")

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
        hidden_states = (residual.float() + hidden_states.float()).half()

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = (residual.float() + hidden_states.float()).half()

        return hidden_states

    for layer in model.model.layers:
        layer.forward = types.MethodType(patched_forward, layer)
        # layer.__class__.forward = patched_forward

    # import inspect

    for name, module in model.named_modules():
        if "layers.7" not in name:
            continue

        # if name == "model.layers.7":
        #     print(inspect.getsource(module.__class__.forward))

        def hook(m, input, output, n=name):
            t = output if isinstance(output, torch.Tensor) else output[0]
            if len(input) > 0 and isinstance(input[0], torch.Tensor):
                print(input[0].abs().max())
            print(
                f"{n}: max={t.abs().max():.1f}, has_inf={t.isinf().any()}, has_nan={t.isnan().any()}"
            )

        module.register_forward_hook(hook)

        # def hook(m, input, output, n=name):
        #     # if "layers.7" not in n:
        #     #     return
        #     t = output if isinstance(output, torch.Tensor) else output[0]
        #     if torch.isnan(t).any():
        #         print(f"{n}: has nan")
        #     if torch.isinf(t).any():
        #         print(f"{n}: has inf")
        #     if t.abs().max() < 1e-6:
        #         print(f"{n}: suspiciously small max={t.abs().max()}")

        # module.register_forward_hook(hook)

    text = "Plants create energy through a process known as"
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)

    output = model.generate(
        **input_ids,
        # cache_implementation="static",
        max_length=(input_ids["input_ids"].shape[1] + 1),
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    test_overunderflow()
