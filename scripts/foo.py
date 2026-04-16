import types
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

from scripts.reference import get_reference_f16


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

    import inspect

    for name, module in model.named_modules():
        if "layers.7" not in name:
            continue

        if name == "model.layers.7":
            print(inspect.getsource(module.__class__.forward))

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
        cache_implementation="static",
        max_length=(input_ids["input_ids"].shape[1] + 10),
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


def test_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")

    text = "Plants create energy through a process known as"
    print("Tokens: ", tokenizer(text))


def test_reference():
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def make_patched_forward(original_forward):
        def patched_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: torch.Tensor = None,
            attention_mask: torch.Tensor | None = None,
            past_key_values: Any | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

            cos, sin = position_embeddings
            print("cos")
            print(cos)
            print("sin")
            print(sin)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            print("query_states")
            print(query_states)
            print("key_states")
            print(key_states)

            return original_forward(
                # self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )

        return patched_forward

    tokenizer, model = get_reference_f16()

    layer = model.model.layers[0].self_attn
    layer.forward = types.MethodType(make_patched_forward(layer.forward), layer)

    for name, module in model.named_modules():
        if "layers.0" not in name:
            continue

        def hook(m, input, output, n=name):
            i = input[0] if len(input) > 0 else None
            o = output if isinstance(output, torch.Tensor) else output[0]
            print(n)
            print(i)
            print(o)

        module.register_forward_hook(hook)

    # text = "Plants create energy through a process known as"
    text = "The capital of France is "
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)

    print(input_ids["input_ids"])

    output = model.generate(
        **input_ids,
        cache_implementation="static",
        max_length=(input_ids["input_ids"].shape[1] + 1),
    )
    print(tokenizer.decode(output[0], skip_special_tokens=False))


def test_modules():
    tokenizer, model = get_reference_f16()

    import inspect

    print(
        inspect.getsource(
            model.get_submodule("model.layers.0.self_attn").__class__.forward
        )
    )
    # input_layernorm_0 = model.get_submodule("model.layers.0.input_layernorm")
    # print(input_layernorm_0.state_dict())


if __name__ == "__main__":
    test_reference()
