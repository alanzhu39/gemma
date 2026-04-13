import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open


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


if __name__ == "__main__":
    test_tokenizer()
