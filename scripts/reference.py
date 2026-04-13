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

    # output = model.generate(**input_ids, cache_implementation="static", max_length=39)
    output = model.generate(
        **input_ids,
        cache_implementation="static",
        max_length=(input_ids["input_ids"].shape[1] + 1),
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


def reference_f16():
    tokenizer_path = "./tokenizer/"
    model_path = "./model_f16/"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map="auto",
        # attn_implementation="sdpa",
    )
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)

    output = model.generate(**input_ids, cache_implementation="static", max_length=39)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    # reference_bf16()
    reference_f16()
