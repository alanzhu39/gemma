from safetensors.torch import load_file, save_file
import torch


def convert_dtype(target_dtype, weights_file):
    tensors = load_file("weights/model.bf16.safetensors")
    target_tensors = {k: v.to(target_dtype) for k, v in tensors.items()}
    save_file(target_tensors, f"weights/{weights_file}")


if __name__ == "__main__":
    convert_dtype(torch.float16, "model.f16.safetensors")
    convert_dtype(torch.float32, "model.f32.safetensors")
