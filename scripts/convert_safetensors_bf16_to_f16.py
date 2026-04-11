from safetensors.torch import load_file, save_file
import torch

tensors = load_file("weights/model.bf16.safetensors")
tensors_f16 = {k: v.to(torch.float16) for k, v in tensors.items()}
save_file(tensors_f16, "weights/model.f16.safetensors")
