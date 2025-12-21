"""
Check memory usage for quantized vs standard model loading
"""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import gc

def get_model_memory_gb(model):
    """Calculate model parameter memory in GB"""
    mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return mem_bytes / (1024 ** 3)

def get_gpu_memory_gb():
    """Get GPU allocated memory in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0

def check_device(model):
    """Check where model parameters are located"""
    devices = set()
    for p in model.parameters():
        devices.add(str(p.device))
    return devices

# Test FP32 (Full Precision)
print("=" * 60)
print("Testing Standard Model Loading (FP32)")
print("=" * 60)

torch.cuda.empty_cache()
gc.collect()

model_fp32 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float32,
    trust_remote_code=True
)

model_mem_fp32 = get_model_memory_gb(model_fp32)
print(f"Model Parameters Memory: {model_mem_fp32:.2f} GB")
print(f"Devices: {check_device(model_fp32)}")
print(f"Expected ~26GB (FP32): {'✓' if 24 <= model_mem_fp32 <= 28 else '✗'}")

del model_fp32
torch.cuda.empty_cache()
gc.collect()

# Test FP16 (Half Precision)
print("\n" + "=" * 60)
print("Testing Standard Model Loading (FP16)")
print("=" * 60)

model_fp16 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model_mem_fp16 = get_model_memory_gb(model_fp16)
print(f"Model Parameters Memory: {model_mem_fp16:.2f} GB")
print(f"Devices: {check_device(model_fp16)}")
print(f"Expected ~13GB (FP16): {'✓' if 12 <= model_mem_fp16 <= 14 else '✗'}")

del model_fp16
torch.cuda.empty_cache()
gc.collect()

print("\n" + "=" * 60)
print("Testing Quantized Model Loading (NF4)")
print("=" * 60)

mem_before = get_gpu_memory_gb()

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_quantized = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=nf4_config,
    device_map="auto",
    trust_remote_code=True
)

mem_after = get_gpu_memory_gb()

model_mem_quant = get_model_memory_gb(model_quantized)
print(f"Model Parameters Memory: {model_mem_quant:.2f} GB")
print(f"Devices: {check_device(model_quantized)}")
print(f"GPU Memory Allocated: {mem_after:.2f} GB")
print(f"Expected ~3.3GB (NF4): {'✓' if 3 <= model_mem_quant <= 4 else '✗'}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"FP32 (Full):       {model_mem_fp32:.2f} GB")
print(f"FP16 (Half):       {model_mem_fp16:.2f} GB")
print(f"NF4 (Quantized):   {model_mem_quant:.2f} GB")
print(f"\nMemory Reduction:")
print(f"  FP32 -> NF4:     {model_mem_fp32 / model_mem_quant:.2f}x")
print(f"  FP16 -> NF4:     {model_mem_fp16 / model_mem_quant:.2f}x")
print(f"\nTheoretical values for Llama-2-7B:")
print(f"  FP32: ~26.0 GB (7B params × 4 bytes)")
print(f"  FP16: ~13.0 GB (7B params × 2 bytes)")
print(f"  NF4:  ~3.3 GB (7B params × 0.5 bytes)")
print(f"  FP32->NF4: ~8x, FP16->NF4: ~4x")
