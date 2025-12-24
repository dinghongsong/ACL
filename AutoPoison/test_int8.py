#!/usr/bin/env python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")

# Test 1: Minimal INT8 config
print("\n=== Test 1: Minimal INT8 ===")
try:
    config1 = BitsAndBytesConfig(load_in_8bit=True)
    model1 = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        quantization_config=config1,
        device_map="auto",
    )
    print("✓ Minimal INT8 works!")
    del model1
except Exception as e:
    print(f"✗ Minimal INT8 failed: {e}")

# Test 2: INT8 with threshold
print("\n=== Test 2: INT8 with threshold ===")
try:
    config2 = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    model2 = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        quantization_config=config2,
        device_map="auto",
    )
    print("✓ INT8 with threshold works!")
    del model2
except Exception as e:
    print(f"✗ INT8 with threshold failed: {e}")

# Test 3: Full config (your current setup)
print("\n=== Test 3: Full INT8 config ===")
try:
    config3 = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    model3 = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        trust_remote_code=True,
        quantization_config=config3,
        device_map="auto",
    )
    print("✓ Full INT8 config works!")
    del model3
except Exception as e:
    print(f"✗ Full INT8 config failed: {e}")

print("\n=== Summary ===")
print("If all tests pass, the issue is model-specific.")
print("If tests fail, there's a bitsandbytes installation issue.")
