#!/usr/bin/env python3
"""Quick comparison test for base model vs LoRA on game-specific prompt"""

import requests
import json

SGLANG_PORT = 30000
PROMPT = "You are playing a game. What is your strategy?"

print("=" * 80)
print("Comparing Base Model vs LoRA Model Responses")
print("=" * 80)
print(f"Prompt: {PROMPT}\n")

# Test base model
print("1. Testing BASE MODEL (Qwen/Qwen2.5-3B-Instruct)...")
base_payload = {
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": PROMPT}],
    "temperature": 0.0,
    "max_tokens": 200,
}

try:
    base_response = requests.post(
        f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
        json=base_payload,
        timeout=60
    )
    base_response.raise_for_status()
    base_result = base_response.json()
    base_content = base_result["choices"][0]["message"]["content"]
    print(f"✅ Base Model Response:\n{base_content}\n")
except Exception as e:
    print(f"❌ Base model request failed: {e}\n")
    base_content = None

# Test LoRA model
print("2. Testing LoRA MODEL (Qwen/Qwen2.5-3B-Instruct:trained_lora)...")
lora_payload = {
    "model": "Qwen/Qwen2.5-3B-Instruct:trained_lora",
    "messages": [{"role": "user", "content": PROMPT}],
    "temperature": 0.0,
    "max_tokens": 200,
}

try:
    lora_response = requests.post(
        f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
        json=lora_payload,
        timeout=60
    )
    lora_response.raise_for_status()
    lora_result = lora_response.json()
    lora_content = lora_result["choices"][0]["message"]["content"]
    print(f"✅ LoRA Model Response:\n{lora_content}\n")
except Exception as e:
    print(f"❌ LoRA model request failed: {e}\n")
    lora_content = None

# Comparison
print("=" * 80)
print("COMPARISON:")
print("=" * 80)
if base_content and lora_content:
    if base_content.strip() != lora_content.strip():
        print("✅ RESPONSES ARE DIFFERENT - LoRA is being applied!")
        print("\n--- Base Model ---")
        print(base_content)
        print("\n--- LoRA Model ---")
        print(lora_content)
    else:
        print("⚠️  RESPONSES ARE IDENTICAL")
        print("\nBoth models produced:")
        print(base_content)
else:
    print("⚠️  Could not compare (one or both requests failed)")

print("\n" + "=" * 80)
