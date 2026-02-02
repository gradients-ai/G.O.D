#!/usr/bin/env python3
"""
Example: How to send requests to SGLang with LoRA using OpenAI-compatible API
"""

import requests
import json

# Configuration
SGLANG_URL = "http://localhost:30000/v1/chat/completions"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_ADAPTER_NAME = "trained_lora"

def send_request_with_lora_openai_api(conversation, stream=False):
    """
    Send request to SGLang with LoRA using OpenAI-compatible API.
    
    Args:
        conversation: List of message dicts, e.g.:
            [{"role": "user", "content": "Hello"}]
        stream: Whether to stream the response
    
    Returns:
        Response object
    """
    params = {
        # Use base-model:adapter-name format to invoke LoRA
        "model": f"{BASE_MODEL}:{LORA_ADAPTER_NAME}",
        "messages": conversation,
        "stream": stream,
        "stream_options": {"include_usage": True} if stream else None,
        # Deterministic sampling
        "temperature": 0.0,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens": 512,
    }
    
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    
    response = requests.post(
        SGLANG_URL,
        json=params,
        stream=stream,
        timeout=60
    )
    response.raise_for_status()
    return response

def send_request_without_lora(conversation, stream=False):
    """
    Send request to SGLang WITHOUT LoRA (base model only).
    """
    params = {
        "model": BASE_MODEL,  # Just base model name
        "messages": conversation,
        "stream": stream,
        "stream_options": {"include_usage": True} if stream else None,
        "temperature": 0.0,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens": 512,
    }
    
    params = {k: v for k, v in params.items() if v is not None}
    
    response = requests.post(
        SGLANG_URL,
        json=params,
        stream=stream,
        timeout=60
    )
    response.raise_for_status()
    return response

# Example usage
if __name__ == "__main__":
    conversation = [
        {"role": "user", "content": "You are playing a game. What is your strategy?"}
    ]
    
    print("=" * 80)
    print("Example: Request with LoRA")
    print("=" * 80)
    
    # With LoRA
    print("\n1. Request WITH LoRA:")
    print(f"   Model: {BASE_MODEL}:{LORA_ADAPTER_NAME}")
    try:
        response = send_request_with_lora_openai_api(conversation, stream=False)
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"   ✅ Response: {content[:200]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Without LoRA
    print("\n2. Request WITHOUT LoRA:")
    print(f"   Model: {BASE_MODEL}")
    try:
        response = send_request_without_lora(conversation, stream=False)
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"   ✅ Response: {content[:200]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("\nKey points:")
    print("1. Use model format: 'base-model:adapter-name' to invoke LoRA")
    print("2. Remove lora_path and extra_body parameters")
    print("3. All other parameters work as normal OpenAI API")
    print("=" * 80)
