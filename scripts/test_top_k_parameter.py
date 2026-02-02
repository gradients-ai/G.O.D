#!/usr/bin/env python3
"""
Test how to pass top_k and other SGLang-specific parameters through OpenAI client
"""

import requests
import json

SGLANG_URL = "http://localhost:30000/v1/chat/completions"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_ADAPTER = "trained_lora"

def test_with_extra_body():
    """Test using extra_body for SGLang-specific parameters"""
    print("=" * 80)
    print("Test 1: Using extra_body for top_k")
    print("=" * 80)
    
    params = {
        "model": f"{BASE_MODEL}:{LORA_ADAPTER}",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        # SGLang-specific parameters via extra_body
        "extra_body": {
            "top_k": -1,
        }
    }
    
    print(f"Params: {json.dumps(params, indent=2)}")
    
    try:
        response = requests.post(SGLANG_URL, json=params, timeout=60)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Success! Response: {result['choices'][0]['message']['content']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_direct_top_k():
    """Test if top_k can be passed directly (will likely fail)"""
    print("\n" + "=" * 80)
    print("Test 2: Trying top_k directly (will likely fail)")
    print("=" * 80)
    
    params = {
        "model": f"{BASE_MODEL}:{LORA_ADAPTER}",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,  # Direct parameter
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    
    print(f"Params: {json.dumps(params, indent=2)}")
    
    try:
        response = requests.post(SGLANG_URL, json=params, timeout=60)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Success! Response: {result['choices'][0]['message']['content']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_all_sglang_params():
    """Test all SGLang-specific parameters"""
    print("\n" + "=" * 80)
    print("Test 3: All SGLang-specific parameters via extra_body")
    print("=" * 80)
    
    params = {
        "model": f"{BASE_MODEL}:{LORA_ADAPTER}",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        # All SGLang-specific parameters
        "extra_body": {
            "top_k": -1,
            "use_beam_search": False,
        }
    }
    
    print(f"Params: {json.dumps(params, indent=2)}")
    
    try:
        response = requests.post(SGLANG_URL, json=params, timeout=60)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Success! Response: {result['choices'][0]['message']['content']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    # Check if SGLang is running
    try:
        response = requests.get("http://localhost:30000/v1/models", timeout=2)
        if response.status_code != 200:
            print("❌ SGLang is not running. Please start it first.")
            exit(1)
    except:
        print("❌ SGLang is not running. Please start it first.")
        exit(1)
    
    test_with_extra_body()
    test_direct_top_k()
    test_all_sglang_params()
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("For OpenAI Python client, use extra_body parameter:")
    print("  client.chat.completions.create(")
    print("      model='base-model:lora',")
    print("      messages=[...],")
    print("      temperature=0.0,")
    print("      extra_body={'top_k': -1}  # SGLang-specific params")
    print("  )")
    print("=" * 80)
