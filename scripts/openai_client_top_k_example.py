#!/usr/bin/env python3
"""
Example: How to pass top_k and other SGLang-specific parameters 
through OpenAI Python client using extra_body
"""

from openai import AsyncOpenAI
import asyncio

# Initialize client pointing to SGLang server
client = AsyncOpenAI(
    base_url="http://localhost:30000/v1",
    api_key="not-needed"  # SGLang doesn't require API key
)

async def example_with_top_k():
    """Example showing correct way to pass top_k"""
    
    # CORRECT: Use extra_body for SGLang-specific parameters
    params = {
        "model": "Qwen/Qwen2.5-3B-Instruct:trained_lora",
        "messages": [
            {"role": "user", "content": "You are playing a game. What is your strategy?"}
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        # Standard OpenAI parameters (accepted directly)
        "temperature": 0.0,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        # SGLang-specific parameters via extra_body
        "extra_body": {
            "top_k": -1,
            # Other SGLang-specific params can go here too
            # "use_beam_search": False,
        }
    }
    
    try:
        stream = await client.chat.completions.create(**params)
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)
        print()  # New line after stream
        
    except Exception as e:
        print(f"Error: {e}")

# For non-streaming requests
async def example_non_streaming():
    """Non-streaming example"""
    
    params = {
        "model": "Qwen/Qwen2.5-3B-Instruct:trained_lora",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "extra_body": {
            "top_k": -1,
        }
    }
    
    try:
        response = await client.chat.completions.create(**params)
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("OpenAI Client with top_k via extra_body")
    print("=" * 80)
    print("\nKey points:")
    print("1. Standard OpenAI params go directly: temperature, top_p, etc.")
    print("2. SGLang-specific params go in extra_body: top_k, use_beam_search, etc.")
    print("3. Model format: 'base-model:adapter-name' for LoRA")
    print("=" * 80)
