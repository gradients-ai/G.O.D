#!/usr/bin/env python3
"""
Test sending request to SGLang with LoRA using the exact parameter format provided
"""

import docker
import time
import requests
import json
from huggingface_hub import snapshot_download
from pathlib import Path

# Configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_MODEL_NAME = "gradients-io-tournaments/affine_goofspiel_random"
SGLANG_IMAGE = "lmsysorg/sglang:latest"
SGLANG_PORT = 30000
HF_CACHE_DIR = "/mnt/hf_cache"
LORA_ADAPTER_NAME = "trained_lora"

def main():
    client = docker.from_env()
    container = None
    
    try:
        print("=" * 80)
        print("Starting SGLang with LoRA and testing request")
        print("=" * 80)
        
        # Download LoRA if needed
        print("\n1. Downloading LoRA adapter...")
        safe_lora_name = LORA_MODEL_NAME.replace("/", "_")
        lora_dir = f"/tmp/sglang_lora_test/{safe_lora_name}"
        Path(lora_dir).parent.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id=LORA_MODEL_NAME,
            revision=None,
            local_dir=lora_dir,
            local_dir_use_symlinks=False,
        )
        print(f"   ✅ LoRA downloaded to {lora_dir}")
        
        # Start SGLang container
        print("\n2. Starting SGLang container...")
        sglang_command = (
            f"python3 -m sglang.launch_server --model-path {BASE_MODEL_NAME} "
            "--enable-lora --lora-paths trained_lora=/lora/trained_lora "
            "--lora-backend triton "
            f"--host 0.0.0.0 --port {SGLANG_PORT} --tensor-parallel-size 1 --dtype float16 "
            f"--random-seed 42"
        )
        
        container = client.containers.run(
            SGLANG_IMAGE,
            command=sglang_command,
            name="sglang-lora-test-request",
            detach=True,
            ports={f"{SGLANG_PORT}/tcp": SGLANG_PORT},
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
            environment={
                "HF_HOME": "/hf",
                "TRANSFORMERS_CACHE": "/hf",
                "HUGGINGFACE_HUB_CACHE": "/hf",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PYTHONHASHSEED": "42",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "NVIDIA_TF32_OVERRIDE": "0",
            },
            volumes={
                HF_CACHE_DIR: {"bind": "/hf", "mode": "rw"},
                lora_dir: {"bind": "/lora/trained_lora", "mode": "ro"},
            },
            ipc_mode="host",
        )
        print(f"   ✅ Container started: {container.id[:12]}")
        
        # Wait for readiness
        print("\n3. Waiting for SGLang to be ready...")
        max_wait = 300
        elapsed = 0
        ready = False
        
        while elapsed < max_wait:
            try:
                response = requests.get(f"http://localhost:{SGLANG_PORT}/v1/models", timeout=2)
                if response.status_code == 200:
                    ready = True
                    print(f"   ✅ SGLang is ready! (took {elapsed}s)")
                    break
            except:
                pass
            time.sleep(5)
            elapsed += 5
            if elapsed % 15 == 0:
                print(f"   Waiting... ({elapsed}s)")
        
        if not ready:
            print("   ❌ SGLang failed to become ready")
            return
        
        # Send test request with exact format provided
        print("\n4. Sending test request with LoRA...")
        
        # Mock conversation
        mock_conversation = [
            {"role": "user", "content": "You are playing a game. What is your strategy?"}
        ]
        
        # Exact params format as provided, but with base-model:lora format
        params = {
            "model": f"{BASE_MODEL_NAME}:{LORA_ADAPTER_NAME}",  # Base model + LoRA
            "messages": mock_conversation,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Deterministic sampling (top-level for SGLang)
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
        }
        
        print(f"   Model: {params['model']}")
        print(f"   Prompt: {mock_conversation[0]['content']}")
        print(f"   Params: {json.dumps({k: v for k, v in params.items() if k != 'messages'}, indent=2)}")
        print("\n   Sending request...")
        
        # Send request
        response = requests.post(
            f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
            json=params,
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        print("   ✅ Request successful! Streaming response:\n")
        print("   " + "-" * 76)
        
        # Process stream
        full_content = ""
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta and delta['content'] is not None:
                                content = delta['content']
                                print(content, end='', flush=True)
                                full_content += content
                    except json.JSONDecodeError:
                        pass
        
        print("\n   " + "-" * 76)
        print(f"\n   ✅ Full response received ({len(full_content)} characters)")
        
        # Also test non-streaming for comparison
        print("\n5. Testing non-streaming request for comparison...")
        params_no_stream = params.copy()
        params_no_stream["stream"] = False
        del params_no_stream["stream_options"]
        
        response2 = requests.post(
            f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
            json=params_no_stream,
            timeout=60
        )
        response2.raise_for_status()
        result = response2.json()
        content2 = result["choices"][0]["message"]["content"]
        print(f"   ✅ Non-streaming response: {content2[:200]}...")
        
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if container:
            print("\nCleaning up container...")
            try:
                container.stop(timeout=10)
                container.remove()
                print("   ✅ Container removed")
            except Exception as e:
                print(f"   ⚠️  Failed to remove container: {e}")

if __name__ == "__main__":
    main()
