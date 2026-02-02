#!/usr/bin/env python3
"""
Test script to verify LoRA loading and serving in SGLang.
Compares outputs with and without LoRA to confirm LoRA is being applied.
"""
import docker
import time
import requests
import json
from huggingface_hub import snapshot_download

# Configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_MODEL_NAME = "gradients-io-tournaments/affine_goofspiel_random"
LORA_MODEL_REVISION = None
SGLANG_IMAGE = "lmsysorg/sglang:latest"
SGLANG_PORT = 30000
HF_CACHE_DIR = "/mnt/hf_cache"
RANDOM_SEED = 42

client = docker.from_env()

def test_lora_loading():
    """Test LoRA loading and verify it's being applied."""
    container = None
    
    try:
        print("=" * 60)
        print("Testing SGLang LoRA Loading and Serving")
        print("=" * 60)
        
        # 1. Download LoRA locally
        print(f"\n📥 Downloading LoRA: {LORA_MODEL_NAME}")
        safe_lora_name = LORA_MODEL_NAME.replace("/", "_")
        lora_dir = f"/tmp/sglang_lora_test/{safe_lora_name}"
        snapshot_download(
            repo_id=LORA_MODEL_NAME,
            revision=LORA_MODEL_REVISION,
            local_dir=lora_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✅ LoRA downloaded to: {lora_dir}")
        
        # 2. Start SGLang container with LoRA
        print(f"\n🚀 Starting SGLang container with LoRA...")
        sglang_command = (
            f"python3 -m sglang.launch_server --model-path {BASE_MODEL_NAME} "
            "--enable-lora --lora-paths trained_lora=/lora/trained_lora "
            "--lora-backend triton "
            "--host 0.0.0.0 --port 30000 --tensor-parallel-size 1 --dtype float16 "
            f"--random-seed {RANDOM_SEED}"
        )
        
        container = client.containers.run(
            SGLANG_IMAGE,
            command=sglang_command,
            name="sglang-lora-test",
            detach=True,
            ports={f"{SGLANG_PORT}/tcp": SGLANG_PORT},
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
            environment={
                "HF_HOME": "/hf",
                "TRANSFORMERS_CACHE": "/hf",
                "HUGGINGFACE_HUB_CACHE": "/hf",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            },
            volumes={
                HF_CACHE_DIR: {"bind": "/hf", "mode": "rw"},
                lora_dir: {"bind": "/lora/trained_lora", "mode": "ro"},
            },
            ipc_mode="host",
        )
        
        # 3. Wait for server to be ready
        print("⏳ Waiting for SGLang to be ready...")
        max_wait = 300  # 5 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                resp = requests.get(f"http://localhost:{SGLANG_PORT}/v1/models", timeout=5)
                if resp.status_code == 200:
                    models = resp.json()
                    print(f"✅ SGLang is ready!")
                    print(f"   Available models: {json.dumps(models, indent=2)}")
                    break
            except Exception as e:
                time.sleep(5)
        else:
            print("❌ Timeout waiting for SGLang to start")
            print("Container logs:")
            print(container.logs().decode('utf-8', errors='ignore')[-2000:])
            return False
        
        # 4. Check available models/adapters
        print(f"\n📋 Checking available models and adapters...")
        try:
            models_resp = requests.get(f"http://localhost:{SGLANG_PORT}/v1/models", timeout=10)
            models_data = models_resp.json()
            print(f"Models endpoint response: {json.dumps(models_data, indent=2)}")
        except Exception as e:
            print(f"⚠️  Error checking models: {e}")
        
        # 5. Send test request WITH LoRA (using OpenAI-compatible API)
        print(f"\n🧪 Sending test request WITH LoRA adapter...")
        test_prompt = "What is 2+2? Answer briefly."
        
        # Format: base-model:adapter-name per SGLang docs
        model_with_lora = f"{BASE_MODEL_NAME}:trained_lora"
        
        payload_with_lora = {
            "model": model_with_lora,
            "messages": [
                {"role": "user", "content": test_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 50,
        }
        
        print(f"   Model: {model_with_lora}")
        print(f"   Prompt: {test_prompt}")
        
        try:
            response_with_lora = requests.post(
                f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
                json=payload_with_lora,
                timeout=60
            )
            response_with_lora.raise_for_status()
            result_with_lora = response_with_lora.json()
            output_with_lora = result_with_lora['choices'][0]['message']['content']
            print(f"✅ Response WITH LoRA:")
            print(f"   {output_with_lora}")
        except Exception as e:
            print(f"❌ Error with LoRA request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text}")
            return False
        
        # 6. Send test request WITHOUT LoRA (base model only)
        print(f"\n🧪 Sending test request WITHOUT LoRA (base model)...")
        payload_without_lora = {
            "model": BASE_MODEL_NAME,
            "messages": [
                {"role": "user", "content": test_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 50,
        }
        
        try:
            response_without_lora = requests.post(
                f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
                json=payload_without_lora,
                timeout=60
            )
            response_without_lora.raise_for_status()
            result_without_lora = response_without_lora.json()
            output_without_lora = result_without_lora['choices'][0]['message']['content']
            print(f"✅ Response WITHOUT LoRA:")
            print(f"   {output_without_lora}")
        except Exception as e:
            print(f"⚠️  Error without LoRA request: {e}")
            output_without_lora = None
        
        # 7. Compare outputs
        print(f"\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"Prompt: {test_prompt}\n")
        print(f"WITH LoRA ({model_with_lora}):")
        print(f"  {output_with_lora}\n")
        if output_without_lora:
            print(f"WITHOUT LoRA ({BASE_MODEL_NAME}):")
            print(f"  {output_without_lora}\n")
        
        if output_without_lora:
            if output_with_lora.strip() != output_without_lora.strip():
                print("✅ SUCCESS: LoRA is being applied! Outputs differ.")
                return True
            else:
                print("⚠️  WARNING: Outputs are identical. LoRA may not be applied or has no effect.")
                return False
        else:
            print("✅ LoRA request succeeded. (Could not compare with base model)")
            return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if container:
            print(f"\n🧹 Cleaning up container...")
            try:
                container.remove(force=True)
            except:
                pass

if __name__ == "__main__":
    success = test_lora_loading()
    exit(0 if success else 1)
