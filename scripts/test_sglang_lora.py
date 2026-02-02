#!/usr/bin/env python3
"""
Test script to verify SGLang LoRA loading and serving.
Downloads LoRA, starts SGLang container, sends test requests, and confirms LoRA is applied.
"""

import docker
import time
import requests
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from datetime import datetime

# Configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_MODEL_NAME = "gradients-io-tournaments/affine_goofspiel_random"
LORA_MODEL_REVISION = None
SGLANG_IMAGE = "lmsysorg/sglang:latest"
SGLANG_PORT = 30000
HF_CACHE_DIR = "/mnt/hf_cache"
LORA_ADAPTER_NAME = "trained_lora"

# Log file
LOG_FILE = f"sglang_lora_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message, print_to_console=True):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    if print_to_console:
        print(log_message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

def main():
    log("=" * 80)
    log("SGLang LoRA Test Script")
    log("=" * 80)
    log(f"Base Model: {BASE_MODEL_NAME}")
    log(f"LoRA Model: {LORA_MODEL_NAME}")
    log(f"SGLang Image: {SGLANG_IMAGE}")
    log(f"Log File: {LOG_FILE}")
    log("")

    client = docker.from_env()
    container = None

    try:
        # Step 1: Download LoRA
        log("Step 1: Downloading LoRA adapter...")
        safe_lora_name = LORA_MODEL_NAME.replace("/", "_")
        lora_dir = f"/tmp/sglang_lora_test/{safe_lora_name}"
        Path(lora_dir).parent.mkdir(parents=True, exist_ok=True)
        
        log(f"  Downloading {LORA_MODEL_NAME} to {lora_dir}...")
        snapshot_download(
            repo_id=LORA_MODEL_NAME,
            revision=LORA_MODEL_REVISION,
            local_dir=lora_dir,
            local_dir_use_symlinks=False,
        )
        log(f"  ✅ LoRA downloaded successfully")
        log("")

        # Step 2: Start SGLang container with LoRA
        log("Step 2: Starting SGLang container with LoRA...")
        sglang_command = (
            f"python3 -m sglang.launch_server --model-path {BASE_MODEL_NAME} "
            "--enable-lora --lora-paths trained_lora=/lora/trained_lora "
            "--lora-backend triton "
            f"--host 0.0.0.0 --port {SGLANG_PORT} --tensor-parallel-size 1 --dtype float16 "
            f"--random-seed 42"
        )
        
        log(f"  Command: {sglang_command}")
        log("  Starting container...")
        
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
        log(f"  ✅ Container started: {container.id[:12]}")
        log("")

        # Step 3: Stream logs and wait for readiness
        log("Step 3: Waiting for SGLang to be ready...")
        log("  Streaming container logs...")
        log("")
        
        def stream_logs():
            for line in container.logs(stream=True, follow=True):
                try:
                    log_line = line.decode('utf-8').strip()
                    if log_line:
                        log(f"  [CONTAINER] {log_line}", print_to_console=False)
                except:
                    pass
        
        import threading
        log_thread = threading.Thread(target=stream_logs, daemon=True)
        log_thread.start()
        
        # Wait for health check
        max_wait = 300  # 5 minutes
        wait_interval = 5
        elapsed = 0
        ready = False
        
        while elapsed < max_wait:
            try:
                response = requests.get(f"http://localhost:{SGLANG_PORT}/v1/models", timeout=2)
                if response.status_code == 200:
                    ready = True
                    log(f"  ✅ SGLang is ready! (took {elapsed}s)")
                    break
            except Exception as e:
                log(f"  Waiting... ({elapsed}s) - {str(e)}")
            
            time.sleep(wait_interval)
            elapsed += wait_interval
        
        if not ready:
            log("  ❌ SGLang failed to become ready within timeout")
            return
        
        log("")

        # Step 4: Check available models
        log("Step 4: Checking available models...")
        try:
            models_response = requests.get(f"http://localhost:{SGLANG_PORT}/v1/models", timeout=10)
            models_data = models_response.json()
            log(f"  Response: {json.dumps(models_data, indent=2)}")
            log("")
        except Exception as e:
            log(f"  ⚠️  Failed to get models: {e}")
            log("")

        # Step 5: Test prompts - comparing base vs LoRA
        test_prompts = [
            ("What is 2+2? Answer briefly.", 50),
            ("You are playing a game. What is your strategy?", 200),
            ("Explain how to play Goofspiel.", 200),
        ]
        
        base_model_name = BASE_MODEL_NAME
        lora_model_name = f"{BASE_MODEL_NAME}:{LORA_ADAPTER_NAME}"
        
        all_comparisons = []
        
        for prompt_idx, (test_prompt, max_tokens) in enumerate(test_prompts, start=5):
            log(f"Step {prompt_idx}: Testing prompt: '{test_prompt}'")
            log("")
            
            # Test base model
            log(f"  Testing BASE MODEL ({base_model_name})...")
            base_payload = {
                "model": base_model_name,
                "messages": [
                    {"role": "user", "content": test_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            
            base_content = None
            try:
                log("  Sending request...")
                base_response = requests.post(
                    f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
                    json=base_payload,
                    timeout=60
                )
                base_response.raise_for_status()
                base_result = base_response.json()
                base_content = base_result["choices"][0]["message"]["content"]
                log(f"  ✅ Base model response: {base_content}")
            except Exception as e:
                log(f"  ❌ Base model request failed: {e}")
            log("")
            
            # Test LoRA model
            log(f"  Testing LoRA MODEL ({lora_model_name})...")
            lora_payload = {
                "model": lora_model_name,
                "messages": [
                    {"role": "user", "content": test_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            
            lora_content = None
            try:
                log("  Sending request...")
                lora_response = requests.post(
                    f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
                    json=lora_payload,
                    timeout=60
                )
                lora_response.raise_for_status()
                lora_result = lora_response.json()
                lora_content = lora_result["choices"][0]["message"]["content"]
                log(f"  ✅ LoRA model response: {lora_content}")
            except Exception as e:
                log(f"  ❌ LoRA request failed: {e}")
            log("")
            
            # Compare responses
            log("  COMPARISON:")
            if base_content and lora_content:
                if base_content.strip() != lora_content.strip():
                    log("  ✅ Responses are DIFFERENT - LoRA is being applied!")
                    log(f"  Base:   {base_content}")
                    log(f"  LoRA:   {lora_content}")
                    all_comparisons.append({
                        "prompt": test_prompt,
                        "different": True,
                        "base": base_content,
                        "lora": lora_content
                    })
                else:
                    log("  ⚠️  Responses are IDENTICAL")
                    log(f"  Both:   {base_content}")
                    all_comparisons.append({
                        "prompt": test_prompt,
                        "different": False,
                        "base": base_content,
                        "lora": lora_content
                    })
            else:
                log("  ⚠️  Could not compare (one or both requests failed)")
                all_comparisons.append({
                    "prompt": test_prompt,
                    "different": None,
                    "base": base_content,
                    "lora": lora_content
                })
            log("")
            log("-" * 80)
            log("")
        
        # Step 8: Summary
        log("Step 8: Summary of all comparisons...")
        log("")
        different_count = sum(1 for c in all_comparisons if c.get("different") is True)
        identical_count = sum(1 for c in all_comparisons if c.get("different") is False)
        failed_count = sum(1 for c in all_comparisons if c.get("different") is None)
        
        log(f"  Total prompts tested: {len(all_comparisons)}")
        log(f"  Different responses: {different_count}")
        log(f"  Identical responses: {identical_count}")
        log(f"  Failed comparisons: {failed_count}")
        log("")
        
        if different_count > 0:
            log("  ✅ LoRA is confirmed to be working! Found differences in responses.")
        elif identical_count == len(all_comparisons):
            log("  ⚠️  All responses were identical. LoRA may not be affecting these prompts.")
        log("")

        log("=" * 80)
        log("Test completed!")
        log(f"All logs saved to: {LOG_FILE}")
        log("=" * 80)

    except Exception as e:
        log(f"❌ Fatal error: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Cleanup
        if container:
            log("")
            log("Cleaning up container...")
            try:
                container.stop(timeout=10)
                container.remove()
                log("  ✅ Container removed")
            except Exception as e:
                log(f"  ⚠️  Failed to remove container: {e}")

if __name__ == "__main__":
    main()
