import docker
import time
import requests
import random
from datetime import datetime

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_MODEL_NAME = None # Place the name of your HuggingFace repo with the trained LORA here.
VLLM_IMAGE = "vllm/vllm-openai:latest"
AGENTGYM_IMAGE = "affinefoundation/agentgym:alfworld"
NETWORK_NAME = "agent_eval_net"

# Evaluation Params
NUM_EVALS = 500
DATA_LEN_RANGE = 2500
TEMPERATURE = 0.0
RANDOM_SEED = 42


client = docker.from_env()

def run_random_eval_suite():
    containers = {}
    all_results = []

    try:
        # 1. Infrastructure Setup
        networks = client.networks.list(names=[NETWORK_NAME])
        if not networks: client.networks.create(NETWORK_NAME, driver="bridge")

        if LORA_MODEL_NAME:
            print(f"🚀 Starting vLLM: {BASE_MODEL_NAME} w/ lora {LORA_MODEL_NAME}")
            vllm_command = f"--model {BASE_MODEL_NAME} --enable-lora --lora-modules trained_lora={LORA_MODEL_NAME} --port 8000 --trust-remote-code"

        else:
            print(f"🚀 Starting vLLM: {BASE_MODEL_NAME}")
            vllm_command = f"--model {BASE_MODEL_NAME} --port 8000 --trust-remote-code"

        vllm = client.containers.run(
            VLLM_IMAGE,
            command=vllm_command,
            name="vllm-server",
            detach=True,
            network=NETWORK_NAME,
            ports={'8000/tcp': 8000},
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
        )
        containers['vllm'] = vllm

        print("🚀 Starting AgentGym Server...")
        agent = client.containers.run(
            AGENTGYM_IMAGE,
            name="agentgym-server",
            detach=True,
            network=NETWORK_NAME,
            ports={'8000/tcp': 8001} 
        )
        containers['agent'] = agent

        # 2. Wait for Readiness
        print("⏳ Waiting for vLLM health check...")
        while True:
            try:
                if requests.get("http://localhost:8000/v1/models", timeout=2).status_code == 200:
                    break
            except:
                time.sleep(5)
        print("✅ vLLM Ready.\n")

        # 3. Evaluation Loop
        random.seed(RANDOM_SEED)
        eval_list = random.sample(range(1, DATA_LEN_RANGE + 1), NUM_EVALS)
        total_score = 0.0
        total_time = 0.0

        if LORA_MODEL_NAME:
            inference_model_name = "trained_lora"
        else:
            inference_model_name = BASE_MODEL_NAME

        for i, task_id in enumerate(eval_list):
            print(f"🔄 [{i+1}/{NUM_EVALS}] Task ID: {task_id}...", end="", flush=True)

            payload = {
                "model": inference_model_name,
                "base_url": "http://vllm-server:8000/v1",
                "task_id": task_id,
                "temperature": TEMPERATURE,
                "max_round": 30
            }

            try:
                start_ts = time.time()
                response = requests.post("http://localhost:8001/evaluate", json=payload, timeout=2500)
                result = response.json()

                latency = result.get('time_taken', time.time() - start_ts)
                score = result.get('score', 0.0)

                total_score += score
                total_time += latency

                all_results.append({
                    "task_id": task_id,
                    "task_name": result.get('task_name', 'unknown'),
                    "score": score,
                    "success": result.get('success', False),
                    "time": latency,
                    "error": result.get('error')
                })
                print(f" Done (Score: {score})")
            except Exception as e:
                print(f" Failed: {e}")

        # 4. Final Aggregation & File Writing
        avg_score = total_score / len(all_results) if all_results else 0
        avg_time = total_time / len(all_results) if all_results else 0


        safe_model_name = BASE_MODEL_NAME.split("/")[1]

        if LORA_MODEL_NAME:
            safe_lora_name = LORA_MODEL_NAME.split("/")[1]
            filename = f"eval_results_{safe_model_name}_{safe_lora_name}.txt"
        else:
            filename = f"eval_results_{safe_model_name}.txt"

        with open(filename, "w") as f:
            f.write("="*40 + "\n")
            f.write(f"EVALUATION REPORT - {datetime.now()}\n")
            f.write(f"Model: {BASE_MODEL_NAME}\n")
            f.write("="*40 + "\n\n")
            f.write(f"SUMMARY STATS:\n")
            f.write(f"- Total Tasks: {len(all_results)}\n")
            f.write(f"- Average Score: {avg_score:.4f}\n")
            f.write(f"- Average Time Per Episode: {avg_time:.2f}s\n\n")
            f.write("DETAILED RESULTS:\n")
            f.write(f"{'Task ID':<10} | {'Name':<15} | {'Score':<7} | {'Success':<8} | {'Time':<7}\n")
            f.write("-" * 60 + "\n")
            for res in all_results:
                f.write(f"{res['task_id']:<10} | {res['task_name']:<15} | {res['score']:<7} | {str(res['success']):<8} | {res['time']:<7.2f}s\n")
                if res['error']:
                    f.write(f"   └─ Error: {res['error']}\n")

        print(f"\n✅ Evaluation complete. Results saved to: {filename}")

    finally:
        print("🧹 Cleaning up containers...")
        for c in containers.values():
            try: c.remove(force=True)
            except: pass

if __name__ == "__main__":
    run_random_eval_suite()