import docker
import time
import requests
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import snapshot_download

# --- Model Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_REVISION = None
LORA_MODEL_NAME = None # Put the name of your repo containing the LORA here
LORA_MODEL_REVISION = None

# --- Evaluation Configuration ---
GAME_TO_EVAL = "gin_rummy"
OPPONENT_TYPE = "mcts"
MCTS_MAX_SIMULATIONS = 25
MCTS_NUM_ROLLOUTS = 1
NUM_EVALS = 100
TEMPERATURE = 0.0
RANDOM_SEED = 42
NUM_CONCURRENT_EVAL_WORKERS = 5

##############################################################################################


client = docker.from_env()

GAMES_TO_TASK_ID_RANGE = {
    "goofspiel": (0, 99999999),
    "liars_dice": (100000000, 199999999),
    "leduc_poker": (200000000, 299999999),
    "gin_rummy": (300000000, 399999999),
    "othello": (400000000, 499999999),
    "backgammon": (500000000, 599999999),
    "hex": (600000000, 699999999),
    "clobber": (700000000, 799999999),
}
SGLANG_IMAGE = "lmsysorg/sglang:latest"
AGENTGYM_IMAGE = "phoenixbeaudry/game:mcts-api"
NETWORK_NAME = "agent_eval_net"
SGLANG_PORT = 30000
HF_CACHE_DIR = "/mnt/hf_cache"
TASK_ID_MIN, TASK_ID_MAX = GAMES_TO_TASK_ID_RANGE[GAME_TO_EVAL]

def run_evaluation():
    containers = {}
    avg_score = 0.0
    win_count = 0

    try:
        # 1. Infrastructure Setup
        networks = client.networks.list(names=[NETWORK_NAME])
        if not networks: client.networks.create(NETWORK_NAME, driver="bridge")

        lora_dir = None
        if LORA_MODEL_NAME:
            print(f"🚀 Starting SGLang: {BASE_MODEL_NAME} w/ lora {LORA_MODEL_NAME}")
            safe_lora_name = LORA_MODEL_NAME.replace("/", "_")
            lora_dir = f"/tmp/sglang_lora/{safe_lora_name}"
            print(f"⬇️  Downloading LoRA to {lora_dir}...")
            snapshot_download(
                repo_id=LORA_MODEL_NAME,
                revision=LORA_MODEL_REVISION,
                local_dir=lora_dir,
                local_dir_use_symlinks=False,
            )
            sglang_command = (
                f"python3 -m sglang.launch_server --model-path {BASE_MODEL_NAME} "
                "--enable-lora --lora-paths trained_lora=/lora/trained_lora "
                "--lora-backend triton "
                "--host 0.0.0.0 --port 30000 --tensor-parallel-size 1 --dtype float16 --enable-deterministic-inference "
                f"--random-seed {RANDOM_SEED}"
            )
        else:
            print(f"🚀 Starting SGLang: {BASE_MODEL_NAME}")
            sglang_command = (
                f"python3 -m sglang.launch_server --model-path {BASE_MODEL_NAME} "
                f"{'--revision ' + BASE_MODEL_REVISION if BASE_MODEL_REVISION else ''} "
                "--host 0.0.0.0 --port 30000 --tensor-parallel-size 1 --dtype float16 --enable-deterministic-inference "
                f"--random-seed {RANDOM_SEED}"
            )

        sglang = client.containers.run(
            SGLANG_IMAGE,
            command=sglang_command,
            name="sglang-server",
            detach=True,
            network=NETWORK_NAME,
            ports={f"{SGLANG_PORT}/tcp": SGLANG_PORT},
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
            environment={
                "HF_HOME": "/hf",
                "TRANSFORMERS_CACHE": "/hf",
                "HUGGINGFACE_HUB_CACHE": "/hf",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PYTHONHASHSEED": str(RANDOM_SEED),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "NVIDIA_TF32_OVERRIDE": "0",
            },
            volumes={
                HF_CACHE_DIR: {"bind": "/hf", "mode": "rw"},
                **({lora_dir: {"bind": "/lora/trained_lora", "mode": "ro"}} if lora_dir else {}),
            },
            ipc_mode="host",
        )
        containers['sglang'] = sglang

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
        print("⏳ Waiting for SGLang health check...")
        while True:
            try:
                if requests.get(f"http://localhost:{SGLANG_PORT}/v1/models", timeout=2).status_code == 200:
                    break
            except:
                time.sleep(5)
        print("✅ SGLang Ready.\n")

        # 3. Evaluation Loop
        random.seed(RANDOM_SEED)
        eval_list = random.sample(range(TASK_ID_MIN, TASK_ID_MAX), NUM_EVALS)
        total_score = 0.0

        if LORA_MODEL_NAME:
            # For OpenAI-compatible API, use base-model:adapter-name format per SGLang docs
            # Format: model_path:adapter_name (e.g., "Qwen/Qwen2.5-3B-Instruct:trained_lora")
            inference_model_name = f"{BASE_MODEL_NAME}:trained_lora"
        else:
            inference_model_name = BASE_MODEL_NAME

        def evaluate_task(task_id):
            payload = {
                "model": inference_model_name,
                "base_url": f"http://sglang-server:{SGLANG_PORT}/v1",
                "task_id": task_id,
                "temperature": TEMPERATURE,
                "seed": task_id,
                "opponent": OPPONENT_TYPE,
                "api_key": "test",
                "mcts_max_simulations": MCTS_MAX_SIMULATIONS,
                "mcts_num_rollouts": MCTS_NUM_ROLLOUTS
            }
            try:
                response = requests.post("http://localhost:8001/evaluate", json=payload, timeout=2500)
                result = response.json()
                result_payload = result.get("result") if isinstance(result, dict) else None
                if isinstance(result_payload, dict):
                    data = result_payload
                else:
                    data = result if isinstance(result, dict) else {}
                return task_id, data.get('score', 0.0), None
            except Exception as e:
                return task_id, 0.0, str(e)

        print(f"Running {NUM_EVALS} evaluations with concurrency={NUM_CONCURRENT_EVAL_WORKERS}...")
        completed = 0
        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_EVAL_WORKERS) as executor:
            futures = {executor.submit(evaluate_task, task_id): task_id for task_id in eval_list}
            for future in as_completed(futures):
                task_id, score, error = future.result()
                completed += 1
                total_score += score
                if score == 1.0:
                    win_count += 1
                if error:
                    print(f"[{completed}/{NUM_EVALS}] Task {task_id}: FAILED ({error})")
                else:
                    print(f"[{completed}/{NUM_EVALS}] Task {task_id}: {score}")

        # 4. Final Score
        avg_score = total_score / NUM_EVALS if NUM_EVALS > 0 else 0

        print(f"\n✅ Evaluation complete.")
        print(f"Score: {total_score}/{NUM_EVALS} ({avg_score:.4f})")
        print(f"Win Count: {win_count}/{NUM_EVALS} ({win_count/NUM_EVALS})")

    finally:
        print("🧹 Cleaning up containers...")
        for c in containers.values():
            try: c.remove(force=True)
            except: pass


if __name__ == "__main__":
    start = time.perf_counter()
    run_evaluation()
    end = time.perf_counter()
    elapsed = end-start
    print(f"Evaluation took: {elapsed:.4f} seconds")