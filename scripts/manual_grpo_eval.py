#!/usr/bin/env python3
"""
Manual GRPO evaluation script using the actual evaluation container.
Usage: python -m scripts.manual_grpo_eval <task_id> <model_repo>
"""
import asyncio
import os
import subprocess
import sys
from uuid import UUID

import asyncpg
from huggingface_hub import snapshot_download

from core import constants as cst
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import RewardFunction
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def get_available_gpus(max_gpus=2):
    """Get list of available GPU IDs with low memory usage"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,memory.used,memory.total', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        available_gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                gpu_id, mem_used, mem_total = line.split(', ')
                gpu_id = int(gpu_id)
                mem_used = int(mem_used)
                mem_total = int(mem_total)
                
                # Consider GPU available if less than 10% memory is used
                if mem_used / mem_total < 0.1:
                    available_gpus.append(gpu_id)
                
                if len(available_gpus) >= max_gpus:
                    break
        
        if not available_gpus:
            logger.warning("No GPUs with low memory usage found, using GPU 0")
            return [0]
        
        logger.info(f"Using GPUs: {available_gpus}")
        return available_gpus
        
    except Exception as e:
        logger.error(f"Failed to check GPU availability: {e}")
        logger.info("Falling back to GPU 0")
        return [0]


def load_env_file():
    """Load environment variables from .vali.env file"""
    try:
        with open('.vali.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
                    print(f"🔧 Loaded {key} from .vali.env")
    except FileNotFoundError:
        print("⚠️  .vali.env file not found")


async def get_task_info(task_id: str):
    """Get task information from database"""
    connection_string = os.getenv("DATABASE_URL")
    if not connection_string:
        raise ValueError("DATABASE_URL not found")
    
    pool = await asyncpg.create_pool(connection_string)
    
    try:
        async with pool.acquire() as conn:
            # Get task info including GRPO specific data
            query = """
                SELECT t.task_id, t.model_id, t.ds as dataset_url, t.test_data, t.training_data, 
                       t.task_type, gt.field_prompt, gt.file_format, gt.synthetic_data, gt.extra_column
                FROM tasks t
                JOIN grpo_tasks gt ON t.task_id = gt.task_id
                WHERE t.task_id = $1
            """
            task_row = await conn.fetchrow(query, UUID(task_id))
            
            if not task_row:
                raise ValueError(f"Task {task_id} not found")
            
            # Get reward functions
            reward_query = """
                SELECT rf.reward_id, rf.reward_func, gtf.reward_weight, rf.func_hash, rf.is_generic
                FROM grpo_task_functions gtf
                JOIN reward_functions rf ON gtf.reward_id = rf.reward_id
                WHERE gtf.task_id = $1
            """
            reward_rows = await conn.fetch(reward_query, UUID(task_id))
            
            return task_row, reward_rows
            
    finally:
        await pool.close()


def run_grpo_evaluation(task_info, reward_info, model_repo: str):
    """Run GRPO evaluation using Docker container"""
    
    task_row = task_info
    
    # Get available GPUs
    gpu_ids = get_available_gpus(max_gpus=2)
    
    # Extract info from task
    dataset_url = task_row['dataset_url'] or (task_row['synthetic_data'] if task_row['synthetic_data'] else None)
    original_model = task_row['model_id']
    file_format = FileFormat(task_row['file_format']) if task_row['file_format'] else FileFormat.S3
    
    cache_dir = os.path.expanduser(cst.CACHE_DIR_HUB)
    
    # Download the original GRPO model (exactly like main validator does)
    print(f"📥 Downloading original GRPO model: {original_model}")
    try:
        original_model_path = snapshot_download(
            repo_id=original_model, 
            cache_dir=cache_dir,
            ignore_patterns=None
        )
        print(f"✅ Original model downloaded to: {original_model_path}")
        
        if os.path.exists(original_model_path):
            files = os.listdir(original_model_path)
            print(f"📁 Original model files downloaded: {files}")
            tokenizer_files = [f for f in files if 'tokenizer' in f.lower() or f.endswith('.model')]
            if tokenizer_files:
                print(f"🔤 Tokenizer files found: {tokenizer_files}")
            else:
                print(f"⚠️ WARNING: No tokenizer files found in original model download!")
    except Exception as e:
        print(f"❌ Failed to download original model {original_model}: {e}")
        return False
    
    # Download the model to evaluate (exactly like main validator does)
    print(f"📥 Starting download of model {model_repo}...")
    try:
        model_path = snapshot_download(
            repo_id=model_repo, 
            cache_dir=cache_dir,
            ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.pkl", "*.pth"]
        )
        print(f"✅ Model {model_repo} downloaded to: {model_path}")
        
        # Log what files are actually in the downloaded model (like main validator)
        if os.path.exists(model_path):
            files = os.listdir(model_path)
            print(f"📁 Downloaded files for {model_repo}: {files}")
            
            # Check file sizes to ensure they're not just LFS pointers (like main validator)
            for file in files:
                if file.endswith(('.safetensors', '.bin')):
                    file_path = os.path.join(model_path, file)
                    file_size = os.path.getsize(file_path)
                    print(f"  {file}: {file_size / (1024*1024*1024):.2f} GB")
                    
                    # LFS pointer files are typically < 1KB
                    if file_size < 1000:
                        print(f"⚠️  WARNING: {file} appears to be an LFS pointer (only {file_size} bytes)")
            
            # Check for essential files (like main validator)
            has_config = 'config.json' in files
            has_weights = any(f.endswith(('.safetensors', '.bin')) for f in files)
            print(f"🔍 Model validation - has config.json: {has_config}, has model weights: {has_weights}")
        else:
            print(f"❌ Model path does not exist after download: {model_path}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download model {model_repo}: {e}")
        return False
    
    # Create RewardFunction objects (matching validator pattern)
    reward_functions = []
    for row in reward_info:
        if row['reward_func']:
            reward_function = RewardFunction(
                reward_id=str(row['reward_id']),
                reward_func=row['reward_func'],
                reward_weight=float(row['reward_weight']),
                func_hash=row['func_hash'],
                is_generic=row['is_generic']
            )
            reward_functions.append(reward_function)
    
    # Create GrpoDatasetType (matching validator pattern)
    dataset_type = GrpoDatasetType(
        field_prompt=task_row['field_prompt'],
        reward_functions=reward_functions,
        extra_column=task_row['extra_column'] if task_row['extra_column'] else 'extra_data'
    )
    
    print(f"🚀 Starting GRPO evaluation for task {task_row['task_id']}")
    print(f"📊 Dataset: {dataset_url}")
    print(f"🤖 Original Model: {original_model}")
    print(f"🎯 Model to evaluate: {model_repo}")
    print(f"🏆 Reward functions: {len(reward_functions)}")
    print(f"📋 Field prompt: {task_row['field_prompt']}")
    print(f"📝 Extra column: {dataset_type.extra_column}")
    print(f"🎮 Using GPUs: {gpu_ids}")
    
    # Get HF cache directory (platform independent)
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    
    # Create GPU device specification for Docker
    gpu_device_requests = f"\"device={','.join(map(str, gpu_ids))}\""
    
    # Docker command to run evaluation (matching validator pattern)
    docker_cmd = [
        "docker", "run", "--rm",
        "--gpus", gpu_device_requests,
        "-v", f"{hf_cache}:/root/.cache/huggingface",
        "-e", f"DATASET={dataset_url}",
        "-e", f"ORIGINAL_MODEL={original_model}",
        "-e", f"MODELS={model_repo}",
        "-e", f"FILE_FORMAT={file_format.value}",
        "-e", f"DATASET_TYPE={dataset_type.model_dump_json()}",  # Use pydantic serialization
        "-e", "TRANSFORMERS_ALLOW_TORCH_LOAD=true",
        "-e", "HF_HOME=/root/.cache/huggingface",
        "-e", "TRANSFORMERS_CACHE=/root/.cache/huggingface/hub", 
        "-e", "HF_DATASETS_CACHE=/root/.cache/huggingface/datasets",
        cst.VALIDATOR_DOCKER_IMAGE,
        "python", "-m", "validator.evaluation.eval_grpo"
    ]
    
    print(f"\n🐳 Docker image: {cst.VALIDATOR_DOCKER_IMAGE}")
    print(f"🐳 Running Docker command:")
    print(f"   docker run --rm --gpus {gpu_device_requests} ... {cst.VALIDATOR_DOCKER_IMAGE}")
    
    try:
        print(f"\n📋 Container Output (streaming):")
        print("=" * 80)
        
        # Stream output in real-time without capturing
        result = subprocess.run(
            docker_cmd,
            timeout=1800  # 30 minute timeout
        )
        
        print("=" * 80)
        print(f"\n✅ Exit code: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Evaluation timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False


async def main():
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.manual_grpo_eval <task_id> <model_repo>")
        print("Example: python -m scripts.manual_grpo_eval ba444f8a-dd60-40e2-bdad-524e1e754457 microsoft/Phi-3-mini-128k-instruct")
        sys.exit(1)
    
    task_id = sys.argv[1]
    model_repo = sys.argv[2]
    
    # Load environment variables
    load_env_file()
    
    try:
        print(f"🔍 Fetching task information for {task_id}...")
        task_info, reward_info = await get_task_info(task_id)
        
        print(f"✅ Found task: {task_info['task_type']}")
        print(f"📊 Dataset: {task_info['dataset_url'] or 'synthetic_data'}")
        print(f"🤖 Original model: {task_info['model_id']}")
        
        success = run_grpo_evaluation(task_info, reward_info, model_repo)
        
        if success:
            print("🎉 Evaluation completed successfully!")
        else:
            print("💥 Evaluation failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())