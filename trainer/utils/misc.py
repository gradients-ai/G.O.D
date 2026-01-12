import os
import shutil
from urllib.parse import urlparse

import pynvml
from git import GitCommandError
from git import Repo

import trainer.constants as cst
from core.models.utility_models import GPUInfo
from core.models.utility_models import GPUType
from trainer.tasks import get_running_tasks


def clone_repo(repo_url: str, parent_dir: str, branch: str = None, commit_hash: str = None) -> str:
    repo_name = os.path.basename(urlparse(repo_url).path)
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    repo_dir = os.path.join(parent_dir, repo_name)

    if os.path.exists(repo_dir):
        try:
            repo = Repo(repo_dir)
            current_commit = repo.head.commit.hexsha

            if commit_hash and current_commit.startswith(commit_hash):
                return repo_dir
            elif branch and repo.active_branch.name == branch:
                return repo_dir
            shutil.rmtree(repo_dir)
        except:
            shutil.rmtree(repo_dir)

    try:
        repo = Repo.clone_from(repo_url, repo_dir, branch=branch) if branch else Repo.clone_from(repo_url, repo_dir)

        if commit_hash:
            repo.git.fetch("--all")
            try:
                repo.git.checkout(commit_hash)
            except GitCommandError as checkout_error:
                # Check if it's an invalid commit hash format issue
                if "pathspec" in str(checkout_error) and "did not match any file(s) known to git" in str(checkout_error):
                    raise RuntimeError(f"Invalid commit hash '{commit_hash}' - commit not found in repository")
                else:
                    # Re-raise other git checkout errors
                    raise

        return repo_dir

    except GitCommandError as e:
        raise RuntimeError(f"Error in cloning: {str(e)}")

    except Exception as e:
        raise RuntimeError(f"Unexpected error while cloning: {str(e)}")


async def get_gpu_info() -> list[GPUInfo]:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    index_to_type: dict[int, GPUType] = {}
    index_to_vram: dict[int, int] = {}

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle).decode("utf-8").upper()
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_vram_gb = int(mem_info.total / 1024 / 1024 / 1024)

        for gpu_type in GPUType:
            if gpu_type.value in name:
                index_to_type[i] = gpu_type
                index_to_vram[i] = total_vram_gb
                break

    busy_gpu_ids: set[int] = set()
    running_tasks = get_running_tasks()

    if running_tasks:
        for task in running_tasks:
            for gpu_id in task.gpu_ids:
                busy_gpu_ids.add(gpu_id)

    gpu_infos: list[GPUInfo] = []
    for gpu_id in range(device_count):
        if gpu_id not in index_to_type:
            continue

        gpu_info = GPUInfo(
            gpu_id=gpu_id,
            gpu_type=index_to_type[gpu_id],
            vram_gb=index_to_vram[gpu_id],
            available=gpu_id not in busy_gpu_ids,
        )
        gpu_infos.append(gpu_info)

    pynvml.nvmlShutdown()
    return gpu_infos


def build_wandb_env(task_id: str, hotkey: str) -> dict:
    wandb_path = f"{cst.WANDB_LOGS_DIR}/{task_id}_{hotkey}"

    env = {
        "WANDB_MODE": "offline",
        **{key: wandb_path for key in cst.WANDB_DIRECTORIES}
    }

    return env

def extract_container_error(logs: str) -> str | None:
    lines = logs.strip().splitlines()
    if lines:
        for line in reversed(lines):
            line = line.strip()
            if line and ":" in line and any(word in line for word in ["Error", "Exception"]):
                return line

    return None


def are_gpus_available(requested_gpu_ids: list[int]) -> bool:
    """
    Check if any of the requested GPU IDs are already in use by training tasks.
    
    Returns:
        bool: True if all requested GPUs are available, False otherwise
    """
    running_tasks = get_running_tasks()
    
    for task in running_tasks:
        for gpu_id in requested_gpu_ids:
            if gpu_id in task.gpu_ids:
                return False
    
    return True

