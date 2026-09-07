import asyncio
import os
import shutil
from urllib.parse import urlparse

import docker
from git import GitCommandError
from git import Repo

import trainer.constants as cst
from core.git import build_authenticated_git_url
from core.git import sanitize_git_text
from core.models.trainer_contract_models import GPUInfo
from core.models.trainer_contract_models import GPUInterconnect
from core.models.trainer_contract_models import GPUType
from trainer.job_state import get_running_jobs


def _gpu_has_nvlink(pynvml, handle) -> bool:
    """Return True if any NVLink link is enabled. Older pynvml builds may lack these APIs."""
    get_state = getattr(pynvml, "nvmlDeviceGetNvLinkState", None)
    if get_state is None:
        return False
    max_links = getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 18)
    enabled = getattr(pynvml, "NVML_FEATURE_ENABLED", 1)
    for link in range(max_links):
        try:
            if get_state(handle, link) == enabled:
                return True
        except Exception:
            # Links past the device's max raise; stop scanning.
            break
    return False


def _detect_interconnect(pynvml, handle, product_name: str) -> tuple[GPUInterconnect, bool]:
    """Infer SXM / PCIe / NVL from product name, multi-GPU board bit, and NVLink state.

    Product names usually encode the form factor (e.g. "H100 PCIe", "H100 NVL",
    "A100-SXM4-80GB"). Plain "H100 80GB HBM3" on NVLink is typically SXM/HGX.
    """
    upper = product_name.upper()
    has_nvlink = _gpu_has_nvlink(pynvml, handle)

    multi_gpu_board = False
    try:
        multi_gpu_board = bool(pynvml.nvmlDeviceGetMultiGpuBoard(handle))
    except Exception:
        pass

    if "NVL" in upper or multi_gpu_board:
        return GPUInterconnect.NVL, has_nvlink
    if "PCIE" in upper or "PCI-E" in upper:
        return GPUInterconnect.PCIE, has_nvlink
    if "SXM" in upper:
        return GPUInterconnect.SXM, has_nvlink
    # H100 SXM usually reports as "NVIDIA H100 80GB HBM3" (no "SXM"/"PCIe" token).
    if "H100" in upper and "HBM3" in upper:
        return GPUInterconnect.SXM, has_nvlink
    if has_nvlink:
        return GPUInterconnect.SXM, has_nvlink
    if "H100" in upper or "A100" in upper:
        return GPUInterconnect.PCIE, has_nvlink
    return GPUInterconnect.UNKNOWN, has_nvlink


def clone_repo(
    repo_url: str,
    parent_dir: str,
    commit_hash: str,
    github_token: str | None = None,
    task_id: str | None = None,
    hotkey: str | None = None,
) -> str:
    path = urlparse(repo_url).path.rstrip("/")
    repo_name = os.path.basename(path) or "repo"
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    unique_suffix = f"{task_id}_{hotkey[:8]}" if task_id and hotkey else None
    dir_name = f"{repo_name}_{unique_suffix}" if unique_suffix else repo_name
    repo_dir = os.path.join(parent_dir, dir_name)
    os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(repo_dir):
        try:
            repo = Repo(repo_dir)
            current_commit = repo.head.commit.hexsha
            if current_commit.startswith(commit_hash):
                return repo_dir
            shutil.rmtree(repo_dir)
        except Exception:
            shutil.rmtree(repo_dir)

    try:
        clone_url = build_authenticated_git_url(repo_url, github_token)
        repo = Repo.clone_from(clone_url, repo_dir)
        repo.git.fetch("--all")
        repo.git.fetch("origin")
        try:
            repo.git.checkout(commit_hash)
        except GitCommandError as checkout_error:
            if "pathspec" in str(checkout_error) and "did not match any file(s) known to git" in str(checkout_error):
                raise RuntimeError(f"Invalid commit hash '{commit_hash}' - commit not found in repository")
            raise

        return repo_dir

    except GitCommandError as e:
        raise RuntimeError(f"Error in cloning: {sanitize_git_text(str(e), github_token)}")

    except Exception as e:
        raise RuntimeError(f"Unexpected error while cloning: {sanitize_git_text(str(e), github_token)}")


def _get_gpu_info_sync() -> list[GPUInfo]:
    import pynvml

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    index_to_type: dict[int, GPUType] = {}
    index_to_vram: dict[int, int] = {}
    index_to_product_name: dict[int, str] = {}
    index_to_interconnect: dict[int, GPUInterconnect] = {}
    index_to_nvlink: dict[int, bool] = {}

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        raw_name = pynvml.nvmlDeviceGetName(handle)
        product_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else raw_name
        name = product_name.upper()
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_vram_gb = int(mem_info.total / 1024 / 1024 / 1024)

        for gpu_type in GPUType:
            if gpu_type.value in name:
                index_to_type[i] = gpu_type
                index_to_vram[i] = total_vram_gb
                index_to_product_name[i] = product_name
                interconnect, has_nvlink = _detect_interconnect(pynvml, handle, product_name)
                index_to_interconnect[i] = interconnect
                index_to_nvlink[i] = has_nvlink
                break

    busy_gpu_ids: set[int] = set()
    running_jobs = get_running_jobs()

    if running_jobs:
        for job in running_jobs:
            for gpu_id in job.gpu_ids:
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
            product_name=index_to_product_name[gpu_id],
            interconnect=index_to_interconnect[gpu_id],
            nvlink=index_to_nvlink[gpu_id],
        )
        gpu_infos.append(gpu_info)

    pynvml.nvmlShutdown()
    return gpu_infos


async def get_gpu_info() -> list[GPUInfo]:
    return await asyncio.to_thread(_get_gpu_info_sync)


def build_wandb_env(task_id: str, hotkey: str) -> dict:
    wandb_path = f"{cst.WANDB_LOGS_DIR}/{task_id}_{hotkey}"

    env = {"WANDB_MODE": "offline", **{key: wandb_path for key in cst.WANDB_DIRECTORIES}}

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
    Check if any of the requested GPU IDs are already in use by running jobs.

    Returns:
        bool: True if all requested GPUs are available, False otherwise
    """
    running_jobs = get_running_jobs()

    for job in running_jobs:
        for gpu_id in requested_gpu_ids:
            if gpu_id in job.gpu_ids:
                return False

    busy_gpu_ids = _get_busy_gpu_ids_from_running_containers()
    for gpu_id in requested_gpu_ids:
        if gpu_id in busy_gpu_ids:
            return False

    return True


def _get_busy_gpu_ids_from_running_containers() -> set[int]:
    busy_gpu_ids: set[int] = set()
    try:
        client = docker.from_env()
        containers = client.containers.list()
        trainer_containers = [
            c
            for c in containers
            if c.name.startswith("text-trainer-")
            or c.name.startswith("image-trainer-")
            or c.name.startswith("model-prep-")
        ]
        for container in trainer_containers:
            device_requests = container.attrs.get("HostConfig", {}).get("DeviceRequests", []) or []
            for request in device_requests:
                device_ids = request.get("DeviceIDs") or []
                for device_id in device_ids:
                    if str(device_id).isdigit():
                        busy_gpu_ids.add(int(device_id))
    except Exception:
        return set()
    return busy_gpu_ids
