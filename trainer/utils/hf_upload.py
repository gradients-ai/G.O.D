import glob
import json
import os
import re
import shutil
import subprocess

import wandb
from huggingface_hub import HfApi
from huggingface_hub import login


def patch_model_metadata(output_dir: str, base_model_id: str):
    try:
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                config = json.load(f)

            config["base_model_name_or_path"] = base_model_id

            with open(adapter_config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"Updated adapter_config.json with base_model: {base_model_id}", flush=True)
        else:
            print(" adapter_config.json not found", flush=True)

        readme_path = os.path.join(output_dir, "README.md")

        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.strip().startswith("base_model:"):
                    new_lines.append(f"base_model: {base_model_id}\n")
                else:
                    new_lines.append(line)

            with open(readme_path, "w") as f:
                f.writelines(new_lines)

            print(f"Updated README.md with base_model: {base_model_id}", flush=True)
        else:
            print("README.md not found", flush=True)

    except Exception as e:
        print(f"Error updating metadata: {e}", flush=True)
        pass

            
def is_folder_empty_or_metadata_only(folder_path: str) -> bool:
    """Check if folder is empty or only contains small metadata files like .gitattributes."""
    if not os.path.isdir(folder_path):
        return True
    
    items = os.listdir(folder_path)
    if not items:
        return True
    
    metadata_files = {'.git', '.gitattributes', '.gitignore', '.gitkeep'}
    for item in items:
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            if item != '.git':
                return False
        elif item not in metadata_files:
            if os.path.getsize(item_path) > 1024:
                return False
    
    return True


def sync_wandb_logs(cache_dir: str):
    sync_root = os.path.join(cache_dir, "wandb")
    run_dirs = glob.glob(os.path.join(sync_root, "offline-run-*"))

    if not run_dirs:
        print("No offline runs found.")
        return

    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir).split("-")[-1]
        print(f"Syncing run: {run_dir}")

        try:
            proc = subprocess.run(
                ["wandb", "sync", "--include-offline", run_dir],
                check=True,
                capture_output=True,
                text=True
            )
            output = proc.stdout + proc.stderr
            match = re.search(r"https://wandb\.ai/\S+", output)
            run_url = match.group(0) if match else None

            if run_url:
                print(f"Synced W&B Run: {run_url}")

            print(f"Synced Run: {run_id}")
            shutil.rmtree(run_dir)
            print(f"Deleted synced folder: {run_dir}")

        except Exception as e:
            print(f"Failed to sync {run_dir}: {e}")


def detect_subfolder(base_folder: str) -> str | None:
    if not os.path.isdir(base_folder):
        return None
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if not os.path.isdir(item_path):
            continue
        has_checkpoint_files = False
        for file in os.listdir(item_path):
            file_path = os.path.join(item_path, file)
            if file.endswith('.safetensors'):
                has_checkpoint_files = True
                break
        
        if has_checkpoint_files:
            return item_path
    
    return None


def main():
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    hf_user = os.getenv("HUGGINGFACE_USERNAME")
    wandb_token = os.getenv("WANDB_TOKEN")
    task_id = os.getenv("TASK_ID")
    repo_name = os.getenv("EXPECTED_REPO_NAME")
    local_folder = os.getenv("LOCAL_FOLDER")
    model = os.getenv("MODEL")
    repo_subfolder = os.getenv("HF_REPO_SUBFOLDER", None)
    wandb_logs_path = os.getenv("WANDB_LOGS_PATH", None)

    if repo_subfolder:
        repo_subfolder = repo_subfolder.strip("/")

    if not all([hf_token, hf_user, task_id, repo_name]):
        raise RuntimeError("Missing one or more required environment variables")

    login(token=hf_token)

    repo_id = f"{hf_user}/{repo_name}"

    if not os.path.isdir(local_folder):
        raise FileNotFoundError(f"Local folder {local_folder} does not exist")

    checkpoint_subfolder = detect_subfolder(local_folder)
    if checkpoint_subfolder:
        local_folder = checkpoint_subfolder
    
    if is_folder_empty_or_metadata_only(local_folder):
        raise ValueError(f"Local folder {local_folder} is empty or only contains metadata files. Nothing to upload.")

    patch_model_metadata(local_folder, model)

    print(f"Creating repo {repo_id}...", flush=True)
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, private=False)

    print(f"Uploading contents of {local_folder} to {repo_id}", flush=True)
    if repo_subfolder:
        print(f"Uploading into subfolder: {repo_subfolder}", flush=True)

    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_folder,
        path_in_repo=repo_subfolder if repo_subfolder else None,
        commit_message=f"Upload task output {task_id}",
        token=hf_token,
    )

    print(f"Uploaded successfully to https://huggingface.co/{repo_id}", flush=True)

    if wandb_token:
        try:
            wandb.login(key=wandb_token)
            sync_wandb_logs(cache_dir=wandb_logs_path)
        except Exception as e:
            print(f"Failed to sync W&B logs: {e}", flush=True)

    try:
        if os.path.isdir(local_folder):
            shutil.rmtree(local_folder)
            print(f"Deleted local folder: {local_folder}", flush=True)
    except Exception as e:
        print(f"Failed to delete local folder {local_folder}: {e}", flush=True)


if __name__ == "__main__":
    main()
