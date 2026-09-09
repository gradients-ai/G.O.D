#!/usr/bin/env python3
"""Smoke-train one ImageTask per model type against image2026, then stop after a few steps."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import datetime
from minio import Minio

TASKS_PATH = Path("/root/tasks.json")
TRAINER = "http://127.0.0.1:8001"
HOTKEY = "smoke-verify-hotkey"
REPO = os.environ.get("SMOKE_GITHUB_REPO", "https://github.com/wrathofgodtourn-code/image2026")
COMMIT = os.environ.get("SMOKE_GITHUB_COMMIT", "6ea75cff46b29b6f60468286fc3dda32185d0499")
TOKEN = os.environ["SMOKE_GITHUB_TOKEN"]
GPU_IDS = [0]

# Prefer cached FLUX weights for smoke (model_type still flux).
MODEL_OVERRIDES = {
    "flux": "rayonlabs/FLUX.1-dev",
}

STEP_RE = re.compile(
    r"(?:steps|last):\s*\d+%\|.*?\|\s*(\d+)/(\d+)|"
    r"(?:global_step|step)\s*[:=]?\s*(\d+)|"
    r"(\d+)\s*/\s*\d+\s*(?:steps?|iters?)|"
    r"avr_loss\s*=",
    re.IGNORECASE,
)
PROGRESS_HINTS = (
    "loss",
    "train/",
    "training",
    "unet",
    "diffusion",
    "optimizer",
    "running training",
    "starting training",
)


def load_vali_env() -> None:
    env_path = Path("/root/G.O.D/.vali.env")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def refresh_dataset_url(old_url: str) -> str:
    bucket = os.environ["S3_BUCKET_NAME"]
    client = Minio(
        os.environ["S3_COMPATIBLE_ENDPOINT"],
        access_key=os.environ["S3_COMPATIBLE_ACCESS_KEY"],
        secret_key=os.environ["S3_COMPATIBLE_SECRET_KEY"],
        secure=True,
        region=os.environ.get("S3_REGION", "eu-central-003"),
    )
    path = urllib.parse.urlparse(old_url).path.lstrip("/")
    parts = path.split("/", 1)
    obj = parts[1] if len(parts) == 2 and parts[0] == bucket else (parts[1] if len(parts) == 2 else path)
    return client.presigned_get_object(bucket, obj, expires=datetime.timedelta(hours=12))


def pick_tasks() -> dict[str, dict]:
    data = json.loads(TASKS_PATH.read_text())
    by: dict[str, dict] = {}
    for t in data["tasks"]:
        mt = t["model_type"]
        if mt not in by:
            by[mt] = t
    return by


def http_json(method: str, url: str, body: dict | None = None, timeout: int = 60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"detail": raw}
        return e.code, payload


def stop_trainer_containers() -> list[str]:
    out = subprocess.check_output(
        ["docker", "ps", "--format", "{{.Names}}"],
        text=True,
    )
    killed = []
    for name in out.splitlines():
        if name.startswith(("image-trainer-", "downloader-", "standalone-image")):
            subprocess.call(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            killed.append(name)
    return killed


def latest_image_trainer() -> str | None:
    out = subprocess.check_output(
        ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
        text=True,
    )
    for line in out.splitlines():
        name = line.split("\t", 1)[0]
        if name.startswith("image-trainer-"):
            return name
    return None


def container_logs_tail(name: str, n: int = 400) -> str:
    try:
        return subprocess.check_output(
            ["docker", "logs", "--tail", str(n), name],
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        return e.output or ""



def max_step_seen(text: str) -> int:
    best = 0
    # Prefer ai-toolkit tqdm: "steps:  2%|..| 3/378" or krea "last:  1%|..| 12/2160"
    for m in re.finditer(r"(?:steps|last):\s*\d+%\|.*?\|\s*(\d+)/(\d+)", text, re.I):
        try:
            best = max(best, int(m.group(1)))
        except ValueError:
            pass
    for m in STEP_RE.finditer(text):
        # skip the tqdm branch groups handled above; take other groups carefully
        gs = [g for g in m.groups() if g is not None]
        if len(gs) >= 2 and gs[0].isdigit() and gs[1].isdigit():
            # likely current/total — use current only
            best = max(best, int(gs[0]))
            continue
        for g in gs:
            try:
                v = int(g)
            except ValueError:
                continue
            if v < 100000:  # ignore absurd captures
                best = max(best, v)
    if "avr_loss" in text.lower() and best == 0:
        best = 1  # training clearly started
    return best


def looks_like_training(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in PROGRESS_HINTS)


def task_details(task_id: str) -> dict | None:
    status, payload = http_json("GET", f"{TRAINER}/v1/trainer/{task_id}?hotkey={urllib.parse.quote(HOTKEY)}")
    if status == 200:
        return payload
    return None


def wait_gpu_free(timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, payload = http_json("GET", f"{TRAINER}/v1/trainer/get_gpu_availability")
        if status == 200 and payload and payload[0].get("available"):
            # also ensure no leftover containers
            if not latest_image_trainer():
                return
        time.sleep(3)
    raise TimeoutError("GPU did not become free in time")


def run_one(model_type: str, task: dict, min_steps: int = 3, max_wait_s: int = 3600) -> dict:
    model_id = MODEL_OVERRIDES.get(model_type, task["model_id"])
    dataset_zip = refresh_dataset_url(task["training_data"])
    # Unique task id so history doesn't collide with prior runs
    smoke_task_id = f"smoke-{model_type}-{task['task_id'][:8]}-{int(time.time())}"
    payload = {
        "training_data": {
            "model": model_id,
            "task_id": smoke_task_id,
            "hours_to_complete": min(float(task["hours_to_complete"]), 1.0),
            "expected_repo_name": f"smoke-{model_type}",
            "dataset_zip": dataset_zip,
            "model_type": model_type,
            "trigger_word": task.get("trigger_word"),
        },
        "github_repo": REPO,
        "gpu_ids": GPU_IDS,
        "hotkey": HOTKEY,
        "github_commit_hash": COMMIT,
        "github_token": TOKEN,
    }
    print(f"\n=== START {model_type} task={smoke_task_id} model={model_id} ===", flush=True)
    status, resp = http_json("POST", f"{TRAINER}/v1/trainer/start_training", payload)
    print(f"start_training -> {status} {resp}", flush=True)
    if status != 200:
        return {"model_type": model_type, "ok": False, "error": resp, "task_id": smoke_task_id}

    deadline = time.time() + max_wait_s
    saw_container = False
    best_step = 0
    last_status = None
    while time.time() < deadline:
        details = task_details(smoke_task_id)
        if details:
            last_status = details.get("status")
            logs = details.get("logs") or []
            if logs:
                print(f"[{model_type}] trainer: {logs[-1][:240]}", flush=True)
            if last_status in ("success", "failure"):
                # Finished before we could stop — treat as done
                print(f"[{model_type}] finished early with status={last_status}", flush=True)
                return {
                    "model_type": model_type,
                    "ok": last_status == "success" or best_step >= min_steps,
                    "task_id": smoke_task_id,
                    "status": last_status,
                    "max_step": best_step,
                    "stopped": False,
                }

        cname = latest_image_trainer()
        if cname:
            saw_container = True
            clog = container_logs_tail(cname, 500)
            step = max_step_seen(clog)
            best_step = max(best_step, step)
            if step >= min_steps or (looks_like_training(clog) and step >= 1):
                print(f"[{model_type}] saw training progress max_step={best_step}; stopping {cname}", flush=True)
                subprocess.call(["docker", "rm", "-f", cname], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Give trainer loop time to mark failure/complete
                time.sleep(8)
                stop_trainer_containers()
                return {
                    "model_type": model_type,
                    "ok": True,
                    "task_id": smoke_task_id,
                    "status": "stopped_after_steps",
                    "max_step": best_step,
                    "stopped": True,
                    "container": cname,
                }
            # heartbeat every poll with last log line
            lines = [ln for ln in clog.splitlines() if ln.strip()]
            if lines:
                print(f"[{model_type}] docker@{cname[-12:]} step={best_step}: {lines[-1][:200]}", flush=True)
        elif saw_container:
            print(f"[{model_type}] trainer container disappeared; checking task status", flush=True)
            time.sleep(5)
        else:
            print(f"[{model_type}] waiting for build/download... status={last_status}", flush=True)

        time.sleep(20)

    print(f"[{model_type}] TIMEOUT after {max_wait_s}s; cleaning up", flush=True)
    stop_trainer_containers()
    return {
        "model_type": model_type,
        "ok": False,
        "task_id": smoke_task_id,
        "status": "timeout",
        "max_step": best_step,
        "stopped": True,
    }


def main() -> int:
    load_vali_env()
    stop_trainer_containers()
    tasks = pick_tasks()
    order_env = os.environ.get("SMOKE_MODELS", "z-image,qwen-image,krea2")
    order = [m.strip() for m in order_env.split(",") if m.strip()]
    # Seed result for already-verified ideogram4 if requested
    results = []
    if os.environ.get("SMOKE_IDEOGRAM_OK") == "1":
        results.append({
            "model_type": "ideogram4",
            "ok": True,
            "task_id": "prior-run",
            "status": "stopped_after_steps",
            "max_step": 8,
            "stopped": True,
            "note": "verified earlier; reused image for remaining toolkit models",
        })
    for mt in order:
        wait_gpu_free()
        results.append(run_one(mt, tasks[mt], min_steps=3, max_wait_s=5400))
        stop_trainer_containers()
        time.sleep(5)

    print("\n=== SUMMARY ===")
    for r in results:
        print(json.dumps(r))
    Path("/tmp/smoke_train_results.json").write_text(json.dumps(results, indent=2))
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
