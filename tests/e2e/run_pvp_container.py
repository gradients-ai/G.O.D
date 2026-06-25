#!/usr/bin/env python3
"""
Container-level E2E test for PvP evaluation.

Builds the pvp-eval Docker image, runs it with a config, and validates
the output JSON. This is the final integration gate — tests the actual
container that will run in production.

Requires:
  - Docker with GPU support (nvidia-container-toolkit)
  - 2x GPU
  - Internet access (SGLang downloads models from HuggingFace)

Usage:
    python tests/e2e/run_pvp_container.py
    python tests/e2e/run_pvp_container.py --lora-test
    python tests/e2e/run_pvp_container.py --full
    python tests/e2e/run_pvp_container.py --skip-build
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


_IMAGE_NAME = "pvp-eval:test"
_DOCKERFILE = "ops/docker/pvp-eval.dockerfile"
_RESULTS_CONTAINER_PATH = "/app/pvp_results.json"

_BASE_MODEL = "NousResearch/Hermes-3-Llama-3.2-3B"
_LORA_ADAPTER = (
    "gradients-io-tournaments/tournament-tourn_cda90edddb13aba6_20260504"
    "-e1a8f105-cceb-47d6-9848-94663b9f0fa0-5GKSa6y1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Container E2E test for PvP evaluation")
    parser.add_argument("--lora-test", action="store_true", help="Test LoRA adapter vs base model")
    parser.add_argument("--full", action="store_true", help="Run both base-vs-base and lora-vs-base")
    parser.add_argument("--skip-build", action="store_true", help="Skip docker build (use existing image)")
    parser.add_argument("--time-budget-seconds", type=float, default=60.0, help="Wall-clock budget per environment")
    parser.add_argument("--envs", nargs="+", default=["liars_dice", "leduc_poker"], help="Environments to test")
    parser.add_argument("--gpu-a", type=int, default=0, help="GPU for model A")
    parser.add_argument("--gpu-b", type=int, default=1, help="GPU for model B")
    return parser.parse_args()


def build_image() -> bool:
    """Build the pvp-eval Docker image. Returns True on success."""
    print("Building Docker image...")
    result = subprocess.run(
        ["docker", "build", "-f", _DOCKERFILE, "-t", _IMAGE_NAME, "."],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr[-2000:]}")
        return False
    print("Build succeeded.")
    return True


def build_config(
    model_a: str,
    model_b: str,
    base_model: str,
    time_budget_seconds: float,
    envs: list[str],
    gpu_a: int,
    gpu_b: int,
) -> dict:
    """Build a PvP eval config dict."""
    return {
        "model_a": {"repo": model_a, "original_model": base_model, "gpu_id": gpu_a},
        "model_b": {"repo": model_b, "original_model": base_model, "gpu_id": gpu_b},
        "matchups": {env: {"time_budget_seconds": time_budget_seconds} for env in envs},
        "seed": 42,
        "temperature": 0.0,
    }


def run_container(config: dict, gpu_a: int, gpu_b: int) -> tuple[int, dict | None]:
    """Run the pvp-eval container and return (exit_code, results_dict or None)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "pvp_eval.json"
        config_path.write_text(json.dumps(config))

        gpu_devices = f"{gpu_a},{gpu_b}"
        cmd = [
            "docker", "run", "--rm",
            "--gpus", f'"device={gpu_devices}"',
            "-v", f"{config_path}:/config/pvp_eval.json:ro",
            "-v", f"{tmpdir}:/output",
            "-e", f"NVIDIA_VISIBLE_DEVICES={gpu_devices}",
            "--shm-size=16g",
            _IMAGE_NAME,
        ]

        print(f"Running container with GPUs {gpu_devices}...")
        print(f"Config: {json.dumps(config, indent=2)}")

        start = time.time()
        result = subprocess.run(
            " ".join(cmd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        elapsed = time.time() - start

        print(f"Container exited with code {result.returncode} in {elapsed:.0f}s")

        if result.returncode != 0:
            print(f"STDERR (last 3000 chars):\n{result.stderr[-3000:]}")
            print(f"STDOUT (last 3000 chars):\n{result.stdout[-3000:]}")
            return result.returncode, None

        # The container writes results to /app/pvp_results.json inside the container.
        # We need to copy it out. Since we used --rm, let's check if the entrypoint
        # also wrote to a mounted path. If not, we need to adjust.
        # Actually, the results path is fixed at /app/pvp_results.json (PVP_RESULTS_PATH).
        # We need to mount a volume there or copy before container exits.
        # Let's re-run with the results path mounted.

        # For now, check stdout for the results JSON (the container logs it)
        # Actually, let's parse from the container output.
        # Better approach: override PVP_RESULTS_PATH to write to /output/
        return result.returncode, None


def run_container_v2(config: dict, gpu_a: int, gpu_b: int) -> tuple[int, dict | None]:
    """Run the pvp-eval container with results written to a mounted volume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "pvp_eval.json"
        config_path.write_text(json.dumps(config))
        results_host_path = Path(tmpdir) / "pvp_results.json"

        gpu_devices = f"{gpu_a},{gpu_b}"
        cmd = [
            "docker", "run", "--rm",
            f"--gpus=device={gpu_devices}",
            "-v", f"{config_path}:/config/pvp_eval.json:ro",
            "-v", f"{results_host_path.parent}:/app/results",
            "-e", "PVP_RESULTS_PATH=/app/results/pvp_results.json",
            "--shm-size=16g",
            _IMAGE_NAME,
        ]

        print(f"Running container with GPUs {gpu_devices}...")
        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        elapsed = time.time() - start
        print(f"Container exited with code {result.returncode} in {elapsed:.0f}s")

        if result.returncode != 0:
            print(f"STDERR (last 3000 chars):\n{result.stderr[-3000:]}")
            return result.returncode, None

        results_file = Path(tmpdir) / "pvp_results.json"
        if not results_file.exists():
            print("ERROR: Results file not written by container")
            print(f"STDOUT:\n{result.stdout[-3000:]}")
            return 1, None

        results = json.loads(results_file.read_text())
        return 0, results


def validate_results(config: dict, results: dict) -> list[str]:
    """Validate result structure and accounting invariants."""
    errors: list[str] = []

    if results.get("model_a") != config["model_a"]["repo"]:
        errors.append(f"model_a mismatch: {results.get('model_a')} != {config['model_a']['repo']}")
    if results.get("model_b") != config["model_b"]["repo"]:
        errors.append(f"model_b mismatch: {results.get('model_b')} != {config['model_b']['repo']}")

    for env_name, matchup in config["matchups"].items():
        if env_name not in results.get("results", {}):
            errors.append(f"Missing results for {env_name}")
            continue

        env_result = results["results"][env_name]
        if env_result["total_games"] <= 0:
            errors.append(f"{env_name}: expected at least one completed game pair")
        if env_result["total_games"] % 2:
            errors.append(f"{env_name}: total_games={env_result['total_games']} is not position-balanced")

        accounting = env_result["model_a_wins"] + env_result["model_b_wins"] + env_result["draws"]
        if accounting != env_result["total_games"]:
            errors.append(f"{env_name}: wins+losses+draws={accounting} != total={env_result['total_games']}")

    metadata = results.get("metadata", {})
    if metadata.get("wall_time_seconds", 0) <= 0:
        errors.append("wall_time_seconds should be positive")
    if not metadata.get("position_swapped", False):
        errors.append("position_swapped should be true")

    return errors


def run_test(args: argparse.Namespace, is_lora: bool) -> int:
    """Run a single container test. Returns 0 on success."""
    label = "lora-vs-base" if is_lora else "base-vs-base"
    print(f"\n{'=' * 60}")
    print(f"Container Test: {label}")
    print(f"{'=' * 60}")

    if is_lora:
        config = build_config(
            _LORA_ADAPTER,
            _BASE_MODEL,
            _BASE_MODEL,
            args.time_budget_seconds,
            args.envs,
            args.gpu_a,
            args.gpu_b,
        )
    else:
        config = build_config(
            _BASE_MODEL,
            _BASE_MODEL,
            _BASE_MODEL,
            args.time_budget_seconds,
            args.envs,
            args.gpu_a,
            args.gpu_b,
        )

    exit_code, results = run_container_v2(config, args.gpu_a, args.gpu_b)
    if exit_code != 0 or results is None:
        print(f"FAIL: Container exited with code {exit_code}")
        return 1

    print(f"\nResults:\n{json.dumps(results, indent=2)}")

    errors = validate_results(config, results)
    if errors:
        for error in errors:
            print(f"  FAIL: {error}")
        return 1

    print("  All checks passed.")

    for env_name, env_result in results.get("results", {}).items():
        total = env_result["total_games"]
        if total > 0:
            a_pct = env_result["model_a_wins"] / total * 100
            b_pct = env_result["model_b_wins"] / total * 100
            d_pct = env_result["draws"] / total * 100
            print(f"  {env_name}: A={a_pct:.0f}% B={b_pct:.0f}% D={d_pct:.0f}%")

    return 0


def main() -> int:
    args = parse_args()

    if not args.skip_build:
        if not build_image():
            return 1

    if args.full:
        results = []
        results.append(("base-vs-base", run_test(args, is_lora=False)))
        results.append(("lora-vs-base", run_test(args, is_lora=True)))

        print(f"\n{'=' * 60}")
        print("Overall Results")
        print(f"{'=' * 60}")
        all_passed = True
        for name, code in results:
            status = "PASS" if code == 0 else "FAIL"
            print(f"  {name}: {status}")
            if code != 0:
                all_passed = False
        return 0 if all_passed else 1

    return run_test(args, is_lora=args.lora_test)


if __name__ == "__main__":
    sys.exit(main())
