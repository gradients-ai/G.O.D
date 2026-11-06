#!/usr/bin/env python3
"""
Standalone PvP evaluation for tournament (continuation) models — any round, no DB.

Reconstructs each model's real serving topology the way production does, but takes
the lineage on the CLI instead of looking it up in the validator DB:

  - Round 1 (adapter trained directly on the foundation):
        --model-a <adapter>  --base-model <foundation>
  - Round N>1 (continuation adapter trained on the previous round's winner):
        --model-a <RN-adapter>  --base-model <foundation>  --start-a <R(N-1)-adapter>

The --start seed is the *immediate* prior-round repo the candidate trained on. The
full lineage down to the foundation is discovered automatically by walking each
adapter's adapter_config.json (validator.evaluation.pvp.materialize._resolve_chain),
so a single seed handles R2, R3, ... RN. Every hop is merged into the base; the
candidate is then served on top.

Serving decision per candidate (mirrors validator/evaluation/evaluators/environment.py):
  - LoRA, no added tokens   -> served natively:  --enable-lora --lora-paths
  - LoRA, added_tokens.json -> merged into the reconstructed base, served full weights
                               (SGLang native LoRA cannot apply a vocab-resizing adapter)
  - full weights            -> served as-is

Requires (same as run_pvp_eval.py):
  - 2x GPU (one per model), HuggingFace access, open_spiel, the validator-env stack
    (sglang, peft, accelerate). No validator DB, no PM2, no Config.

Usage:
    PYTHONPATH=. python tests/e2e/run_tournament_pvp_eval.py \
        --model-a org/tournament-...-5Ca32LwM \
        --model-b org/tournament-...-5GsVPezj \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --start-a org/previous-round-winner-A \
        --gpu-a 0 --gpu-b 1 \
        --time-budget-seconds 300 \
        --envs gin_rummy othello leduc_poker \
        --seed 1099583717
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
from pydantic import BaseModel

from core.constants.environments import EnvironmentName
from validator.evaluation.evaluators.environment import _download_lora_with_retry
from validator.evaluation.evaluators.environment import _merge_base_and_lora
from validator.evaluation.model_checks import check_for_lora
from validator.evaluation.model_checks import check_lora_has_added_tokens
from validator.evaluation.pvp.__main__ import _run_evaluation
from validator.evaluation.pvp.materialize import materialize_base_model
from validator.evaluation.pvp.models import PvPEvalConfig
from validator.evaluation.pvp.models import PvPMatchupConfig
from validator.evaluation.pvp.models import PvPModelSpec
from validator.evaluation.utils import configure_eval_logging


class PreparedSpec(BaseModel):
    """A resolved PvPModelSpec plus how it was prepared, for reporting."""

    spec: PvPModelSpec
    display_name: str  # the original candidate repo, even when we serve a merged path
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone tournament PvP evaluation (any round, no DB)")
    parser.add_argument("--model-a", required=True, help="Candidate A repo (round-N adapter or full weights)")
    parser.add_argument("--model-b", required=True, help="Candidate B repo (round-N adapter or full weights)")
    parser.add_argument("--base-model", required=True, help="Foundation model both lineages root at")
    parser.add_argument(
        "--start-a", default=None,
        help="Immediate prior-round repo model A trained on (lineage seed). Omit for round 1.",
    )
    parser.add_argument(
        "--start-b", default=None,
        help="Immediate prior-round repo model B trained on (lineage seed). Omit for round 1.",
    )
    parser.add_argument("--time-budget-seconds", type=float, default=300.0, help="Wall-clock budget per environment")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--gpu-a", type=int, default=0, help="GPU for model A")
    parser.add_argument("--gpu-b", type=int, default=1, help="GPU for model B")
    parser.add_argument(
        "--envs", nargs="+", default=["leduc_poker"],
        choices=[e.value for e in EnvironmentName],
        help="Environments to play",
    )
    parser.add_argument(
        "--prep-dir", default="/tmp/pvp_prep",
        help="Scratch dir for downloaded adapters and merged outputs",
    )
    parser.add_argument(
        "--skip-added-token-merge", action="store_true",
        help="Force native --enable-lora even if the candidate ships added_tokens.json (diverges from prod)",
    )
    return parser.parse_args()


def prepare(
    repo: str,
    start_repo: str | None,
    base_model: str,
    gpu_id: int,
    out_root: str,
    label: str,
    merge_added_tokens: bool,
) -> PreparedSpec:
    """Resolve serving topology for one candidate, doing any required merge up front."""
    is_lora = check_for_lora(repo, local_files_only=False)
    if not is_lora:
        if start_repo:
            print(f"  [{label}] WARNING: --start given but {repo} is not a LoRA adapter; ignoring lineage.")
        return PreparedSpec(
            spec=PvPModelSpec(repo=repo, original_model=base_model, base_chain=[], gpu_id=gpu_id),
            display_name=repo,
            note="full weights",
        )

    # base_chain holds only the immediate seed; _resolve_chain walks the rest to the foundation.
    base_chain = [start_repo] if start_repo else []
    has_added = merge_added_tokens and check_lora_has_added_tokens(repo, local_files_only=False)

    if not has_added:
        where = f"base reconstructed from seed {start_repo}" if base_chain else f"foundation {base_model}"
        return PreparedSpec(
            spec=PvPModelSpec(repo=repo, original_model=base_model, base_chain=base_chain, gpu_id=gpu_id),
            display_name=repo,
            note=f"native --enable-lora on {where}",
        )

    # Candidate resized the vocab: SGLang's native LoRA path can't apply it. Reconstruct the
    # base (merging every prior round), merge the candidate in, and serve the full weights.
    device = f"cuda:{gpu_id}"
    print(f"  [{label}] added_tokens.json present -> merging candidate into reconstructed base on {device}")
    base_path = materialize_base_model(base_model, base_chain, label=label, device=device)
    lora_dir = os.path.join(out_root, f"{label}_candidate_lora")
    _download_lora_with_retry(repo, lora_dir)
    merged_dir = os.path.join(out_root, f"{label}_merged")
    merged_path = _merge_base_and_lora(base_path, lora_dir, output_dir=merged_dir, device=device)
    # Free the merge's CUDA allocation before SGLang subprocesses claim the same GPU.
    gc.collect()
    torch.cuda.empty_cache()
    return PreparedSpec(
        # original_model stays the foundation so _prepare_model's tool-call-parser fallback resolves;
        # repo is the local merged dir, which check_for_lora treats as full weights.
        spec=PvPModelSpec(repo=merged_path, original_model=base_model, base_chain=[], gpu_id=gpu_id),
        display_name=repo,
        note=f"added_tokens -> merged full weights at {merged_path}",
    )


def build_config(args: argparse.Namespace, prep_a: PreparedSpec, prep_b: PreparedSpec) -> PvPEvalConfig:
    matchups = {
        EnvironmentName(env): PvPMatchupConfig(time_budget_seconds=args.time_budget_seconds)
        for env in args.envs
    }
    return PvPEvalConfig(
        model_a=prep_a.spec,
        model_b=prep_b.spec,
        matchups=matchups,
        seed=args.seed,
        temperature=args.temperature,
    )


def validate_results(config: PvPEvalConfig, results_json: dict) -> list[str]:
    """Return list of validation errors (empty = pass)."""
    errors: list[str] = []
    for env_name in config.matchups:
        env_key = env_name.value
        env_result = results_json["results"].get(env_key)
        if env_result is None:
            errors.append(f"Missing results for {env_key}")
            continue
        if env_result["total_games"] <= 0:
            errors.append(f"{env_key}: expected at least one completed game pair")
        if env_result["total_games"] % 2:
            errors.append(f"{env_key}: total_games={env_result['total_games']} is not position-balanced")
        accounting = env_result["model_a_wins"] + env_result["model_b_wins"] + env_result["draws"]
        if accounting != env_result["total_games"]:
            errors.append(f"{env_key}: wins+losses+draws={accounting} != total_games={env_result['total_games']}")
    if results_json.get("metadata", {}).get("wall_time_seconds", 0) <= 0:
        errors.append("wall_time_seconds should be positive")
    return errors


def main() -> int:
    args = parse_args()
    configure_eval_logging()
    os.makedirs(args.prep_dir, exist_ok=True)

    print("=" * 72)
    print("Tournament PvP Evaluation (standalone, no DB)")
    print("=" * 72)
    print(f"Foundation : {args.base_model}")
    print(f"Model A    : {args.model_a} (GPU {args.gpu_a}, seed {args.start_a or '-'})")
    print(f"Model B    : {args.model_b} (GPU {args.gpu_b}, seed {args.start_b or '-'})")
    print(f"Envs       : {args.envs}")
    print(f"Budget/env : {args.time_budget_seconds:.0f}s   Seed: {args.seed}   Temp: {args.temperature}")
    print("=" * 72)

    merge_added = not args.skip_added_token_merge
    print("\nPreparing models...")
    prep_a = prepare(args.model_a, args.start_a, args.base_model, args.gpu_a, args.prep_dir, "a", merge_added)
    prep_b = prepare(args.model_b, args.start_b, args.base_model, args.gpu_b, args.prep_dir, "b", merge_added)
    print(f"  A: {prep_a.display_name}\n     -> {prep_a.note}")
    print(f"  B: {prep_b.display_name}\n     -> {prep_b.note}")

    config = build_config(args, prep_a, prep_b)

    start = time.time()
    try:
        results = _run_evaluation(config)
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    elapsed = time.time() - start

    results_json = json.loads(results.model_dump_json())

    print("\n" + "=" * 72)
    print("Results")
    print("=" * 72)
    print(json.dumps(results_json, indent=2))

    print("\n" + "=" * 72)
    print("Validation")
    print("=" * 72)
    errors = validate_results(config, results_json)
    if errors:
        for error in errors:
            print(f"  FAIL: {error}")
        return 1
    print("  All checks passed.")
    print(f"\n  Wall time: {elapsed:.1f}s")

    print("\n  Head-to-head:")
    print(f"    A = {prep_a.display_name}")
    print(f"    B = {prep_b.display_name}")
    for env_key, env_result in results_json["results"].items():
        total = env_result["total_games"]
        a_pct = env_result["model_a_wins"] / total * 100 if total else 0
        b_pct = env_result["model_b_wins"] / total * 100 if total else 0
        d_pct = env_result["draws"] / total * 100 if total else 0
        print(f"    {env_key}: A={a_pct:.0f}% B={b_pct:.0f}% D={d_pct:.0f}%  (n={total})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
