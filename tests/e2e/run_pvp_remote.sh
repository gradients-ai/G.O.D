#!/bin/bash
#
# PvP E2E test suite — runs on a pre-setup remote GPU machine.
#
# Prerequisites:
#   ./tests/e2e/pvp_remote_setup.sh <host> [-i <keyfile>]
#
# Usage:
#   ./tests/e2e/run_pvp_remote.sh <host> [-i <keyfile>] [--time-budget-seconds N]
#
# Tests:
#   1. Symmetric base-vs-base  (win rates should be ~50/50)
#   2. LoRA-vs-base            (trained adapter vs untrained base)
#   3. LoRA-vs-LoRA            (two different adapters)
#

set -euo pipefail

HOST=""
SSH_KEY=""
TIME_BUDGET_SECONDS=300
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i) SSH_KEY="$2"; shift 2 ;;
        --time-budget-seconds) TIME_BUDGET_SECONDS="$2"; shift 2 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) HOST="$1"; shift ;;
    esac
done

if [[ -z "$HOST" ]]; then
    echo "Usage: $0 <host> [-i <keyfile>] [--time-budget-seconds N]"
    exit 1
fi

[[ -n "$SSH_KEY" ]] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"

SSH="ssh $SSH_OPTS root@$HOST"
SCP="scp $SSH_OPTS"
RESULTS_DIR="/tmp/pvp-e2e-results"

# --- Models ---
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
LORA_BASE="NousResearch/Hermes-3-Llama-3.2-3B"
LORA_A="gradients-io-tournaments/tournament-tourn_6ded1f069d76cb0e_20260427-7a724209-10e5-4a0b-8ed3-810c8bf53402-5C7vE26G"
LORA_B="gradients-io-tournaments/tournament-tourn_6ded1f069d76cb0e_20260427-7a724209-10e5-4a0b-8ed3-810c8bf53402-5CmAQ61V"

echo "============================================================"
echo "PvP Remote E2E Test Suite"
echo "============================================================"
echo "Host:       $HOST"
echo "Budget/env: ${TIME_BUDGET_SECONDS}s"
echo "============================================================"
echo ""

PASSED=0
FAILED=0
TOTAL=3

# --- Helper: run a test ---
run_pvp_test() {
    local test_name="$1"
    local config_json="$2"

    echo ""
    echo "------------------------------------------------------------"
    echo "TEST: $test_name"
    echo "------------------------------------------------------------"

    $SSH "mkdir -p $RESULTS_DIR && cat > /tmp/pvp_config.json << 'CONFIGEOF'
$config_json
CONFIGEOF
docker run --rm --gpus all \
    -v /tmp/pvp_config.json:/config/pvp_eval.json:ro \
    -v $RESULTS_DIR:/app/results \
    -v cache:/cache \
    -e HF_HOME=/cache/hf_cache \
    -e PVP_RESULTS_PATH=/app/results/${test_name}.json \
    --shm-size=16g \
    pvp-eval:test 2>&1 | tail -20"

    local local_results="/tmp/pvp_remote_${test_name}.json"
    $SCP "root@$HOST:$RESULTS_DIR/${test_name}.json" "$local_results" 2>/dev/null || true

    if [[ ! -f "$local_results" ]]; then
        echo "  FAIL: No results file produced"
        FAILED=$((FAILED + 1))
        return 1
    fi

    echo "  Results received. Validating..."
    return 0
}

# --- Helper: validate results ---
validate_results() {
    local test_name="$1"
    local local_results="/tmp/pvp_remote_${test_name}.json"

    python3 - "$local_results" << 'PYEOF'
import json, sys

results_path = sys.argv[1]
results = json.load(open(results_path))
errors = []

if "results" not in results:
    errors.append("Missing pair-mode results")
else:
    for env, res in results["results"].items():
        total = res["total_games"]
        if total <= 0:
            errors.append(f"{env}: expected at least one completed game pair")
        if total % 2:
            errors.append(f"{env}: total_games={total} is not position-balanced")
        accounting = res["model_a_wins"] + res["model_b_wins"] + res["draws"]
        if accounting != total:
            errors.append(f"{env}: wins+draws={accounting} != total={total}")

wall_time = results.get("metadata", {}).get("wall_time_seconds", 0)
if wall_time <= 0:
    errors.append("wall_time_seconds should be positive")

if errors:
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)

# Print summary
print(f"  Model A: {results.get('model_a', 'N/A')}")
print(f"  Model B: {results.get('model_b', 'N/A')}")
for env, res in results["results"].items():
    t = res["total_games"]
    a = res["model_a_wins"] / t * 100 if t else 0
    b = res["model_b_wins"] / t * 100 if t else 0
    d = res["draws"] / t * 100 if t else 0
    print(f"  {env}: A={a:.0f}% B={b:.0f}% D={d:.0f}% ({t} games)")

print(f"  Wall time: {wall_time:.0f}s")
print(f"  PASS")
PYEOF
}

# =============================================================
# Test 1: Symmetric base-vs-base
# =============================================================
echo "============================================================"
echo "Test 1/$TOTAL: Symmetric base-vs-base"
echo "============================================================"

if run_pvp_test "symmetric" "{
    \"model_a\": {\"repo\": \"$BASE_MODEL\", \"original_model\": \"$BASE_MODEL\"},
    \"model_b\": {\"repo\": \"$BASE_MODEL\", \"original_model\": \"$BASE_MODEL\"},
    \"matchups\": {\"liars_dice\": {\"time_budget_seconds\": $TIME_BUDGET_SECONDS}, \"leduc_poker\": {\"time_budget_seconds\": $TIME_BUDGET_SECONDS}},
    \"seed\": 42, \"temperature\": 0.0
}" && validate_results "symmetric"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# =============================================================
# Test 2: LoRA vs base
# =============================================================
echo ""
echo "============================================================"
echo "Test 2/$TOTAL: LoRA adapter vs base model"
echo "============================================================"

if run_pvp_test "lora_vs_base" "{
    \"model_a\": {\"repo\": \"$LORA_A\", \"original_model\": \"$LORA_BASE\"},
    \"model_b\": {\"repo\": \"$LORA_BASE\", \"original_model\": \"$LORA_BASE\"},
    \"matchups\": {\"liars_dice\": {\"time_budget_seconds\": $TIME_BUDGET_SECONDS}, \"leduc_poker\": {\"time_budget_seconds\": $TIME_BUDGET_SECONDS}},
    \"seed\": 42, \"temperature\": 0.0
}" && validate_results "lora_vs_base"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# =============================================================
# Test 3: LoRA vs LoRA
# =============================================================
echo ""
echo "============================================================"
echo "Test 3/$TOTAL: LoRA vs LoRA"
echo "============================================================"

if run_pvp_test "lora_vs_lora" "{
    \"model_a\": {\"repo\": \"$LORA_A\", \"original_model\": \"$LORA_BASE\"},
    \"model_b\": {\"repo\": \"$LORA_B\", \"original_model\": \"$LORA_BASE\"},
    \"matchups\": {\"liars_dice\": {\"time_budget_seconds\": $TIME_BUDGET_SECONDS}, \"leduc_poker\": {\"time_budget_seconds\": $TIME_BUDGET_SECONDS}},
    \"seed\": 42, \"temperature\": 0.0
}" && validate_results "lora_vs_lora"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# =============================================================
# Summary
# =============================================================
echo ""
echo "============================================================"
echo "SUITE RESULTS: $PASSED/$TOTAL passed, $FAILED failed"
echo "============================================================"

# Symmetric win rate sanity
if [[ -f /tmp/pvp_remote_symmetric.json ]]; then
    echo ""
    echo "--- Symmetric Win Rate Check ---"
    python3 -c "
import json
r = json.load(open('/tmp/pvp_remote_symmetric.json'))
for env, res in r['results'].items():
    t = res['total_games']
    if t == 0: continue
    a = res['model_a_wins'] / t * 100
    b = res['model_b_wins'] / t * 100
    skew = abs(a - b)
    tag = 'OK' if skew < 80 else 'WARN: extreme skew'
    print(f'  {env}: A={a:.0f}% B={b:.0f}% skew={skew:.0f}pp [{tag}]')
"
fi

exit $FAILED
