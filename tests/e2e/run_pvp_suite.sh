#!/bin/bash
set -e

TIME_BUDGET_SECONDS=900
IMAGE="pvp-eval:test"
OUTPUT_DIR="/tmp/pvp-suite-results"
mkdir -p "$OUTPUT_DIR"

BASE_3B="Qwen/Qwen2.5-3B-Instruct"
LORA_A="gradients-io-tournaments/tournament-tourn_6ded1f069d76cb0e_20260427-7a724209-10e5-4a0b-8ed3-810c8bf53402-5C7vE26G"
LORA_B="gradients-io-tournaments/tournament-tourn_6ded1f069d76cb0e_20260427-7a724209-10e5-4a0b-8ed3-810c8bf53402-5CmAQ61V"

run_test() {
    local test_name="$1"
    local config_file="$2"
    local results_file="$OUTPUT_DIR/${test_name}.json"

    echo ""
    echo "============================================================"
    echo "TEST: $test_name"
    echo "============================================================"
    echo "Config: $(cat "$config_file")"
    echo ""

    rm -f "$results_file"
    local start=$SECONDS

    docker run --rm --gpus all \
        -v "$config_file":/config/pvp_eval.json:ro \
        -v "$OUTPUT_DIR":/app/results \
        -e "PVP_RESULTS_PATH=/app/results/${test_name}.json" \
        --shm-size=16g \
        "$IMAGE" 2>&1

    local elapsed=$(( SECONDS - start ))
    echo ""
    echo "Completed in ${elapsed}s"

    if [ -f "$results_file" ]; then
        echo "Results:"
        cat "$results_file" | python3 -m json.tool
        echo ""
        echo "PASS: $test_name"
    else
        echo "FAIL: $test_name — no results file"
        return 1
    fi
}

# --- Test 1: Symmetric (3B base vs 3B base) ---
cat > /tmp/pvp_suite_test1.json << EOF
{
    "model_a": {"repo": "$BASE_3B", "original_model": "$BASE_3B"},
    "model_b": {"repo": "$BASE_3B", "original_model": "$BASE_3B"},
    "matchups": {"liars_dice": {"time_budget_seconds": $TIME_BUDGET_SECONDS}, "leduc_poker": {"time_budget_seconds": $TIME_BUDGET_SECONDS}},
    "seed": 42,
    "temperature": 0.0
}
EOF

# --- Test 2: 3B LoRA vs 3B base ---
cat > /tmp/pvp_suite_test2.json << EOF
{
    "model_a": {"repo": "$LORA_B", "original_model": "$BASE_3B"},
    "model_b": {"repo": "$BASE_3B", "original_model": "$BASE_3B"},
    "matchups": {"liars_dice": {"time_budget_seconds": $TIME_BUDGET_SECONDS}, "leduc_poker": {"time_budget_seconds": $TIME_BUDGET_SECONDS}},
    "seed": 42,
    "temperature": 0.0
}
EOF

# --- Test 3: 3B LoRA vs 3B LoRA ---
cat > /tmp/pvp_suite_test3.json << EOF
{
    "model_a": {"repo": "$LORA_A", "original_model": "$BASE_3B"},
    "model_b": {"repo": "$LORA_B", "original_model": "$BASE_3B"},
    "matchups": {"liars_dice": {"time_budget_seconds": $TIME_BUDGET_SECONDS}, "leduc_poker": {"time_budget_seconds": $TIME_BUDGET_SECONDS}},
    "seed": 42,
    "temperature": 0.0
}
EOF

echo ""
echo "============================================================"
echo "PvP Evaluation Test Suite"
echo "============================================================"
echo "Time budget per env: ${TIME_BUDGET_SECONDS}s"
echo "Environments: liars_dice, leduc_poker"
echo ""

PASSED=0
FAILED=0

for i in 1 2 3; do
    test_names=("symmetric_base_vs_base" "lora_vs_base" "lora_vs_lora")
    name="${test_names[$((i-1))]}"
    if run_test "$name" "/tmp/pvp_suite_test${i}.json"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "SUITE RESULTS: $PASSED passed, $FAILED failed"
echo "============================================================"

# Print summary table
echo ""
echo "Summary:"
for name in symmetric_base_vs_base lora_vs_base lora_vs_lora; do
    f="$OUTPUT_DIR/${name}.json"
    if [ -f "$f" ]; then
        echo "  $name:"
        python3 -c "
import json, sys
r = json.load(open('$f'))
for env, res in r['results'].items():
    t = res['total_games']
    a = res['model_a_wins'] / t * 100 if t else 0
    b = res['model_b_wins'] / t * 100 if t else 0
    d = res['draws'] / t * 100 if t else 0
    print(f'    {env}: A={a:.0f}% B={b:.0f}% D={d:.0f}% ({t} games)')
print(f'    wall_time: {r[\"metadata\"][\"wall_time_seconds\"]:.0f}s')
"
    fi
done

exit $FAILED
