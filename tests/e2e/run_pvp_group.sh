#!/bin/bash
set -e

NUM_GAMES="${1:-150}"
IMAGE="pvp-eval:test"
OUTPUT_DIR="/tmp/pvp-group-results"
mkdir -p "$OUTPUT_DIR"

BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"

cat > /tmp/pvp_group_config.json << EOF
{
    "mode": "group",
    "base_model": "$BASE_MODEL",
    "models": [
        {"repo": "gradients-io-tournaments/tournament-tourn_6ded1f069d76cb0e_20260427-7a724209-10e5-4a0b-8ed3-810c8bf53402-5C7vE26G", "hotkey": "5C7vE26G77n7CvUkAdgHKjT7scqfiNhWcaCg8WVyB8A57Mt1"},
        {"repo": "gradients-io-tournaments/tournament-tourn_6ded1f069d76cb0e_20260427-7a724209-10e5-4a0b-8ed3-810c8bf53402-5Ca32LwM", "hotkey": "5Ca32LwMizgx7VGESpkuRCYTsHZoVc9JX7gUML2enXSqpk3R"},
        {"repo": "gradients-io-tournaments/tournament-tourn_6ded1f069d76cb0e_20260427-7a724209-10e5-4a0b-8ed3-810c8bf53402-5CAMXmFr", "hotkey": "5CAMXmFrLxwdR3MykwK5vQrHHrvPrntu8Ak2T9wMGwvsgpmS"},
        {"repo": "gradients-io-tournaments/tournament-tourn_6ded1f069d76cb0e_20260427-7a724209-10e5-4a0b-8ed3-810c8bf53402-5CcPTwt8", "hotkey": "5CcPTwt8hcXTRPxZq4raohntP4MfduTFeLRjEG3gUPf71ECG"},
        {"repo": "gradients-io-tournaments/tournament-tourn_6ded1f069d76cb0e_20260427-7a724209-10e5-4a0b-8ed3-810c8bf53402-5CmAQ61V", "hotkey": "5CmAQ61VxoLhmvnCLB4UhahwKWsaSxRSzUbEJnZK5iPrMSSo"}
    ],
    "matchups": {"liars_dice": {"num_games": $NUM_GAMES}, "leduc_poker": {"num_games": $NUM_GAMES}},
    "seed": 42,
    "temperature": 0.0
}
EOF

echo "============================================================"
echo "PvP Group Round-Robin Test"
echo "============================================================"
echo "Base model: $BASE_MODEL"
echo "Models: 5 tournament LoRA adapters"
echo "Pairings: 10 (C(5,2))"
echo "Games per env: $NUM_GAMES (x2 for position swap)"
echo "Environments: liars_dice, leduc_poker"
echo "============================================================"
echo ""

RESULTS_FILE="$OUTPUT_DIR/group_results.json"
rm -f "$RESULTS_FILE"

START=$SECONDS
docker run --rm --gpus all \
    -v /tmp/pvp_group_config.json:/config/pvp_eval.json:ro \
    -v "$OUTPUT_DIR":/app/results \
    -e "PVP_RESULTS_PATH=/app/results/group_results.json" \
    --shm-size=16g \
    "$IMAGE" 2>&1
ELAPSED=$(( SECONDS - START ))

echo ""
echo "============================================================"
echo "Completed in ${ELAPSED}s"
echo "============================================================"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "FAIL: No results file"
    exit 1
fi

echo ""
echo "Results:"
cat "$RESULTS_FILE" | python3 -m json.tool

echo ""
echo "============================================================"
echo "Leaderboard"
echo "============================================================"
python3 -c "
import json

results = json.load(open('$RESULTS_FILE'))
hotkeys = results['hotkeys']
scores = {h: 0 for h in hotkeys}

for pair in results['pair_results']:
    for env, res in pair['results'].items():
        if res['model_a_wins'] > res['model_b_wins']:
            scores[pair['hotkey_a']] += 3
        elif res['model_b_wins'] > res['model_a_wins']:
            scores[pair['hotkey_b']] += 3
        else:
            scores[pair['hotkey_a']] += 1
            scores[pair['hotkey_b']] += 1

ranked = sorted(scores.items(), key=lambda x: -x[1])
print(f'  {\"Hotkey\":<50} Points')
print(f'  {\"-\"*50} ------')
for hotkey, pts in ranked:
    short = hotkey[:12] + '...' + hotkey[-4:]
    print(f'  {short:<50} {pts}')

print()
print(f'  Wall time: {results[\"metadata\"][\"wall_time_seconds\"]:.0f}s')
"

echo ""
echo "PASS"
exit 0
