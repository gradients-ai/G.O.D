#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
OUTPUT_DIR="$REPO_DIR/output/sft_dataset"

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before running}"

# Activate venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -q anthropic httpx tqdm
fi

# Check env server
if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "ERROR: mcts-api not running on :8000"
    echo "Start it with: sudo docker run -d --name mcts-api -p 8000:8000 diagonalge/mcts-api:latest"
    exit 1
fi

echo "Starting dataset generation: 400 liars_dice + 400 leduc_poker + 300 gin_rummy (80% Haiku, 20% Sonnet)"
echo "Output: $OUTPUT_DIR"
echo "Checkpoint saves after every episode — safe to Ctrl+C and resume."
echo ""

PYTHONPATH="$REPO_DIR" "$VENV_DIR/bin/python" -m scripts.sft_dataset_gen.generate \
    --concurrency 4 \
    --haiku-ratio 0.8 \
    --output-dir "$OUTPUT_DIR" \
    "$@"
