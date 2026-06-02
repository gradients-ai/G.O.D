#!/bin/bash
#
# One-time setup for PvP E2E tests on a remote GPU machine.
#
# Usage:
#   ./tests/e2e/pvp_remote_setup.sh <host> [-i <keyfile>]
#
# What it does:
#   1. Syncs the repo
#   2. Builds the pvp-eval Docker image
#   3. Builds the trainer-downloader Docker image
#   4. Pre-downloads all test models via the downloader container
#      into a persistent Docker volume (same as production)
#
# After this completes, run:
#   ./tests/e2e/run_pvp_remote.sh <host> [-i <keyfile>]
#

set -euo pipefail

HOST=""
SSH_KEY=""
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i) SSH_KEY="$2"; shift 2 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) HOST="$1"; shift ;;
    esac
done

if [[ -z "$HOST" ]]; then
    echo "Usage: $0 <host> [-i <keyfile>]"
    exit 1
fi

[[ -n "$SSH_KEY" ]] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"

SSH="ssh $SSH_OPTS root@$HOST"
SCP="scp $SSH_OPTS"
REMOTE_DIR="/root/pvp-test"

# Models used by the test suite
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
LORA_BASE="NousResearch/Hermes-3-Llama-3.2-3B"
LORA_A="gradients-io-tournaments/tournament-tourn_6ded1f069d76cb0e_20260427-7a724209-10e5-4a0b-8ed3-810c8bf53402-5C7vE26G"
LORA_B="gradients-io-tournaments/tournament-tourn_6ded1f069d76cb0e_20260427-7a724209-10e5-4a0b-8ed3-810c8bf53402-5CmAQ61V"
LORA_C="gradients-io-tournaments/tournament-tourn_6ded1f069d76cb0e_20260427-7a724209-10e5-4a0b-8ed3-810c8bf53402-5Ca32LwM"

echo "============================================================"
echo "PvP Remote Setup"
echo "============================================================"
echo "Host: $HOST"
echo ""

# --- Step 1: Probe ---
echo ">>> Step 1: Probing remote..."
$SSH "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo '---'; docker --version"
echo ""

# --- Step 2: Sync repo ---
echo ">>> Step 2: Syncing repo..."
$SSH "rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR"
tar czf /tmp/pvp_repo_sync.tar.gz \
    --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='.pytest_cache' --exclude='node_modules' \
    --exclude='output' \
    -C "$(pwd)" .
$SCP /tmp/pvp_repo_sync.tar.gz "root@$HOST:/tmp/pvp_repo_sync.tar.gz"
$SSH "tar xzf /tmp/pvp_repo_sync.tar.gz -C $REMOTE_DIR && rm /tmp/pvp_repo_sync.tar.gz"
rm -f /tmp/pvp_repo_sync.tar.gz
echo "  Done"
echo ""

# --- Step 3: Build images ---
echo ">>> Step 3a: Building pvp-eval:test..."
$SSH "cd $REMOTE_DIR && docker build -f ops/docker/pvp-eval.dockerfile -t pvp-eval:test . 2>&1 | tail -3"
echo ""

echo ">>> Step 3b: Building trainer-downloader:test..."
$SSH "cd $REMOTE_DIR && docker build -f ops/docker/trainer-downloader.dockerfile -t trainer-downloader:test . 2>&1 | tail -3"
echo ""

# --- Step 4: Create cache volume ---
echo ">>> Step 4: Creating cache volume..."
$SSH "docker volume create cache 2>/dev/null || true; echo '  Volume ready'"
echo ""

# --- Step 5: Download models via downloader container ---
echo ">>> Step 5: Pre-downloading models via trainer-downloader..."

download_model() {
    local model="$1"
    local task_id="download-$(echo "$model" | tr '/' '-' | cut -c1-30)"

    echo "  Downloading: $model"
    $SSH "docker run --rm \
        -v cache:/cache \
        trainer-downloader:test \
        --task-id '$task_id' \
        --model '$model' \
        --task-type 'InstructTextTask' \
        --dataset 'dummy' \
        --file-format 'hf' \
        2>&1 | tail -3"
}

# Download base models via the production downloader (goes to /cache/models/)
download_model "$BASE_MODEL"
download_model "$LORA_BASE"

# SGLang uses HF hub cache, not the trainer cache layout.
# Pre-populate the HF cache so SGLang doesn't download at startup.
# We override the entrypoint to run a simple download script.
echo "  Caching all models in HF hub format for SGLang..."
$SSH "docker run --rm \
    -v cache:/cache \
    -e HF_HOME=/cache/hf_cache \
    --entrypoint python3 \
    pvp-eval:test -c '
from huggingface_hub import snapshot_download
models = [
    \"$BASE_MODEL\",
    \"$LORA_BASE\",
    \"$LORA_A\",
    \"$LORA_B\",
    \"$LORA_C\",
]
for m in models:
    print(f\"  {m}...\", flush=True)
    snapshot_download(m)
print(\"  All models cached.\", flush=True)
'"

echo ""
echo "============================================================"
echo "Setup complete. Run tests with:"
echo "  ./tests/e2e/run_pvp_remote.sh $HOST ${SSH_KEY:+-i $SSH_KEY}"
echo "============================================================"
