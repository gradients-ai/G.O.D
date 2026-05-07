#!/bin/bash
# Direct test: POST a TrainerProxyRequest with requested_datasets to the trainer.
# This bypasses the validator/orchestrator and tests the trainer-side flow:
#   1. Clone test repo
#   2. Download whitelisted dataset into cache volume
#   3. Start container with MINER_DATASETS_DIR/MINER_DATASETS env vars
#   4. Container verifies datasets exist and exits 0/1

TRAINER_IP="185.141.218.172"
TRAINER_PORT="8001"
TASK_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"

echo "=== Dataset Whitelist Direct Test ==="
echo "Trainer: ${TRAINER_IP}:${TRAINER_PORT}"
echo "Task ID: ${TASK_ID}"
echo ""

# TrainerProxyRequest with TrainRequestText (EnvTask uses text path)
# The training_data needs: model, task_id, hours_to_complete, dataset, dataset_type, file_format
PAYLOAD=$(cat <<EOF
{
    "training_data": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "task_id": "${TASK_ID}",
        "hours_to_complete": 0.5,
        "expected_repo_name": "dataset-whitelist-test",
        "dataset": "dummy",
        "dataset_type": {"environment_name": "goofspiel"},
        "file_format": "s3"
    },
    "github_repo": "https://github.com/wonderingwanderingwonders/god-dataset-whitelist-test",
    "gpu_ids": [0],
    "hotkey": "5Gwwwk2hiYjsk7jhj9grvMagD7g6nqQDfFe4Y9NtM3C3KjXy",
    "github_branch": null,
    "github_commit_hash": "main",
    "github_token": null,
    "requested_datasets": ["tasksource/Boardgame-QA"]
}
EOF
)

echo "Payload:"
echo "${PAYLOAD}" | python3 -m json.tool
echo ""

echo "Sending request..."
RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST \
    "http://${TRAINER_IP}:${TRAINER_PORT}/v1/trainer/start_training" \
    -H "Content-Type: application/json" \
    -d "${PAYLOAD}")

HTTP_CODE=$(echo "${RESPONSE}" | tail -1)
BODY=$(echo "${RESPONSE}" | head -n -1)

echo "HTTP ${HTTP_CODE}"
echo "${BODY}" | python3 -m json.tool 2>/dev/null || echo "${BODY}"
echo ""

if [ "${HTTP_CODE}" = "200" ]; then
    echo "Training request accepted. Monitor with:"
    echo "  curl http://${TRAINER_IP}:${TRAINER_PORT}/v1/trainer/${TASK_ID}/5Gwwwk2hiYjsk7jhj9grvMagD7g6nqQDfFe4Y9NtM3C3KjXy"
else
    echo "FAILED: HTTP ${HTTP_CODE}"
fi
