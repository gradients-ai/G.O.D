#!/bin/bash

echo "Creating test training containers with various labels..."

# Clean up any existing test containers
docker rm -f test-image-trainer test-text-trainer test-dpo-trainer 2>/dev/null || true

# Test 1: Image training task
echo "Starting image trainer test container..."
docker run -d \
  --name test-image-trainer \
  --label task_id=image-task-$(date +%s) \
  --label hotkey=5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY \
  --label model=flux-dev \
  --label task_type=image \
  --label expected_repo=test-image-repo \
  --rm \
  alpine:latest \
  sh -c "for i in \$(seq 1 180); do echo \"\$(date '+%Y-%m-%d %H:%M:%S') INFO [IMAGE] Training step \$i/180 - loss: 0.\$((RANDOM % 999))\"; sleep 5; done"

# Test 2: Text training task
echo "Starting text trainer test container..."
docker run -d \
  --name test-text-trainer \
  --label task_id=text-task-$(date +%s) \
  --label hotkey=5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty \
  --label model=llama-3.1-8b \
  --label task_type=text \
  --label expected_repo=test-text-repo \
  --rm \
  alpine:latest \
  sh -c "for i in \$(seq 1 180); do echo \"\$(date '+%Y-%m-%d %H:%M:%S') INFO [TEXT] Training epoch \$i/180 - perplexity: \$((1000 - i*5))\"; sleep 5; done"

# Test 3: DPO training task
echo "Starting DPO trainer test container..."
docker run -d \
  --name test-dpo-trainer \
  --label task_id=dpo-task-$(date +%s) \
  --label hotkey=5GNJqTPyTXQr1QKjPLWVKPg9NTSjPW5GHsHgtGc9rDKNr7gn \
  --label model=mistral-7b \
  --label task_type=dpo \
  --label expected_repo=test-dpo-repo \
  --rm \
  alpine:latest \
  sh -c "for i in \$(seq 1 180); do echo \"\$(date '+%Y-%m-%d %H:%M:%S') INFO [DPO] Preference learning step \$i/180 - reward: 0.\$((RANDOM % 999))\"; sleep 5; done"

echo "========================================="
echo "Test containers created! Check Grafana for logs."
echo ""
echo "Try these queries in Grafana Explore:"
echo "  {job=\"docker-training-containers\"} - All logs"
echo "  {task_type=\"image\"} - Image training logs"
echo "  {task_type=\"text\"} - Text training logs"
echo "  {task_type=\"dpo\"} - DPO training logs"
echo "  {model=~\".+\"} - All logs grouped by model"
echo "========================================="