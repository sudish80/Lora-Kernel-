#!/usr/bin/env bash
# CI script: build and verify Docker image
set -euo pipefail

IMAGE="lora-kernel:ci"
TEST_INPUT='{"prompt": "Hello", "max_tokens": 10}'

echo "[CI] Building Docker image..."
docker build -t "$IMAGE" .
echo "[CI] Docker build succeeded"

echo "[CI] Running container with test input..."
OUTPUT=$(echo "$TEST_INPUT" | docker run --rm -i "$IMAGE" 2>&1)
echo "[CI] Container output: $OUTPUT"

echo "[CI] PASSED: Docker image built and ran successfully"
exit 0
