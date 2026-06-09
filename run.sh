#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== EGT309 Pipeline ==="
echo "Building Docker image..."
docker build -t egt309 -f devel/Dockerfile .

echo ""
echo "Running pipeline (clean + train + evaluate)..."
docker run --rm egt309

echo ""
echo "=== Pipeline complete ==="
