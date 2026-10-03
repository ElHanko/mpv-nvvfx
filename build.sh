#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="mpv-nvvfx-build:debian13-cuda13.2"

echo '=== Build Docker image ==='

docker build \
  -t "$IMAGE" \
  -f "$ROOT/Dockerfile" \
  "$ROOT"

echo
echo '=== Build mpv inside Docker ==='

docker run \
  --rm \
  --init \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$ROOT:/src" \
  -w /src \
  "$IMAGE" \
  ./docker/build-mpv.sh
