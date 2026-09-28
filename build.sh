#!/usr/bin/env bash
# Сборка Docker-образа up-and-run-stt.
#
#   ./build.sh                              # локальный CPU-образ up-and-run-stt:latest
#   ./build.sh myuser/up-and-run-stt        # образ с тегом для Docker Hub
#   ./build.sh myuser/up-and-run-stt --push # + docker push
#   ./build.sh --cuda                       # GPU-образ up-and-run-stt-cuda:latest (Dockerfile.cuda)
#   ./build.sh --cuda myuser/up-and-run-stt:cuda --push
#
# Публичный запуск собранного образа (веса скачиваются при первом старте
# в том gigaam-models, ~845 МБ на модель):
#   docker run -d --name up-and-run-stt -p 9007:9007 \
#     -v gigaam-models:/app/data \
#     -e AUTH_TOKEN=<секрет>  \
#     myuser/up-and-run-stt
# GPU-вариант — то же плюс `--gpus all` (см. README, раздел GPU).
set -euo pipefail
cd "$(dirname "$0")"

DOCKERFILE=Dockerfile
DEFAULT_IMAGE=up-and-run-stt:latest
if [[ "${1:-}" == "--cuda" ]]; then
    DOCKERFILE=Dockerfile.cuda
    DEFAULT_IMAGE=up-and-run-stt-cuda:latest
    shift
fi

IMAGE="${1:-$DEFAULT_IMAGE}"
docker build -f "$DOCKERFILE" -t "$IMAGE" .
echo "Built: $IMAGE ($DOCKERFILE)"

if [[ "${2:-}" == "--push" ]]; then
    docker push "$IMAGE"
    echo "Pushed: $IMAGE"
fi
