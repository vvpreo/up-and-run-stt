#!/usr/bin/env bash
# Сборка Docker-образа up-and-run-stt.
#
#   ./build.sh                       # локальный образ up-and-run-stt:cpu
#   ./build.sh myuser/up-and-run-stt     # образ с тегом для Docker Hub
#   ./build.sh myuser/up-and-run-stt --push   # + docker push (тег latest)
#
# Публичный запуск собранного образа (веса скачиваются при первом старте
# в том gigaam-models, ~420 МБ):
#   docker run -d --name up-and-run-stt -p 9007:9007 \
#     -v gigaam-models:/app/data \
#     -e AUTH_TOKEN=<секрет>  \
#     myuser/up-and-run-stt
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${1:-up-and-run-stt:latest}"
docker build -f Dockerfile -t "$IMAGE" .
echo "Built: $IMAGE"

if [[ "${2:-}" == "--push" ]]; then
    docker push "$IMAGE"
    echo "Pushed: $IMAGE"
fi
