#!/usr/bin/env bash
# Сборка Docker-образа gigaam-stt.
#
#   ./build.sh                       # локальный образ gigaam-stt:cpu
#   ./build.sh myuser/gigaam-stt     # образ с тегом для Docker Hub
#   ./build.sh myuser/gigaam-stt --push   # + docker push (тег latest)
#
# Публичный запуск собранного образа (веса скачиваются при первом старте
# в том gigaam-models, ~420 МБ):
#   docker run -d --name gigaam-stt -p 9007:9007 \
#     -v gigaam-models:/app/data \
#     -e AUTH_TOKEN=<секрет>  \
#     myuser/gigaam-stt
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${1:-gigaam-stt:latest}"
docker build -f Dockerfile -t "$IMAGE" .
echo "Built: $IMAGE"

if [[ "${2:-}" == "--push" ]]; then
    docker push "$IMAGE"
    echo "Pushed: $IMAGE"
fi
