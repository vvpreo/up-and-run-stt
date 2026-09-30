#!/usr/bin/env bash
# Сборка Docker-образа up-and-run-stt.
#
#   ./build.sh                              # локальный CPU-образ up-and-run-stt:latest
#   ./build.sh myuser/up-and-run-stt        # образ с тегом для Docker Hub
#   ./build.sh myuser/up-and-run-stt --push # + docker push
#   ./build.sh --cuda13                     # GPU-образ под CUDA 13 (Turing+, драйвер 580+), up-and-run-stt-cuda13:latest
#   ./build.sh --cuda12                     # GPU-образ под CUDA 12.8 (Pascal+, драйвер 525+), up-and-run-stt-cuda12:latest
#   ./build.sh --cuda13 myuser/up-and-run-stt:cuda13 --push
# Про выбор между cuda12 и cuda13 — шапка Dockerfile.cuda.
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
BUILD_ARGS=()
if [[ "${1:-}" == "--cuda13" ]]; then
    DOCKERFILE=Dockerfile.cuda
    DEFAULT_IMAGE=up-and-run-stt-cuda13:latest
    shift
elif [[ "${1:-}" == "--cuda12" ]]; then
    # CUDA 12.8: для GPU, которых нет в CUDA 13 (Pascal/Volta, compute capability 6.x–7.0)
    DOCKERFILE=Dockerfile.cuda
    DEFAULT_IMAGE=up-and-run-stt-cuda12:latest
    BUILD_ARGS=(--build-arg CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
                --build-arg ORT_GPU_VERSION=1.29.0
                --build-arg ORT_INDEX_URL=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/)
    shift
fi

IMAGE="${1:-$DEFAULT_IMAGE}"
docker build -f "$DOCKERFILE" "${BUILD_ARGS[@]}" -t "$IMAGE" .
echo "Built: $IMAGE ($DOCKERFILE)"

if [[ "${2:-}" == "--push" ]]; then
    docker push "$IMAGE"
    echo "Pushed: $IMAGE"
fi
