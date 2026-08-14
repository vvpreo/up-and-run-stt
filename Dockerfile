# Dockerfile — основной образ up-and-run-stt: инференс на ONNX Runtime, без PyTorch.
# ONNX-веса НЕ запечены в образ: скачиваются при первом старте с HF Hub
# (GIGAAM_ONNX_BASE_URL, по умолчанию vvpreo/gigaam-v3-onnx) в том gigaam-models.
# Токенизаторы e2e-моделей — с CDN SberDevices.
# PyTorch-вариант для конвертации весов и сверки — Dockerfile.torch.
#
# Сборка многостадийная: зависимости ставит uv в стадии builder, в финальный
# образ переезжает только готовый venv — ни uv, ни кэши колёс в него не попадают.
# Собирается под linux/amd64 и linux/arm64 (см. ARG TARGETARCH ниже).

# ============================ builder =================================
FROM python:3.11-slim-bookworm AS builder

# uv запинен по версии: он определяет, как читается uv.lock, и обновляться
# молча не должен. Копируется из официального образа — ставить его pip'ом
# в сборочную стадию незачем.
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Только манифест и лок: правка кода приложения не должна инвалидировать
# слой с зависимостями.
COPY pyproject.toml uv.lock ./

# --frozen: лок не пересчитывается, сборка падает, если он разошёлся с
# pyproject — то есть образ ставит ровно те версии, что зафиксированы в git.
# --no-dev: pytest/requests в рантайм-образ не нужны.
RUN uv sync --frozen --no-dev --no-install-project

# ============================ runtime =================================
FROM python:3.11-slim-bookworm

# OCI-метаданные. VERSION/REVISION/CREATED подставляет CI при релизной сборке;
# при локальной сборке остаются дефолты. Нужны, чтобы с Docker Hub была
# ссылка на исходники и было видно, из какого коммита собран образ.
ARG VERSION=dev
ARG REVISION=unknown
ARG CREATED=""
LABEL org.opencontainers.image.title="up-and-run-stt" \
      org.opencontainers.image.description="Local Russian speech-to-text service (GigaAM, ONNX Runtime, CPU-only) with an OpenAI-compatible API" \
      org.opencontainers.image.source="https://github.com/vvpreo/up-and-run-stt" \
      org.opencontainers.image.documentation="https://github.com/vvpreo/up-and-run-stt#readme" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.created="${CREATED}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INFERENCE_BACKEND=onnx \
    GIGAAM_MODELS=v3_e2e_ctc \
    DEFAULT_LANGUAGE=ru \
    MODEL_CACHE_DIR=/app/data \
    PATH="/app/.venv/bin:$PATH"

# libsndfile — быстрый путь декодирования (wav/flac/ogg/opus/mp3)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Static ffmpeg binary — фолбэк-декодер для m4a/webm/wma и пр. (см. src/utils/audio.py).
# Один self-contained файл вместо apt-пакета с ~550 МБ библиотек.
# TARGETARCH подставляет buildx (amd64 / arm64) — имена сборок johnvansickle
# совпадают с этими значениями один-в-один. Дефолт нужен для обычного
# `docker build` без buildx, где переменная может быть пустой.
ARG TARGETARCH=amd64
RUN set -eux; \
    url="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-${TARGETARCH}-static.tar.xz"; \
    python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], '/tmp/ffmpeg.tar.xz')" "$url" \
    && python -c "import tarfile; tarfile.open('/tmp/ffmpeg.tar.xz').extractall('/tmp/ffmpeg')" \
    && cp /tmp/ffmpeg/*/ffmpeg /usr/local/bin/ffmpeg \
    && chmod +x /usr/local/bin/ffmpeg \
    && rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz \
    && ffmpeg -version | head -1

# Непривилегированный пользователь (uid 1000) создаётся ДО копирования venv.
# Это принципиально для размера: `chown -R` поверх уже скопированного venv
# переписал бы все 250 МБ файлов и породил бы второй такой же слой (+260 МБ
# к образу). Вместо этого владелец проставляется прямо в COPY --chown.
# Том, созданный старой root-версией образа, требует разового
#   docker run --rm -v gigaam-models:/data alpine chown -R 1000:1000 /data
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Silero-VAD (~2.3 МБ, MIT) — запечён в образ: детекция речи для умного
# чанкования длинного аудио (см. src/asr/vad.py). Зеркало — наш HF-репо.
RUN mkdir -p /app/vad \
    && python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/vvpreo/gigaam-v3-onnx/resolve/main/silero_vad.onnx', '/app/vad/silero_vad.onnx')" \
    && chown -R app:app /app/vad

# Готовое окружение из builder-стадии (пути внутри venv абсолютные, поэтому
# и там, и тут /app/.venv на одной и той же базе python:3.11-slim-bookworm)
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy application code (vendored gigaam НЕ нужен — ONNX-движок самодостаточен)
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app main.py /app/

# Create cache directory (mount point for the models volume)
RUN mkdir -p /app/data && chown app:app /app/data && chmod 755 /app/data

USER app

EXPOSE 9007

# Первый старт скачивает ONNX-веса (~845 МБ fp32 на модель) — щедрый start-period
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9007/health')" || exit 1

CMD ["python", "main.py"]
