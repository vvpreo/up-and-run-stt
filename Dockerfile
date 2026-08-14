# Dockerfile — основной образ up-and-run-stt: инференс на ONNX Runtime, без PyTorch.
# ONNX-веса НЕ запечены в образ: скачиваются при первом старте с HF Hub
# (GIGAAM_ONNX_BASE_URL, по умолчанию vvpreo/gigaam-v3-onnx) в том gigaam-models.
# Токенизаторы e2e-моделей — с CDN SberDevices.
# PyTorch-вариант для конвертации весов и сверки — Dockerfile.torch.

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    INFERENCE_BACKEND=onnx \
    GIGAAM_MODELS=v3_e2e_ctc \
    DEFAULT_LANGUAGE=ru \
    MODEL_CACHE_DIR=/app/data

# libsndfile — быстрый путь декодирования (wav/flac/ogg/opus/mp3)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Static ffmpeg binary — фолбэк-декодер для m4a/webm/wma и пр. (см. src/utils/audio.py).
# Один self-contained файл вместо apt-пакета с ~550 МБ библиотек.
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz', '/tmp/ffmpeg.tar.xz')" \
    && python -c "import tarfile; tarfile.open('/tmp/ffmpeg.tar.xz').extractall('/tmp/ffmpeg')" \
    && cp /tmp/ffmpeg/*/ffmpeg /usr/local/bin/ffmpeg \
    && chmod +x /usr/local/bin/ffmpeg \
    && rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz \
    && ffmpeg -version | head -1

WORKDIR /app

# Silero-VAD (~2.3 МБ, MIT) — запечён в образ: детекция речи для умного
# чанкования длинного аудио (см. src/asr/vad.py). Зеркало — наш HF-репо.
RUN mkdir -p /app/vad && python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/vvpreo/gigaam-v3-onnx/resolve/main/silero_vad.onnx', '/app/vad/silero_vad.onnx')"

# Инференс и сервис: onnxruntime вместо torch (~50 МБ вместо ~850)
RUN pip install --no-cache-dir \
    onnxruntime \
    numpy \
    scipy \
    soundfile \
    sentencepiece \
    omegaconf \
    fastapi \
    uvicorn \
    pydantic \
    python-multipart \
    psutil

# Copy application code (vendored gigaam НЕ нужен — ONNX-движок самодостаточен)
COPY src/ /app/src/
COPY main.py /app/

# Create cache directory (mount point for the models volume)
RUN mkdir -p /app/data && chmod -R 755 /app/data

# Непривилегированный пользователь (uid 1000); владеет /app.
# Том, созданный root-версией образа, требует разового
#   docker run --rm -v gigaam-models:/data alpine chown -R 1000:1000 /data
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app

EXPOSE 9007

# Первый старт скачивает ONNX-веса (~845 МБ fp32 на модель) — щедрый start-period
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9007/health')" || exit 1

CMD ["python", "main.py"]
