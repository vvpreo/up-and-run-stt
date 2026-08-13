# Dockerfile for gigaam-stt (CPU mode)
# Manual PyTorch CPU installation (no official CPU image available).
# GigaAM model weights are NOT baked into the image — they are downloaded once
# at first startup into MODEL_CACHE_DIR, which is a named Docker volume
# (see docker-compose.yml). This keeps the image light and lets weights survive
# rebuilds and model switches without a re-download.

FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEVICE=cpu \
    GIGAAM_MODELS=v3_e2e_ctc \
    DEFAULT_LANGUAGE=ru \
    MODEL_CACHE_DIR=/app/data \
    HF_HOME=/app/data \
    TORCH_HOME=/app/data/torch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch CPU 2.11.0 (cached layer)
RUN pip install --no-cache-dir \
    torch==2.11.0+cpu \
    torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install ML/audio dependencies (cached layer)
# transformers/onnx/onnxruntime намеренно НЕ ставятся: GigaAM native их не
# импортирует в рабочих путях (проверено), это только лишний вес образа.
RUN pip install --no-cache-dir \
    numpy==2.0.2 \
    scipy \
    soundfile \
    librosa \
    sentencepiece

# Install web/framework dependencies (cached layer)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    python-multipart \
    psutil \
    tqdm

# Install GigaAM dependencies (cached layer)
RUN pip install --no-cache-dir \
    hydra-core \
    omegaconf

# Copy vendor/gigaam and install it
COPY vendor/gigaam /app/vendor/gigaam
RUN pip install --no-cache-dir --no-deps /app/vendor/gigaam && \
    rm -rf /app/vendor/gigaam

# NOTE: model weights are intentionally NOT copied here. They are fetched at
# runtime into MODEL_CACHE_DIR (a mounted volume). See docs/RU_QUALITY.md and README.

# Copy application code
COPY src/ /app/src/
COPY main.py /app/

# Create cache directory (mount point for the models volume)
RUN mkdir -p /app/data && chmod -R 755 /app/data

# Expose port
EXPOSE 9007

# Health check. start-period is generous because the FIRST startup downloads the
# GigaAM weights (~420 MB per v3 checkpoint) before the server becomes ready.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9007/health')" || exit 1

# Run the application
CMD ["python", "main.py"]
