"""
Конфигурация приложения.
Все настройки загружаются из переменных окружения.
"""

import os
import logging

# Configure logging (уровень — через env LOG_LEVEL)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# =============================================================================
# ASR Engine Configuration
# =============================================================================

# Движок один — GigaAM native; он зафиксирован на уровне endpoint'ов
# (/gigaam/asr), а не переменной окружения. Константа используется в
# /health и заголовках ответов.
ENGINE = "gigaam"

# Бэкенд инференса: "torch" (PyTorch, текущий) или "onnx" (ONNX Runtime,
# образ без torch). Меняет только реализацию — API и модели те же.
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "torch").lower()

# Набор моделей, который обслуживает этот инстанс (через запятую).
# Все перечисленные модели загружаются при старте (веса докачиваются в том
# при необходимости) и держатся в RAM (~1.4 ГБ каждая). Конкретная модель
# для инференса выбирается полем `model` в запросе; ПЕРВАЯ в списке — модель
# по умолчанию (используется для 'whisper-1', пустого или неизвестного имени).
# Известные варианты: v3_e2e_ctc, v3_e2e_rnnt, v3_ctc, v3_rnnt.
# GIGAAM_MODEL (единственное число) поддерживается как алиас для docker run.
_models_raw = os.getenv("GIGAAM_MODELS") or os.getenv("GIGAAM_MODEL") or "v3_e2e_ctc"
GIGAAM_MODELS = [m.strip() for m in _models_raw.split(",") if m.strip()]

# Модель по умолчанию — первая в списке.
DEFAULT_MODEL = GIGAAM_MODELS[0]

# Все валидные имена вариантов GigaAM (для отличения «опечатка/чужая модель»
# от «известная, но не включённая на этом инстансе»).
KNOWN_GIGAAM_VARIANTS = {
    "v3_e2e_ctc", "v3_e2e_rnnt", "v3_ctc", "v3_rnnt",
    "v2_ctc", "v2_rnnt", "v1_ctc", "v1_rnnt",
}

# Maximum audio duration (seconds) considered "short" for GigaAM `.transcribe()`.
# For longer audio the service will split the audio into chunks and transcribe each chunk
# with repeated calls to `model.transcribe()`.
GIGAAM_MAX_SHORT_AUDIO_SEC = float(os.getenv("GIGAAM_MAX_SHORT_AUDIO_SEC", "25.0"))

# VAD-чанкование длинного аудио (silero-vad): резка по паузам речи вместо
# жёстких границ, тишина между чанками пропускается. Переопределяется
# per-request (native: ?vad=, OpenAI: chunking_strategy=auto|none).
VAD_CHUNKING = os.getenv("VAD_CHUNKING", "true").lower() == "true"

# Chunk size (seconds) used when splitting long audio into fixed-size chunks for repeated
# calls to `model.transcribe()`. Configure via env vars:
#   - GIGAAM_CHUNK_SEC: preferred chunk size in seconds (default: 30)
#   - GIGAAM_MIN_CHUNK_SEC: minimum chunk size in seconds to attempt before giving up (default: 5)
GIGAAM_CHUNK_SEC = int(os.getenv("GIGAAM_CHUNK_SEC", "30"))
GIGAAM_MIN_CHUNK_SEC = int(os.getenv("GIGAAM_MIN_CHUNK_SEC", "5"))

# Device to use: "auto", "cuda", "cpu", "mps"
DEVICE = os.getenv("DEVICE", "auto")

# Number of model worker instances for parallel inference (1 = single model, no parallelism)
MODEL_WORKERS = int(os.getenv("MODEL_WORKERS", "1"))

# Seconds before unloading idle model (0 = never)
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", "0"))

# Directory for caching downloaded models
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./data")

# Directory for saving audio chunks and results for debugging (None = disabled)
DEBUG_LOG_DIR = os.getenv("DEBUG_LOG_DIR", None)

# Audio sample rate (Whisper requirement)
SAMPLE_RATE = 16000

# Default language for transcription (used when not specified via API)
# Set to None or empty string to use auto-detection
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "ru")

# =============================================================================
# Server Configuration
# =============================================================================

# Host to bind the server
HOST = os.getenv("HOST", "0.0.0.0")

# Port to bind the server
PORT = int(os.getenv("PORT", "9007"))

# Максимальный размер загружаемого аудиофайла, МБ (защита от OOM,
# когда образ работает без реверс-прокси). 0 = без лимита.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))

# Максимум одновременно обрабатываемых/ожидающих транскрипций.
# При переполнении — 429 Too Many Requests. 0 = без лимита.
MAX_PENDING_REQUESTS = int(os.getenv("MAX_PENDING_REQUESTS", "8"))

# CORS: список origin'ов через запятую ('*' = все). Пусто = CORS выключен.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

# Swagger UI (/docs, /redoc, /openapi.json). false = скрыть.
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "true").lower() == "true"

# Уровень логирования (DEBUG/INFO/WARNING/ERROR)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# =============================================================================
# Adaptive Timeout Configuration
# =============================================================================

# Enable/disable adaptive timeout
TIMEOUT_ENABLED = os.getenv("TIMEOUT_ENABLED", "false").lower() == "true"

# Multiplier for expected processing time before timeout (e.g., 2.0 = allow 2x average time)
TIMEOUT_MULTIPLIER = float(os.getenv("TIMEOUT_MULTIPLIER", "2.0"))

# Minimum timeout in seconds (fallback when no history available)
TIMEOUT_MIN_SECONDS = float(os.getenv("TIMEOUT_MIN_SECONDS", "30.0"))

# Maximum timeout in seconds (safety cap)
TIMEOUT_MAX_SECONDS = float(os.getenv("TIMEOUT_MAX_SECONDS", "300.0"))

# Number of recent samples to keep for average calculation
TIMEOUT_HISTORY_SIZE = int(os.getenv("TIMEOUT_HISTORY_SIZE", "100"))

# =============================================================================
# Confidence Thresholds Configuration
# =============================================================================

# avg_logprob: average log probability of tokens (higher is better, typically -0.5 to 0)
CONFIDENCE_AVG_LOGPROB_THRESHOLD = float(os.getenv("CONFIDENCE_AVG_LOGPROB_THRESHOLD", "-1.0"))

# no_speech_prob: probability that segment contains no speech (lower is better, 0 to 1)
CONFIDENCE_NO_SPEECH_THRESHOLD = float(os.getenv("CONFIDENCE_NO_SPEECH_THRESHOLD", "0.6"))

# word_score: minimum average word alignment score (higher is better, 0 to 1)
CONFIDENCE_WORD_SCORE_THRESHOLD = float(os.getenv("CONFIDENCE_WORD_SCORE_THRESHOLD", "0.5"))

# word_prob: minimum average word probability from model (higher is better, 0 to 1)
CONFIDENCE_WORD_PROB_THRESHOLD = float(os.getenv("CONFIDENCE_WORD_PROB_THRESHOLD", "0.4"))

# low_prob_word_ratio: maximum ratio of low-probability words allowed (0 to 1)
CONFIDENCE_LOW_PROB_RATIO_THRESHOLD = float(os.getenv("CONFIDENCE_LOW_PROB_RATIO_THRESHOLD", "0.5"))

# Enable/disable automatic filtering of low-confidence results
CONFIDENCE_FILTER_ENABLED = os.getenv("CONFIDENCE_FILTER_ENABLED", "false").lower() == "true"

# =============================================================================
# Characters-per-second (chars/sec) Configuration
# =============================================================================
# Baseline characters per second considered normal (approx 20-30 chars/sec typical speaking rate)
MAX_CHARS_PER_SECOND = float(os.getenv("MAX_CHARS_PER_SECOND", "25.0"))

# =============================================================================
# Memory Monitoring Configuration
# =============================================================================

# Enable/disable memory monitoring
MEMORY_LOG_ENABLED = os.getenv("MEMORY_LOG_ENABLED", "false").lower() == "true"

# Interval for memory logging in seconds (default: 60)
MEMORY_LOG_INTERVAL = int(os.getenv("MEMORY_LOG_INTERVAL", "60"))

# Number of top allocation sites to log (0 to disable tracemalloc)
MEMORY_LOG_TOP_ALLOCATIONS = int(os.getenv("MEMORY_LOG_TOP_ALLOCATIONS", "5"))

# Multiplier - if observed chars/sec exceeds baseline * multiplier, mark as suspicious
CHARS_PER_SECOND_MULTIPLIER = float(os.getenv("CHARS_PER_SECOND_MULTIPLIER", "3.0"))

# Minimum audio duration (seconds) to apply chars/sec checks (avoid noisy short-audio edge cases)
CHARS_PER_SECOND_MIN_AUDIO_SEC = float(os.getenv("CHARS_PER_SECOND_MIN_AUDIO_SEC", "0.5"))


# =============================================================================
# Authentication Configuration
# =============================================================================

# API token for authentication. Empty/unset = auth disabled (open service).
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

# =============================================================================
# Initialize Cache Directories
# =============================================================================

def init_cache_directories():
    """Set cache directories if specified."""
    if MODEL_CACHE_DIR:
        os.environ["HF_HOME"] = MODEL_CACHE_DIR
        os.environ["TORCH_HOME"] = os.path.join(MODEL_CACHE_DIR, "torch")

    if DEBUG_LOG_DIR:
        os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
        logger.info(f"Debug logging enabled, saving to: {DEBUG_LOG_DIR}")


# Initialize on import
init_cache_directories()
