"""
Главный модуль приложения FastAPI.
Создаёт и настраивает экземпляр FastAPI приложения,
регистрирует маршруты и обработчики событий жизненного цикла.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src import __version__
from src.asr.factory import create_asr_model
from src.asr.base import ASRModel
from src.asr.pool import ASRWorkerPool
from src.asr.registry import set_registry, list_models
from src.config import HOST, PORT, MODEL_WORKERS, GIGAAM_MODELS, DEFAULT_MODEL
from src.routes import asr_router, health_router, openai_router
from src.routes.asr import set_asr_model as set_asr_model_asr
from src.routes.health import set_asr_model as set_asr_model_health
from src.routes.openai import set_asr_model as set_asr_model_openai

logger = logging.getLogger(__name__)

# Global ASR model instance
_asr_model: ASRModel | None = None


def get_asr_model() -> ASRModel | None:
    """
    Возвращает глобальный экземпляр ASR модели.

    Returns:
        Экземпляр ASR модели или None, если модель не инициализирована.
    """
    return _asr_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет жизненным циклом приложения.

    Загружает модель при запуске и освобождает ресурсы при остановке.

    Args:
        app: Экземпляр FastAPI приложения.
    """
    global _asr_model

    # Startup
    logger.info(f"Starting gigaam-stt v{__version__}")
    logger.info("Initializing GigaAM ASR")

    try:
        if MODEL_WORKERS > 1:
            # Пул worker'ов работает с одной моделью (по умолчанию).
            # Выбор модели полем `model` доступен при MODEL_WORKERS=1.
            logger.info(f"Using worker pool with {MODEL_WORKERS} model instances")
            pool = ASRWorkerPool(num_workers=MODEL_WORKERS)
            pool.load_all()
            _asr_model = pool
            set_registry({DEFAULT_MODEL: pool})
        else:
            # Загружаем весь набор моделей инстанса; каждая держится в RAM.
            models: dict[str, ASRModel] = {}
            for name in GIGAAM_MODELS:
                m = create_asr_model(name)
                m.ensure_model_loaded()
                models[name] = m
            set_registry(models)
            _asr_model = models[DEFAULT_MODEL]

        # Set default model reference in all routers
        set_asr_model_asr(_asr_model)
        set_asr_model_health(_asr_model)
        set_asr_model_openai(_asr_model)

        logger.info(
            f"ASR models loaded successfully: {', '.join(GIGAAM_MODELS)} "
            f"(default={DEFAULT_MODEL}, workers={MODEL_WORKERS})"
        )

    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down gigaam-stt")
    for name, model in list_models().items():
        model.release_model()
        logger.info(f"ASR model released: {name}")


def create_app() -> FastAPI:
    """
    Создаёт и настраивает экземпляр FastAPI приложения.

    Returns:
        FastAPI: Настроенное приложение.

    Example:
        >>> app = create_app()
        >>> # Run with uvicorn
        >>> import uvicorn
        >>> uvicorn.run(app, host="0.0.0.0", port=9007)
    """
    app = FastAPI(
        title="gigaam-stt",
        description=(
            "Speech-to-text service powered by GigaAM (SberDevices) — "
            "high-quality Russian ASR.\n\n"
            "Endpoints:\n"
            "- **/v1/audio/transcriptions** — OpenAI-compatible (drop-in Whisper API replacement)\n"
            "- **/gigaam/asr** — native endpoint with extended response (segments, metrics)\n\n"
            "The instance serves the model set from GIGAAM_MODELS; a specific "
            "model is selected per request via the `model` field."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(asr_router)
    app.include_router(openai_router)

    # Тестовая веб-страница (запись с микрофона / загрузка файла).
    # Сама страница открыта; транскрипция с неё требует Bearer-токен,
    # если задан AUTH_TOKEN.
    index_html = Path(__file__).parent / "static" / "index.html"

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(index_html, media_type="text/html")

    return app


# Create the app instance
app = create_app()


def run_server(host: str | None = None, port: int | None = None) -> None:
    """
    Запускает сервер с помощью uvicorn.

    Args:
        host: Хост для привязки (по умолчанию из конфигурации).
        port: Порт для привязки (по умолчанию из конфигурации).

    Example:
        >>> from src.app import run_server
        >>> run_server(host="0.0.0.0", port=9007)
    """
    import uvicorn

    uvicorn.run(
        app,
        host=host or HOST,
        port=port or PORT,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
