"""
Маршруты FastAPI.

Содержит:
- health: Эндпоинт проверки здоровья сервиса
- asr: Основной эндпоинт транскрипции (/asr)
- openai: OpenAI-совместимый эндпоинт (/v1/audio/transcriptions)
"""

from src.routes.health import router as health_router
from src.routes.asr import router as asr_router
from src.routes.openai import router as openai_router

__all__ = [
    "health_router",
    "asr_router",
    "openai_router",
]
