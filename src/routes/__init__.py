"""
Маршруты FastAPI.

Содержит:
- health: Эндпоинт проверки здоровья сервиса
- asr: Основной эндпоинт транскрипции (/asr)
- openai: OpenAI-совместимый эндпоинт (/v1/audio/transcriptions)
- stream: потоковый приём аудио по WebSocket (/v1/audio/stream)
"""

from src.routes.health import router as health_router
from src.routes.asr import router as asr_router
from src.routes.openai import router as openai_router
from src.routes.emotion import router as emotion_router
from src.routes.stream import router as stream_router

__all__ = [
    "health_router",
    "asr_router",
    "openai_router",
    "emotion_router",
    "stream_router",
]
