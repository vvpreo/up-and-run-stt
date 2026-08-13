"""
Сервисные модули.

Содержит:
- performance: Отслеживание производительности транскрипции
- timeout: Управление таймаутами транскрипции
- debug: Отладочное логирование аудио и результатов
"""

from src.services.performance import PerformanceTracker
from src.services.timeout import transcribe_with_timeout, TranscriptionTimeoutError, get_zombie_count
from src.services.debug import save_debug_log

__all__ = [
    "PerformanceTracker",
    "transcribe_with_timeout",
    "TranscriptionTimeoutError",
    "get_zombie_count",
    "save_debug_log",
]
