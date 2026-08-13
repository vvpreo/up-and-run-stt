"""
Защита сервиса от перегрузки: лимит размера загрузки и лимит очереди.

Рассчитано на «голый» запуск публичного образа (`docker run`) без
реверс-прокси перед сервисом.
"""

import logging
from contextlib import contextmanager

from fastapi import HTTPException, UploadFile

from src.config import MAX_PENDING_REQUESTS, MAX_UPLOAD_MB

logger = logging.getLogger(__name__)

# Счётчик запросов «в полёте» (обрабатываются или ждут model_lock).
# Инкремент/декремент происходят только в event loop — гонок нет.
_inflight = 0


async def read_upload_limited(upload: UploadFile) -> bytes:
    """
    Читает загруженный файл, не позволяя превысить MAX_UPLOAD_MB.

    Raises:
        HTTPException(413): Файл больше лимита.
    """
    if MAX_UPLOAD_MB <= 0:
        return await upload.read()

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        logger.warning(f"Upload rejected: exceeds {MAX_UPLOAD_MB} MB limit")
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds the {MAX_UPLOAD_MB} MB upload limit",
        )
    return data


@contextmanager
def request_slot():
    """
    Слот на обработку запроса. При переполнении очереди — 429.

    Использовать как context manager вокруг всей обработки запроса.
    """
    global _inflight
    if MAX_PENDING_REQUESTS > 0 and _inflight >= MAX_PENDING_REQUESTS:
        logger.warning(
            f"Request rejected: {_inflight} transcriptions already pending "
            f"(limit {MAX_PENDING_REQUESTS})"
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too many pending transcriptions ({_inflight}). "
                f"Retry later or raise MAX_PENDING_REQUESTS."
            ),
        )
    _inflight += 1
    try:
        yield
    finally:
        _inflight -= 1


def pending_count() -> int:
    """Текущее число запросов в обработке/очереди (для /health)."""
    return _inflight
