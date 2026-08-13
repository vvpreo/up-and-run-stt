"""
Реестр моделей инстанса.

Инстанс обслуживает набор моделей из GIGAAM_MODELS (все загружаются при
старте и держатся в RAM). Конкретная модель для инференса выбирается полем
`model` в запросе — как принято в OpenAI-совместимых API. Правила:

- имя из набора инстанса              -> эта модель;
- пусто / 'whisper-1' / прочие имена  -> модель по умолчанию (первая в списке);
- известный вариант GigaAM вне набора -> 400 (модель не включена на инстансе).
"""

import logging
from typing import TYPE_CHECKING, Dict, Optional

from fastapi import HTTPException

from src.config import DEFAULT_MODEL, KNOWN_GIGAAM_VARIANTS

if TYPE_CHECKING:
    from src.asr.base import ASRModel

logger = logging.getLogger(__name__)

_registry: Dict[str, "ASRModel"] = {}


def set_registry(models: Dict[str, "ASRModel"]) -> None:
    """Устанавливает реестр {имя модели -> экземпляр} (вызывается из app)."""
    global _registry
    _registry = models


def list_models() -> Dict[str, "ASRModel"]:
    """Возвращает реестр моделей инстанса."""
    return _registry


def resolve_model(requested: Optional[str]) -> "ASRModel":
    """
    Возвращает экземпляр модели для запрошенного имени.

    Raises:
        HTTPException(400): Известный вариант GigaAM, не включённый на инстансе.
        HTTPException(503): Реестр ещё не инициализирован.
    """
    if not _registry:
        raise HTTPException(status_code=503, detail="ASR models are not loaded yet")

    name = (requested or "").strip()

    if name in _registry:
        return _registry[name]

    if name in KNOWN_GIGAAM_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{name}' is not enabled on this instance. "
                f"Available models: {', '.join(_registry)}"
            ),
        )

    # Пустое имя, 'whisper-1' и прочие «чужие» имена — модель по умолчанию
    return _registry.get(DEFAULT_MODEL) or next(iter(_registry.values()))
