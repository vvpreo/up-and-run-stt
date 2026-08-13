"""
Фабрика для создания ASR моделей.

Движок один — GigaAM native (CPU); он зафиксирован на уровне endpoint'ов
(/gigaam/asr), а не конфигурацией. Фабрика оставлена как точка расширения:
новые движки в будущем добавляются сюда и получают свои endpoint'ы.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.asr.base import ASRModel

logger = logging.getLogger(__name__)


def create_asr_model(model_name: str | None = None) -> "ASRModel":
    """
    Создаёт экземпляр GigaAM ASR модели.

    Args:
        model_name: Вариант модели GigaAM (v3_e2e_ctc, ...); None = дефолт.

    Returns:
        ASRModel: Экземпляр GigaAM ASR модели.
    """
    logger.info(f"Creating GigaAM ASR model ({model_name or 'default'})")

    from src.asr.gigaam import GigaAMASR

    return GigaAMASR(model_name=model_name)
