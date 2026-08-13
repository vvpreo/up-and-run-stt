"""
Pydantic модели для API запросов и ответов.

Содержит:
- schemas: Основные модели транскрипции (WordTimestamp, Segment, TranscriptionResponse)
- openai: OpenAI-совместимые модели (OpenAIWord, OpenAISegment, OpenAITranscriptionResponse)
"""

from src.models.schemas import (
    WordTimestamp,
    Segment,
    ConfidenceMetrics,
    TranscriptionResponse,
)
from src.models.openai import (
    OpenAIWord,
    OpenAISegment,
    OpenAITranscriptionResponse,
)

__all__ = [
    "WordTimestamp",
    "Segment",
    "ConfidenceMetrics",
    "TranscriptionResponse",
    "OpenAIWord",
    "OpenAISegment",
    "OpenAITranscriptionResponse",
]
