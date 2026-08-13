"""
Endpoint распознавания эмоций (GigaAMEmo, только ONNX-бэкенд).

POST /gigaam/emotion — принимает аудиофайл (поле `audio_file`), возвращает
вероятности эмоций. Модель ленивая: грузится при первом запросе (~1 ГБ RAM).
"""

import asyncio
import logging
import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.auth import verify_token
from src.config import INFERENCE_BACKEND, SAMPLE_RATE
from src.services.limits import read_upload_limited, request_slot
from src.utils.audio import load_audio_from_file

logger = logging.getLogger(__name__)

router = APIRouter()

# Выключатель фичи (модель стоит ~1 ГБ RAM после первого запроса)
EMOTIONS_ENABLED = os.getenv("ENABLE_EMOTIONS", "true").lower() == "true"


def emotions_available() -> bool:
    """Эмоции доступны: включены и бэкенд ONNX (torch-путь не реализован)."""
    return EMOTIONS_ENABLED and INFERENCE_BACKEND == "onnx"


@router.post("/gigaam/emotion", dependencies=[Depends(verify_token)])
async def recognize_emotion(
    audio_file: UploadFile = File(..., description="Audio file to analyze"),
):
    """
    Распознаёт эмоции в речи: angry / sad / neutral / positive.

    Returns:
        {"emotions": {label: prob}, "dominant": label, "duration": sec}
    """
    if not emotions_available():
        raise HTTPException(
            status_code=501,
            detail="Emotion recognition is disabled (ENABLE_EMOTIONS=false or torch backend)",
        )

    from src.asr.onnx_emo import emo_model

    audio_content = await read_upload_limited(audio_file)
    with request_slot():
        audio = await asyncio.to_thread(load_audio_from_file, audio_content)
        del audio_content
        try:
            probs = await asyncio.to_thread(emo_model.classify, audio)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return {
        "emotions": probs,
        "dominant": max(probs, key=probs.get),
        "duration": round(len(audio) / SAMPLE_RATE, 2),
    }
