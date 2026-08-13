"""
Дополнительные OpenAI-совместимые endpoints.

- GET /v1/models, GET /v1/models/{id} — энумерация моделей из реестра
  (GUI-клиенты вроде Open WebUI/LibreChat опрашивают их для выпадашек).
- POST /v1/audio/translations — корректный отказ: GigaAM русскоязычная
  модель и перевод на английский не поддерживает.
"""

import time

from fastapi import APIRouter, Depends, HTTPException

from src.asr.registry import list_models
from src.auth import verify_token

router = APIRouter()

# «Дата создания» моделей — время старта процесса (OpenAI-клиенты требуют поле)
_CREATED = int(time.time())


def _model_object(name: str) -> dict:
    return {
        "id": name,
        "object": "model",
        "created": _CREATED,
        "owned_by": "gigaam-stt",
    }


@router.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models_openai() -> dict:
    return {
        "object": "list",
        "data": [_model_object(name) for name in list_models()],
    }


@router.get("/v1/models/{model_id}", dependencies=[Depends(verify_token)])
async def retrieve_model(model_id: str) -> dict:
    if model_id not in list_models():
        raise HTTPException(
            status_code=404,
            detail=f"The model '{model_id}' does not exist on this instance",
        )
    return _model_object(model_id)


@router.post("/v1/audio/translations", dependencies=[Depends(verify_token)])
async def translations_not_supported() -> None:
    raise HTTPException(
        status_code=400,
        detail=(
            "Audio translation is not supported: GigaAM is a Russian-language "
            "ASR model. Use /v1/audio/transcriptions instead."
        ),
    )
