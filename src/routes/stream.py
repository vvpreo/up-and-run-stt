"""
Потоковый приём аудио: WebSocket /v1/audio/stream.

В отличие от `stream=true` на /v1/audio/transcriptions (там стримится только
ОТВЕТ, а запрос уходит целым файлом), здесь стримится вход: клиент льёт PCM
по мере того, как человек говорит, а сервер сам режет поток на фразы и
присылает текст каждой, не дожидаясь конца записи.

Клиент намеренно тупой: шлёт кадры и ничего не решает. Вся сегментация —
на сервере, нейросетевым silero-vad (см. services/stream_session.py).

Промежуточных гипотез внутри фразы нет и быть не может: GigaAM офлайновая,
она принимает законченный отрезок. Гранулярность результата = фраза.
"""

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.asr.registry import resolve_model
from src.config import (
    AUTH_TOKEN,
    DEFAULT_LANGUAGE,
    SAMPLE_RATE,
    STREAM_MAX_PHRASE_SEC,
    STREAM_MAX_QUEUED_PHRASES,
    STREAM_MAX_SESSIONS,
    STREAM_MIN_PHRASE_SEC,
    STREAM_SILENCE_MS,
)
from src.services.stream_session import PhraseSegmenter, pcm16_to_float32

logger = logging.getLogger(__name__)

router = APIRouter()

# Число живых сессий. Открытая сессия стоит ~0.5% ядра (VAD) и ~2 МБ, так что
# лимит здесь на порядок выше, чем MAX_PENDING_REQUESTS для обычных запросов.
_sessions = 0
_sessions_lock = asyncio.Lock()

# Коды закрытия. 1008 = policy violation, 1013 = try again later.
WS_UNAUTHORIZED = 1008
WS_BUSY = 1013


async def _acquire_session() -> bool:
    global _sessions
    async with _sessions_lock:
        if STREAM_MAX_SESSIONS > 0 and _sessions >= STREAM_MAX_SESSIONS:
            return False
        _sessions += 1
        return True


async def _release_session() -> None:
    global _sessions
    async with _sessions_lock:
        _sessions = max(0, _sessions - 1)


def active_sessions() -> int:
    """Для /health."""
    return _sessions


@router.websocket("/v1/audio/stream")
async def audio_stream(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    language: Optional[str] = Query(None),
) -> None:
    """
    Живой поток аудио -> пофразный текст. Протокол описан в GET /v1/audio/stream.
    """
    # Авторизация ДО accept: неаутентифицированное соединение не должно
    # подниматься вообще. Starlette превращает close() до accept() в
    # ответ HTTP 403 на апгрейд — клиент видит именно отказ рукопожатия,
    # а не закрытый сокет. Токен идёт query-параметром, потому что
    # браузерный WebSocket не умеет слать заголовки.
    if AUTH_TOKEN:
        import secrets

        if not token or not secrets.compare_digest(token, AUTH_TOKEN):
            await websocket.close(code=WS_UNAUTHORIZED, reason="invalid token")
            return

    # А вот «занято» — не отказ в доступе, и клиенту важно отличать одно от
    # другого, чтобы знать, стоит ли повторять. До accept() это неразличимо
    # (тоже был бы 403), поэтому соединение принимается и закрывается уже
    # своим кодом с внятной причиной.
    if not await _acquire_session():
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "error": "too many streaming sessions",
                "limit": STREAM_MAX_SESSIONS,
                "retry": True,
            }
        )
        await websocket.close(code=WS_BUSY, reason="too many streaming sessions")
        return

    await websocket.accept()
    started = time.perf_counter()

    try:
        selected = resolve_model(model)
    except Exception as e:  # неизвестная модель — сообщаем и закрываем
        await websocket.send_json({"type": "error", "error": str(e)})
        await websocket.close()
        await _release_session()
        return

    lang = language or DEFAULT_LANGUAGE
    segmenter = PhraseSegmenter()
    # Инференс идёт в отдельной задаче, чтобы приём кадров не вставал на
    # время распознавания: иначе сокет копил бы аудио в буферах ядра.
    queue: asyncio.Queue = asyncio.Queue()
    seq = 0
    text_parts: list[str] = []

    await websocket.send_json(
        {
            "type": "session.created",
            "model": getattr(selected, "model_name", None) or str(model or "default"),
            "language": lang,
            "sample_rate": SAMPLE_RATE,
            "format": "pcm_s16le",
            "silence_ms": STREAM_SILENCE_MS,
            "max_phrase_sec": STREAM_MAX_PHRASE_SEC,
        }
    )

    async def transcribe_worker() -> None:
        nonlocal seq
        while True:
            phrase = await queue.get()
            if phrase is None:
                queue.task_done()
                return
            t0 = time.perf_counter()
            try:
                result = await asyncio.to_thread(
                    selected.transcribe,
                    phrase.audio,
                    "transcribe",
                    lang,
                    False,
                    "text",
                )
                text = (result if isinstance(result, str) else result.text).strip()
            except Exception as e:
                logger.exception("[stream] phrase transcription failed")
                await websocket.send_json({"type": "error", "error": str(e)})
                queue.task_done()
                continue

            if text:
                seq += 1
                text_parts.append(text)
                await websocket.send_json(
                    {
                        "type": "transcript.text.delta",
                        "delta": text,
                        "seq": seq,
                        "start": round(phrase.start_sec, 2),
                        "duration": round(phrase.duration_sec, 2),
                        "inference_sec": round(time.perf_counter() - t0, 3),
                        "forced": phrase.forced,
                    }
                )
            queue.task_done()

    worker = asyncio.create_task(transcribe_worker())
    speaking = False

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if (data := message.get("bytes")) is not None:
                for phrase in segmenter.feed(pcm16_to_float32(data)):
                    # Бэкпрешер: клиент может лить быстрее реального времени
                    # (например, проигрывая файл в сокет). Копить фразы без
                    # предела нельзя — растёт память, поэтому самые старые
                    # выбрасываются, о чём клиент узнаёт явно.
                    if queue.qsize() >= STREAM_MAX_QUEUED_PHRASES:
                        try:
                            queue.get_nowait()
                            queue.task_done()
                        except asyncio.QueueEmpty:
                            pass
                        await websocket.send_json(
                            {
                                "type": "stream.overflow",
                                "message": (
                                    "inference is behind; oldest phrase dropped"
                                ),
                                "queued": queue.qsize(),
                            }
                        )
                    queue.put_nowait(phrase)

                if segmenter.is_speaking != speaking:
                    speaking = segmenter.is_speaking
                    await websocket.send_json(
                        {"type": "speech.started" if speaking else "speech.stopped"}
                    )
                continue

            if (raw := message.get("text")) is not None:
                try:
                    cmd = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "error": "control frame must be JSON"}
                    )
                    continue

                action = cmd.get("type")
                if action == "commit":
                    # Принудительно закрыть текущую фразу, не дожидаясь паузы
                    if (tail := segmenter.flush()) is not None:
                        queue.put_nowait(tail)
                elif action == "close":
                    break
                else:
                    await websocket.send_json(
                        {"type": "error", "error": f"unknown command: {action!r}"}
                    )

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[stream] session failed")
    finally:
        # Хвост последней фразы — иначе последние слова пропали бы
        try:
            if (tail := segmenter.flush()) is not None:
                queue.put_nowait(tail)
        except Exception:
            pass

        queue.put_nowait(None)
        try:
            await asyncio.wait_for(worker, timeout=60)
        except (asyncio.TimeoutError, Exception):
            worker.cancel()

        try:
            await websocket.send_json(
                {
                    "type": "transcript.text.done",
                    "text": " ".join(text_parts),
                    "phrases": seq,
                    "session_sec": round(time.perf_counter() - started, 2),
                }
            )
            await websocket.close()
        except Exception:
            pass  # клиент уже отвалился — нормальный сценарий
        await _release_session()


# ---------------------------------------------------------------------------
# Описание протокола для Swagger.
#
# FastAPI не умеет класть WebSocket-руты в OpenAPI, поэтому рядом висит
# обычный GET, который отдаёт спецификацию машиночитаемо и заодно
# документирует протокол на странице /docs — интегратору не нужно лезть
# в исходники.
# ---------------------------------------------------------------------------


class StreamProtocol(BaseModel):
    """Машиночитаемое описание протокола WebSocket-эндпоинта."""

    endpoint: str = Field(..., examples=["/v1/audio/stream"])
    transport: str = Field(..., examples=["websocket"])
    audio_format: dict
    query_params: dict
    client_messages: dict
    server_events: dict
    limits: dict
    notes: list[str]


@router.get(
    "/v1/audio/stream",
    response_model=StreamProtocol,
    tags=["Streaming"],
    summary="Protocol description for the live audio WebSocket",
    description="""
Describes the **WebSocket** endpoint at the same path, `ws(s)://<host>/v1/audio/stream`,
which OpenAPI cannot represent directly. Fetch it to discover the protocol at
runtime, or just read it here.

### What this endpoint is for

`POST /v1/audio/transcriptions` with `stream=true` streams the **response**: the
audio still goes up as one complete request. This endpoint streams the **input** —
you push audio while the person is still speaking and get text back phrase by
phrase.

Use it for live dictation. For a file you already have, the regular endpoint is
simpler and faster.

### How it works

You send raw PCM frames; the server runs silero-vad over them, decides where
phrases end, and transcribes each finished phrase. **There are no partial
hypotheses inside a phrase** — GigaAM is an offline model and needs a complete
segment, so the finest granularity you can get is one phrase.

Typical latency from the end of a phrase to its text is about a second:
~600 ms to confirm the pause, plus inference (~0.3 s for a 5-second phrase).

### Minimal client

```python
import asyncio, json, websockets, soundfile as sf, numpy as np

async def main():
    url = "ws://localhost:9007/v1/audio/stream?token=YOUR_TOKEN&language=ru"
    async with websockets.connect(url) as ws:
        print(await ws.recv())                      # session.created

        async def receive():
            async for msg in ws:
                event = json.loads(msg)
                if event["type"] == "transcript.text.delta":
                    print(event["delta"])
                elif event["type"] == "transcript.text.done":
                    return
        task = asyncio.create_task(receive())

        audio, sr = sf.read("speech.wav", dtype="int16")   # must be 16 kHz mono
        for i in range(0, len(audio), 1600):               # 100 ms frames
            await ws.send(audio[i:i + 1600].tobytes())
            await asyncio.sleep(0.1)                       # pace it like a live mic
        await ws.send(json.dumps({"type": "close"}))
        await task

asyncio.run(main())
```

### Audio format

Raw **PCM signed 16-bit little-endian, 16 000 Hz, mono**, sent as binary frames.
Any frame size works — the server reassembles them — but 50–200 ms per frame is
a sensible range.

Compressed input (opus, webm) is deliberately not accepted: it would need a
decoder process per connection, whereas raw PCM costs 256 kbit/s and no CPU.
Resample on the client if your capture device runs at another rate.

### Authorization

Browsers cannot set headers on a WebSocket, so the token goes in the query
string: `?token=<AUTH_TOKEN>`. When `AUTH_TOKEN` is unset, the parameter is
ignored.

A bad token is refused **at the handshake**: the upgrade request is answered
with **HTTP 403** and no WebSocket is established. A session-limit refusal looks
different on purpose — there the socket does open, you get an `error` event with
`retry: true`, and it closes with code **1013**. That way a client can tell
"you are not allowed" from "come back later" and only retry the second.

### Cost

An open session costs about 0.5% of a CPU core for continuous voice activity
detection and roughly 2 MB of buffers. Inference happens only when a phrase
closes, so a live speaker consumes on the order of 5% of a core. This is why
the session limit (`STREAM_MAX_SESSIONS`) is much higher than the limit on
concurrent file transcriptions (`MAX_PENDING_REQUESTS`).
""",
)
async def stream_protocol() -> StreamProtocol:
    """Отдаёт протокол WebSocket-эндпоинта машиночитаемо."""
    return StreamProtocol(
        endpoint="/v1/audio/stream",
        transport="websocket",
        audio_format={
            "encoding": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "frame_size": "any; 50-200 ms recommended",
            "compressed_input": "not supported — send raw PCM",
        },
        query_params={
            "token": "AUTH_TOKEN; required when authorization is enabled",
            "model": "model name from GIGAAM_MODELS; omit for the default",
            "language": f"language code; defaults to {DEFAULT_LANGUAGE}",
        },
        client_messages={
            "<binary>": "audio frame, raw PCM s16le",
            '{"type": "commit"}': "close the current phrase now, without waiting for a pause",
            '{"type": "close"}': "finish the session; server flushes the tail and sends transcript.text.done",
        },
        server_events={
            "session.created": "sent once on connect; echoes the effective settings",
            "speech.started": "voice activity detected",
            "speech.stopped": "pause detected",
            "transcript.text.delta": "text of one finished phrase; fields: delta, seq, start, duration, inference_sec, forced",
            "transcript.text.done": "final event; fields: text (everything joined), phrases, session_sec",
            "stream.overflow": "inference fell behind and the oldest queued phrase was dropped",
            "error": "recoverable problem; the session stays open unless stated otherwise",
        },
        limits={
            "max_sessions": STREAM_MAX_SESSIONS,
            "max_queued_phrases": STREAM_MAX_QUEUED_PHRASES,
            "silence_ms": STREAM_SILENCE_MS,
            "max_phrase_sec": STREAM_MAX_PHRASE_SEC,
            "min_phrase_sec": STREAM_MIN_PHRASE_SEC,
        },
        notes=[
            "No partial hypotheses inside a phrase: the model is offline and needs a complete segment.",
            "A phrase longer than max_phrase_sec is cut anyway and arrives with forced=true.",
            "Fragments shorter than min_phrase_sec are dropped as clicks or noise.",
            "Silence is trimmed before inference, which both saves CPU and improves accuracy.",
            "A rejected token fails the handshake with HTTP 403; no WebSocket is opened.",
            "Hitting the session limit opens the socket, sends an error event with retry=true and closes with code 1013.",
        ],
    )
