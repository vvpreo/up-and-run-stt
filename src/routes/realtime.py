"""
OpenAI-совместимый потоковый эндпоинт: WebSocket /v1/realtime.

Единственный живой эндпоинт сервиса: клиент льёт аудио по мере речи, сервер
сам режет поток на фразы (silero-VAD) и присылает текст. Реализует Realtime
API OpenAI в режиме ТОЛЬКО ТРАНСКРИПЦИИ (GA-диалект, `session.type =
"transcription"`), чтобы готовые клиенты — официальные SDK, LiveKit, Pipecat,
голосовые виджеты — работали, если указать им наш base_url. Всё, чего в
стандарте нет (нестабильный хвост, эмоции), — расширения по параметрам:
стандартный клиент их не увидит. Логика сессии — src/services/live_session.py,
здесь только язык сообщений.

Соответствие:

  клиент -> сервер
    session.update                 — формат аудио, модель, язык, turn_detection
    input_audio_buffer.append      — base64 PCM16 LE mono, 24 кГц (стандарт)
                                     или 16 кГц (расширение: {"rate": 16000})
    input_audio_buffer.commit      — закрыть текущую фразу
    input_audio_buffer.clear       — выбросить накопленное
  сервер -> клиент
    session.created / session.updated
    input_audio_buffer.speech_started / speech_stopped / committed / cleared
    conversation.item.added / conversation.item.done
    conversation.item.input_audio_transcription.delta      — зафиксированные слова
    conversation.item.input_audio_transcription.completed  — полный текст фразы
    error

Семантика дельт. `delta` у OpenAI — «дописать к тексту», отозвать её нельзя.
Поэтому дельтами уходят только СТАБИЛИЗИРОВАННЫЕ слова (см.
src/services/stabilizer.py); нестабильный хвост в этот протокол не попадает.
Если модель задним числом переписала уже отправленное (числа, имена), новые
дельты по этой фразе не шлются, а авторитетный текст приходит в `completed`
— клиенты заменяют им накопленные дельты. Без черновиков (CPU) на фразу
приходит одна дельта с целой фразой, как у whisper-1.

Расширения (выключены по умолчанию, стандартные клиенты их не увидят):
  ?tentative=true — событие `x.transcription.tentative` {item_id, committed,
      tentative, rewrite, changed_from} на каждый тик черновика: нестабильный
      хвост фразы и признак того, что зафиксированное было переписано;
  ?emotions=true  — событие `phrase.emotion` {item_id, dominant, emotions};
  ?partials=true|false — включить/выключить черновики (по умолчанию включены,
      когда модель на GPU: STREAM_PARTIALS=auto). Без них на фразу приходит
      одна дельта с целой фразой.

Чего нет: ответы модели (response.*), аудио-выход, эфемерные ключи, G.711.
Старые beta-имена (`transcription_session.update`) принимаются.
"""

import asyncio
import base64
import json
import logging
import secrets
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.asr.registry import resolve_model
from src.config import (
    AUTH_TOKEN,
    DEFAULT_LANGUAGE,
    STREAM_IDLE_TIMEOUT_SEC,
    STREAM_MAX_PHRASE_SEC,
    STREAM_MAX_QUEUED_PHRASES,
    STREAM_MAX_SESSIONS,
    STREAM_MIN_PHRASE_SEC,
    STREAM_PARTIAL_GUARD_MS,
    STREAM_PARTIAL_INTERVAL_MS,
    STREAM_SILENCE_MS,
    STREAM_SOFT_CUT_SEC,
)
from src.services.live_session import (
    LiveSession,
    acquire_session,
    partial_model,
    partials_enabled,
    release_session,
)
from src.services.stream_session import PhraseSegmenter

logger = logging.getLogger(__name__)

router = APIRouter()

# Коды закрытия. 1008 = policy violation, 1013 = try again later.
WS_UNAUTHORIZED = 1008
WS_BUSY = 1013

_KEY_SUBPROTOCOL = "openai-insecure-api-key."
_NO_VAD_SILENCE_MS = 10**9  # turn_detection: null — фразу закрывает только commit


class _Resampler:
    """
    Потоковый ресемплинг 24 кГц -> 16 кГц (polyphase 2/3). Чанки нельзя
    ресемплить по отдельности — на каждом стыке был бы переходный процесс
    фильтра (щелчок). Поэтому слева держится контекст, а справа хвост
    придерживается до следующего чанка; задержка 10 мс.
    """

    CTX = 240  # сэмплов входа (10 мс), кратно 3

    def __init__(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._primed = False

    def feed(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import resample_poly

        data = np.concatenate([self._buf, x])
        usable = len(data) - (len(data) % 3)
        if usable < 3 * self.CTX:
            self._buf = data
            return np.zeros(0, dtype=np.float32)
        block, rest = data[:usable], data[usable:]
        out = resample_poly(block, 2, 3).astype(np.float32)
        left = 0 if not self._primed else self.CTX * 2 // 3
        right = (usable - self.CTX) * 2 // 3
        self._primed = True
        self._buf = np.concatenate([block[-2 * self.CTX:], rest])
        return out[left:right]


def _eid() -> str:
    return "event_" + uuid.uuid4().hex[:20]


def _client_key(ws: WebSocket) -> Optional[str]:
    auth = ws.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    for proto in ws.scope.get("subprotocols") or []:
        if proto.startswith(_KEY_SUBPROTOCOL):
            return proto[len(_KEY_SUBPROTOCOL):]
    return None


@router.websocket("/v1/realtime")
async def realtime(
    websocket: WebSocket,
    model: Optional[str] = Query(None),
    intent: Optional[str] = Query(None),
    token: Optional[str] = Query(None),
    emotions: bool = Query(False),
    tentative: bool = Query(False),
    partials: Optional[bool] = Query(None),
) -> None:
    """OpenAI Realtime API, режим транскрипции. Описание — в докстринге модуля."""
    offered = websocket.scope.get("subprotocols") or []
    # Браузеры передают ключ через subprotocol; сервер обязан выбрать один из
    # предложенных, иначе браузер разорвёт соединение.
    subprotocol = "realtime" if "realtime" in offered else None

    if AUTH_TOKEN:
        key = _client_key(websocket) or token
        if not key or not secrets.compare_digest(key, AUTH_TOKEN):
            await websocket.close(code=WS_UNAUTHORIZED, reason="invalid api key")
            return

    if not await acquire_session():
        await websocket.accept(subprotocol=subprotocol)
        await websocket.send_json(
            {
                "type": "error",
                "event_id": _eid(),
                "error": {
                    "type": "server_error",
                    "code": "rate_limit_exceeded",
                    "message": "too many streaming sessions",
                    "param": None,
                },
            }
        )
        await websocket.close(code=WS_BUSY, reason="too many streaming sessions")
        return

    await websocket.accept(subprotocol=subprotocol)

    # ------------------------------------------------------------ состояние
    beta = False  # клиент говорит на старом диалекте (transcription_session.*)
    cfg = {
        "rate": 24000,
        "model": model,
        "language": None,
        "prompt": None,
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": STREAM_SILENCE_MS,
        },
    }
    session_id = "sess_" + uuid.uuid4().hex[:20]
    live: Optional[LiveSession] = None
    resampler = _Resampler()
    # phrase_id -> {"id": item_id, "sent": текст, уже ушедший дельтами, "stale": bool}
    items: dict = {}
    last_committed: Optional[str] = None
    send_lock = asyncio.Lock()

    async def send(event: dict) -> None:
        event.setdefault("event_id", _eid())
        async with send_lock:
            await websocket.send_json(event)

    async def send_error(message: str, code: str, client_event_id=None, etype="invalid_request_error") -> None:
        await send(
            {
                "type": "error",
                "error": {
                    "type": etype,
                    "code": code,
                    "message": message,
                    "param": None,
                    "event_id": client_event_id,
                },
            }
        )

    def session_object() -> dict:
        td = cfg["turn_detection"]
        transcription = {
            "model": cfg["model"] or "whisper-1",
            "language": cfg["language"],
            "prompt": cfg["prompt"],
        }
        if beta:
            return {
                "id": session_id,
                "object": "realtime.transcription_session",
                "input_audio_format": "pcm16",
                "input_audio_transcription": transcription,
                "turn_detection": td,
                "input_audio_noise_reduction": None,
                "include": None,
            }
        return {
            "type": "transcription",
            "object": "realtime.transcription_session",
            "id": session_id,
            "expires_at": int(time.time()) + 3600,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": cfg["rate"]},
                    "transcription": transcription,
                    "noise_reduction": None,
                    "turn_detection": td,
                }
            },
            "include": None,
        }

    def item_for(phrase_id: int) -> dict:
        if phrase_id not in items:
            items[phrase_id] = {"id": "item_" + uuid.uuid4().hex[:20], "sent": "", "stale": False}
        return items[phrase_id]

    def segmenter_params() -> dict:
        td = cfg["turn_detection"]
        if not td:
            return {"silence_ms": _NO_VAD_SILENCE_MS}
        return {
            "threshold": td.get("threshold", 0.5),
            "silence_ms": td.get("silence_duration_ms", STREAM_SILENCE_MS),
        }

    # ------------------------------------------- события сессии -> протокол OpenAI
    async def emit(ev: dict) -> None:
        nonlocal last_committed
        kind = ev["kind"]
        if kind in ("speech_started", "speech_stopped"):
            if not cfg["turn_detection"]:
                return
            ms = int(ev["at_sec"] * 1000)
            item = item_for(ev["phrase_id"])
            if kind == "speech_started":
                await send({"type": "input_audio_buffer.speech_started", "audio_start_ms": ms, "item_id": item["id"]})
            else:
                await send({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": ms, "item_id": item["id"]})
        elif kind == "phrase_closed":
            # Записи давно закрытых фраз больше никому не нужны (эмоция приходит
            # через секунды после текста). Без чистки словарь рос бы всю сессию:
            # по записи на фразу плюс на каждый отброшенный щелчок или кашель.
            for old_id in [k for k in items if k < ev["phrase_id"] - 8]:
                del items[old_id]
            item = item_for(ev["phrase_id"])
            await send({"type": "input_audio_buffer.committed", "item_id": item["id"], "previous_item_id": last_committed})
            body = {
                "id": item["id"],
                "type": "message",
                "status": "completed",
                "role": "user",
                "content": [{"type": "input_audio", "transcript": None}],
            }
            if beta:
                await send({"type": "conversation.item.created", "previous_item_id": last_committed, "item": body})
            else:
                await send({"type": "conversation.item.added", "previous_item_id": last_committed, "item": body})
                await send({"type": "conversation.item.done", "previous_item_id": last_committed, "item": body})
            last_committed = item["id"]
        elif kind == "partial":
            item = item_for(ev["phrase_id"])
            if ev["rewrite"]:
                # Уже отправленное отозвать нельзя: дельты по этой фразе
                # прекращаются, правильный текст придёт в completed.
                item["stale"] = True
            if tentative:
                # Расширение (?tentative=true): нестабильный хвост и факт перезаписи.
                # В стандарте такого события нет; клиент, которому надо красить
                # «ещё может измениться», включает его явно.
                await send(
                    {
                        "type": "x.transcription.tentative",
                        "item_id": item["id"],
                        "committed": ev["committed"],
                        "tentative": ev["tentative"],
                        "rewrite": ev["rewrite"],
                        "changed_from": ev["changed_from"],
                    }
                )
            if item["stale"] or not ev["new_words"]:
                return
            delta = (" " if item["sent"] else "") + " ".join(ev["new_words"])
            item["sent"] += delta
            await send(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": item["id"],
                    "content_index": 0,
                    "delta": delta,
                }
            )
        elif kind in ("final", "empty"):
            item = item_for(ev["phrase_id"])
            text = ev.get("text", "")
            sent = item["sent"]
            if text and not sent:
                rest = text                      # без черновиков: вся фраза одной дельтой
            elif text.startswith(sent) and not item["stale"]:
                rest = text[len(sent):]          # финал продолжил отправленное
            else:
                rest = ""                        # финал разошёлся с дельтами — только completed
            if rest:
                await send(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": item["id"],
                        "content_index": 0,
                        "delta": rest,
                    }
                )
            await send(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item["id"],
                    "content_index": 0,
                    "transcript": text,
                    "usage": {"type": "duration", "seconds": ev.get("duration", 0.0)},
                }
            )
            if kind == "empty" or not (live and live.emotions_on):
                items.pop(ev["phrase_id"], None)  # эмоция этой фразе не придёт
        elif kind == "emotion":
            # Расширение (только по ?emotions=true): у OpenAI такого события нет,
            # стандартные клиенты неизвестный type игнорируют.
            item = item_for(ev["phrase_id"])
            await send(
                {
                    "type": "phrase.emotion",
                    "item_id": item["id"],
                    "dominant": ev["dominant"],
                    "emotions": ev["emotions"],
                }
            )
            items.pop(ev["phrase_id"], None)
        elif kind == "overflow":
            await send_error("inference is behind; oldest phrase dropped", "server_overloaded", etype="server_error")
        elif kind == "error":
            await send_error(ev["message"], "transcription_failed", etype="server_error")

    def ensure_live() -> LiveSession:
        nonlocal live
        if live is None:
            selected = resolve_model(cfg["model"])
            draft_model = partial_model(selected)
            from src.routes.emotion import emotions_available

            live = LiveSession(
                model=selected,
                draft_model=draft_model,
                language=cfg["language"] or DEFAULT_LANGUAGE,
                partials=partials_enabled(partials, draft_model),
                emotions=bool(emotions) and emotions_available(),
                emit=emit,
                segmenter=PhraseSegmenter(**segmenter_params()),
            )
            live.start()
        return live

    def apply_session(session: dict) -> None:
        """Слить обновление в конфигурацию (обе формы: GA-вложенная и beta-плоская)."""
        audio_in = (session.get("audio") or {}).get("input") or {}
        fmt = audio_in.get("format")
        if isinstance(fmt, dict):
            if fmt.get("type") not in (None, "audio/pcm"):
                raise ValueError(f"unsupported audio format {fmt.get('type')!r}: only audio/pcm")
            rate = int(fmt.get("rate") or 24000)
            if rate not in (24000, 16000):
                raise ValueError("unsupported sample rate: 24000 (standard) or 16000")
            cfg["rate"] = rate
        if session.get("input_audio_format") not in (None, "pcm16"):
            raise ValueError("unsupported input_audio_format: only pcm16")
        tr = audio_in.get("transcription") or session.get("input_audio_transcription") or {}
        if tr.get("model"):
            cfg["model"] = tr["model"]
        if tr.get("language"):
            cfg["language"] = tr["language"]
        if "prompt" in tr:
            cfg["prompt"] = tr["prompt"]
        for holder in (audio_in, session):
            if "turn_detection" in holder:
                td = holder["turn_detection"]
                if td is None:
                    cfg["turn_detection"] = None
                else:
                    merged = dict(cfg["turn_detection"] or {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300})
                    merged.update({k: v for k, v in td.items() if v is not None})
                    merged.setdefault("silence_duration_ms", STREAM_SILENCE_MS)
                    cfg["turn_detection"] = merged
        if live is not None:
            live.segmenter.configure(**segmenter_params())

    await send({"type": "session.created", "session": session_object()})

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=STREAM_IDLE_TIMEOUT_SEC or None
                )
            except asyncio.TimeoutError:
                # Клиент завис или забыл закрыть сокет: слот надо освободить.
                # Молчание в микрофон сюда не попадает — аудио при этом идёт.
                await send_error(
                    f"no client events for {STREAM_IDLE_TIMEOUT_SEC:g} s; closing the session",
                    "session_idle_timeout",
                )
                break
            if message["type"] == "websocket.disconnect":
                break
            raw = message.get("text")
            if raw is None:
                await send_error("binary frames are not supported; send JSON events", "invalid_event")
                continue
            try:
                ev = json.loads(raw)
                etype = ev["type"]
            except Exception:
                await send_error("event must be a JSON object with a 'type' field", "invalid_event")
                continue
            ceid = ev.get("event_id")

            if etype == "input_audio_buffer.append":
                try:
                    pcm = np.frombuffer(base64.b64decode(ev["audio"]), dtype="<i2").astype(np.float32) / 32768.0
                except Exception:
                    await send_error("'audio' must be base64-encoded PCM16", "invalid_audio", ceid)
                    continue
                if cfg["rate"] == 24000:
                    pcm = resampler.feed(pcm)
                if len(pcm):
                    await ensure_live().feed(pcm)

            elif etype in ("session.update", "transcription_session.update"):
                if etype.startswith("transcription_session"):
                    beta = True
                try:
                    apply_session(ev.get("session") or {})
                except ValueError as e:
                    await send_error(str(e), "invalid_value", ceid)
                    continue
                await send(
                    {
                        "type": "transcription_session.updated" if beta else "session.updated",
                        "session": session_object(),
                    }
                )

            elif etype == "input_audio_buffer.commit":
                if live is None or not live.segmenter.has_speech:
                    await send_error(
                        "Error committing input audio buffer: buffer has no speech",
                        "input_audio_buffer_commit_empty",
                        ceid,
                    )
                else:
                    await live.commit()

            elif etype == "input_audio_buffer.clear":
                if live is not None:
                    live.segmenter.clear()
                await send({"type": "input_audio_buffer.cleared"})

            else:
                await send_error(
                    f"event type {etype!r} is not supported: this server implements transcription only",
                    "unknown_event",
                    ceid,
                )

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[realtime] session failed")
    finally:
        if live is not None:
            await live.finish()
        try:
            await websocket.close()
        except Exception:
            pass
        await release_session()


# ---------------------------------------------------------------------------
# Описание протокола для Swagger. FastAPI не кладёт WebSocket-руты в OpenAPI,
# поэтому рядом висит обычный GET: отдаёт протокол машиночитаемо и
# документирует его на /docs.
# ---------------------------------------------------------------------------


class RealtimeProtocol(BaseModel):
    endpoint: str
    transport: str
    standard: str
    auth: dict
    audio: dict
    query_params: dict
    client_events: dict
    server_events: dict
    extensions: dict
    limits: dict
    notes: list[str]


@router.get(
    "/v1/realtime",
    response_model=RealtimeProtocol,
    tags=["Streaming"],
    summary="Protocol description for the live dictation WebSocket",
    description="""
Describes the **WebSocket** at the same path, `ws(s)://<host>/v1/realtime`, which
OpenAPI cannot represent directly. It speaks the **OpenAI Realtime API in
transcription mode** (GA dialect), so an OpenAI client works by changing its
base URL:

```python
from openai import AsyncOpenAI   # pip install "openai[realtime]"

client = AsyncOpenAI(api_key=AUTH_TOKEN, base_url="http://localhost:9007/v1")
async with client.realtime.connect(extra_query={"intent": "transcription"}) as conn:
    await conn.session.update(session={
        "type": "transcription",
        "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": "v3_e2e_ctc", "language": "ru"},
            "turn_detection": {"type": "server_vad", "silence_duration_ms": 600},
        }},
    })
    await conn.input_audio_buffer.append(audio=base64_pcm16)   # while speaking
    async for event in conn:
        if event.type == "conversation.item.input_audio_transcription.delta":
            print(event.delta, end="")
        elif event.type == "conversation.item.input_audio_transcription.completed":
            print("\\nFINAL:", event.transcript)
```

`POST /v1/audio/transcriptions` with `stream=true` streams the **response** for a
file you already have; this endpoint streams the **input** — live dictation.

### What a delta is

A delta appends and cannot be taken back, so only **stabilised words** are sent:
a word is committed once two consecutive drafts agree on it and it is at least
500 ms away from the end of the audio. `completed.transcript` is authoritative —
if the model revised words already sent (numbers being normalised, a name fixed
by later context), deltas for that phrase stop and the client replaces the
accumulated text with the transcript.

### Extensions (opt-in, invisible to standard clients)

- `?tentative=true` — `x.transcription.tentative` on every draft tick: the
  committed text of the open phrase, its unstable tail, and whether committed
  words were rewritten. Use it to render the part that may still change.
- `?emotions=true` — `phrase.emotion` after each phrase.
- `?partials=false` — no drafts: one delta per phrase with the whole text.
""",
)
async def realtime_protocol() -> RealtimeProtocol:
    return RealtimeProtocol(
        endpoint="/v1/realtime",
        transport="websocket",
        standard="OpenAI Realtime API, transcription sessions (GA); beta event names are accepted",
        auth={
            "header": "Authorization: Bearer <AUTH_TOKEN>",
            "browser_subprotocols": ["realtime", "openai-insecure-api-key.<AUTH_TOKEN>"],
            "query": "token=<AUTH_TOKEN>",
            "bad_key": "handshake is refused with HTTP 403",
            "busy": "socket opens, an error event (code rate_limit_exceeded) is sent, close code 1013",
        },
        audio={
            "event": "input_audio_buffer.append with base64 in `audio`",
            "encoding": "PCM signed 16-bit little-endian, mono",
            "sample_rates": {"24000": "standard, resampled on the server", "16000": 'extension: session format {"type":"audio/pcm","rate":16000}'},
            "not_supported": "binary frames, G.711",
        },
        query_params={
            "intent": "transcription (optional, accepted for compatibility)",
            "model": "model from GIGAAM_MODELS; any other name selects the default model",
            "partials": "true/false — drafts while a phrase is open; default: on when the model runs on CUDA",
            "tentative": "true — enable the x.transcription.tentative extension event",
            "emotions": "true — enable the phrase.emotion extension event",
            "token": "AUTH_TOKEN, when neither the header nor the subprotocol can be used",
        },
        client_events={
            "session.update": "session.type=transcription; audio.input.format.rate, transcription.model/language, turn_detection {threshold, silence_duration_ms} or null (cut phrases only on commit)",
            "input_audio_buffer.append": "audio chunk, base64",
            "input_audio_buffer.commit": "close the current phrase now; error input_audio_buffer_commit_empty if it has no speech",
            "input_audio_buffer.clear": "drop the audio of the open phrase",
            "<anything else>": "error event with code unknown_event; the socket stays open",
        },
        server_events={
            "session.created / session.updated": "session object with the effective settings",
            "input_audio_buffer.speech_started": "audio_start_ms, item_id",
            "input_audio_buffer.speech_stopped": "audio_end_ms, item_id",
            "input_audio_buffer.committed": "item_id, previous_item_id — a phrase was cut and goes to the final pass",
            "conversation.item.added / conversation.item.done": "user message item for the phrase",
            "conversation.item.input_audio_transcription.delta": "item_id, content_index, delta — stabilised words only",
            "conversation.item.input_audio_transcription.completed": "item_id, content_index, transcript (authoritative), usage {type: duration, seconds}",
            "error": "error {type, code, message, param, event_id}",
        },
        extensions={
            "x.transcription.tentative": "item_id, committed, tentative, rewrite, changed_from — with ?tentative=true",
            "phrase.emotion": "item_id, dominant, emotions {label: prob} — with ?emotions=true",
        },
        limits={
            "max_sessions": STREAM_MAX_SESSIONS,
            "max_queued_phrases": STREAM_MAX_QUEUED_PHRASES,
            "default_silence_ms": STREAM_SILENCE_MS,
            "soft_cut_sec": STREAM_SOFT_CUT_SEC,
            "max_phrase_sec": STREAM_MAX_PHRASE_SEC,
            "min_phrase_sec": STREAM_MIN_PHRASE_SEC,
            "idle_timeout_sec": STREAM_IDLE_TIMEOUT_SEC,
            "partial_interval_ms": STREAM_PARTIAL_INTERVAL_MS,
            "partial_guard_ms": STREAM_PARTIAL_GUARD_MS,
        },
        notes=[
            "Final text is per phrase: the model is offline and needs a complete segment.",
            "Speech without pauses can go on indefinitely: a phrase longer than soft_cut_sec is closed at the boundary of the last stabilised word (its text comes from the pass that saw context on both sides), and the rest continues as the next item. max_phrase_sec is only a hard safety cut, used when drafts are off or nothing could be stabilised.",
            "A session with no client events for idle_timeout_sec gets an error (code session_idle_timeout) and is closed.",
            "A phrase that turns out to be noise is closed with an empty transcript.",
            "Drafts are always computed with the CTC model; the final uses the session model.",
            "There is no end-of-session event: close the socket (send input_audio_buffer.commit first to flush the last phrase).",
        ],
    )
