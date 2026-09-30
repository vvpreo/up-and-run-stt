#!/usr/bin/env python3
"""
Генератор нагрузки для замеров задержки и масштабирования по сессиям.

Два режима:

  ws    — N параллельных сессий WS /v1/realtime (OpenAI Realtime, транскрипция).
          Каждая льёт один и тот же wav (16 кГц моно s16le) кадрами по --frame-ms
          в реальном времени (--speed 1.0) или быстрее. Для каждой фразы
          считается задержка «сервер подтвердил паузу (speech_stopped) -> пришёл
          completed»: очередь к модели плюс инференс. Чтобы получить «замолчал ->
          текст», прибавьте паузу закрытия фразы (STREAM_SILENCE_MS, 600 мс).
          До 2026-09-30 стенд ходил в удалённый /stt/stream и мерил от конца
          аудио фразы — те цифры больше на ~0.1 с (docs/SPARK_BENCHMARK.md).
  http  — N параллельных POST /v1/audio/transcriptions одного файла; меряется
          время ответа каждого запроса -> RTF под нагрузкой.

Сессии стартуют со сдвигом --stagger, чтобы концы фраз не совпадали секунда
в секунду (иначе получается нереалистичный «залп»).

Зависимости: только стандартная библиотека + numpy + websockets + soundfile —
всё это есть в образе сервиса, поэтому скрипт можно гонять прямо в нём:

  docker run --rm --network host -v $PWD/benchmark:/bench -v $PWD/.tmp/audio:/audio \
      <image> python3 /bench/streaming/loadgen.py ws --audio /audio/x.wav --sessions 3

Результат — JSON в stdout (и в --out): перцентили задержек, overflow-события,
суммарные секунды аудио и wall time.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
import wave
from typing import Dict, List, Optional


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def _summary(values: List[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p50": _pct(values, 0.5),
        "p90": _pct(values, 0.9),
        "p95": _pct(values, 0.95),
        "max": round(max(values), 3),
    }


def _read_pcm(path: str, limit_sec: float) -> bytes:
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2, (
            "нужен wav 16 кГц моно s16le"
        )
        n = w.getnframes()
        if limit_sec > 0:
            n = min(n, int(limit_sec * 16000))
        return w.readframes(n)


# ------------------------------------------------------------------ ws mode

T_EVENT = "conversation.item.input_audio_transcription."


async def _ws_session(idx: int, args, pcm: bytes, results: dict) -> None:
    """Одна сессия WS /v1/realtime (OpenAI Realtime, транскрипция)."""
    import base64

    import websockets

    url = args.url + ("&" if "?" in args.url else "?") + "intent=transcription"
    if args.model:
        url += f"&model={args.model}"
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}

    frame_bytes = int(16000 * args.frame_ms / 1000) * 2
    frame_sec = args.frame_ms / 1000 / args.speed
    total_sec = len(pcm) / 2 / 16000

    rec = results[idx] = {
        "phrases": 0,
        "latency": [],        # пауза подтверждена (speech_stopped) -> текст фразы
        "overflow": 0,
        "errors": [],
        "chars": 0,
        "audio_sec": round(total_sec, 2),
    }

    await asyncio.sleep(idx * args.stagger)
    t_connect = time.perf_counter()
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        first = json.loads(await ws.recv())
        if first.get("type") != "session.created":
            rec["errors"].append(f"unexpected first event: {first}")
            return
        rec["connect_sec"] = round(time.perf_counter() - t_connect, 3)
        # wav уже 16 кГц — просим сервер не ресемплить (расширение format.rate)
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "transcription",
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000},
                                "transcription": {"language": "ru"}}},
        }}))

        t0 = time.perf_counter()  # момент отправки сэмпла с offset 0
        stopped_at: dict = {}      # item_id -> audio_end_ms
        pending: set = set()
        sent_all = asyncio.Event()
        texts: list = []

        async def sender():
            pos = 0
            i = 0
            while pos < len(pcm):
                target = t0 + i * frame_sec
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[pos : pos + frame_bytes]).decode(),
                }))
                pos += frame_bytes
                i += 1
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            sent_all.set()

        async def receiver():
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=6 if sent_all.is_set() else 600)
                except asyncio.TimeoutError:
                    return  # аудио кончилось и сервер молчит — фразы закончились
                ev = json.loads(raw)
                t = ev.get("type")
                if t == "input_audio_buffer.speech_stopped":
                    stopped_at[ev["item_id"]] = ev["audio_end_ms"]
                elif t == "input_audio_buffer.committed":
                    pending.add(ev["item_id"])
                elif t == T_EVENT + "completed":
                    pending.discard(ev["item_id"])
                    if ev.get("transcript"):
                        now = time.perf_counter()
                        end_ms = stopped_at.get(ev["item_id"])
                        if end_ms is not None:
                            # Сэмпл с offset s уходит в момент t0 + s/speed. audio_end_ms —
                            # позиция в аудио, где сервер подтвердил паузу и закрыл фразу.
                            rec["latency"].append(round(now - (t0 + end_ms / 1000 / args.speed), 3))
                        rec["phrases"] += 1
                        rec["chars"] += len(ev["transcript"])
                        texts.append(ev["transcript"])
                    if sent_all.is_set() and not pending:
                        return
                elif t == "error":
                    code = (ev.get("error") or {}).get("code")
                    if code == "server_overloaded":
                        rec["overflow"] += 1
                    elif code != "input_audio_buffer_commit_empty":
                        rec["errors"].append((ev.get("error") or {}).get("message"))

        await asyncio.gather(sender(), receiver())
        rec["wall_sec"] = round(time.perf_counter() - t0, 2)
        rec["text"] = " ".join(texts)


async def run_ws(args) -> dict:
    pcm = _read_pcm(args.audio, args.limit_sec)
    results: Dict[int, dict] = {}
    t_start = time.perf_counter()
    await asyncio.gather(*(_ws_session(i, args, pcm, results) for i in range(args.sessions)))
    wall = time.perf_counter() - t_start

    lat = [v for r in results.values() for v in r["latency"]]
    return {
        "mode": "ws",
        "sessions": args.sessions,
        "speed": args.speed,
        "audio_sec_per_session": results[0]["audio_sec"] if results else 0,
        "wall_sec": round(wall, 2),
        "phrases_total": sum(r["phrases"] for r in results.values()),
        "overflow_total": sum(r["overflow"] for r in results.values()),
        "errors": [e for r in results.values() for e in r["errors"]],
        "latency_end_to_text_sec": _summary(lat),
        # Время инференса в протоколе OpenAI не передаётся — смотреть GET /stats
        "inference_sec": {"n": 0},
        "per_session": {
            i: {k: v for k, v in r.items() if k not in ("latency", "text")}
            for i, r in results.items()
        },
        "texts": {i: r.get("text", "")[:200] for i, r in results.items()} if args.texts else None,
    }


# ---------------------------------------------------------------- http mode


def _post_file(url: str, token: str, path: str, model: Optional[str]) -> float:
    import urllib.request

    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n".encode() + data + b"\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\ntext\r\n".encode(),
    ]
    if model:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n".encode()
        )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        resp.read()
    return time.perf_counter() - t


async def run_http(args) -> dict:
    with wave.open(args.audio, "rb") as w:
        audio_sec = w.getnframes() / w.getframerate()
    loop = asyncio.get_running_loop()
    t_start = time.perf_counter()
    times = await asyncio.gather(
        *(
            loop.run_in_executor(None, _post_file, args.url, args.token, args.audio, args.model)
            for _ in range(args.sessions)
        )
    )
    wall = time.perf_counter() - t_start
    return {
        "mode": "http",
        "requests": args.sessions,
        "audio_sec": round(audio_sec, 2),
        "wall_sec": round(wall, 2),
        "request_sec": _summary(list(times)),
        "rtf_per_request": _summary([t / audio_sec for t in times]),
        "aggregate_speed_x_realtime": round(args.sessions * audio_sec / wall, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["ws", "http"])
    ap.add_argument("--url", default=None, help="ws://host:9007/v1/realtime или http://host:9007/v1/audio/transcriptions")
    ap.add_argument("--token", default="")
    ap.add_argument("--model", default=None)
    ap.add_argument("--audio", required=True, help="wav 16 кГц моно s16le")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--frame-ms", type=int, default=100)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = реальное время")
    ap.add_argument("--stagger", type=float, default=1.3, help="сдвиг старта сессий, с")
    ap.add_argument("--limit-sec", type=float, default=0, help="обрезать аудио, с (0 = всё)")
    ap.add_argument("--texts", action="store_true", help="включить в отчёт тексты")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.url is None:
        args.url = (
            "ws://localhost:9007/v1/realtime" if args.mode == "ws"
            else "http://localhost:9007/v1/audio/transcriptions"
        )

    result = asyncio.run(run_ws(args) if args.mode == "ws" else run_http(args))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
