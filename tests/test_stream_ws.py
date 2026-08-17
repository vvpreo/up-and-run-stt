"""
Потоковый приём аудио: WebSocket /stt/stream.

Проверяется то, чего нет у обычных эндпоинтов: аудио уходит по мере
«произнесения», а текст возвращается пофразно ещё до конца передачи.
Клиент здесь намеренно тупой — шлёт PCM-кадры и не решает, где границы фраз.

pytest-asyncio не используется: асинхронный клиент запускается через
asyncio.run() внутри обычного синхронного теста, чтобы не тащить лишнюю
зависимость ради четырёх тестов.
"""

import asyncio
import json
import subprocess
from urllib.parse import urlparse

import pytest
import requests
import websockets

TIMEOUT = 120


@pytest.fixture(scope="module")
def ws_url(base_url):
    """base_url (http://host:port) -> ws://host:port/stt/stream"""
    u = urlparse(base_url)
    scheme = "wss" if u.scheme == "https" else "ws"
    return f"{scheme}://{u.netloc}/stt/stream"


@pytest.fixture(scope="module")
def pcm16(sample_path, tmp_path_factory):
    """Первые 20 с сэмпла как сырой PCM s16le 16 кГц моно — формат эндпоинта."""
    out = tmp_path_factory.mktemp("stream") / "audio.raw"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(sample_path),
         "-t", "20", "-ar", "16000", "-ac", "1", "-f", "s16le", str(out)],
        check=True,
    )
    return out.read_bytes()


async def _stream(url, token, pcm, *, chunk_ms=100, realtime=False, commit=False):
    """Прогнать PCM через сокет и собрать события сервера."""
    if token:
        url = f"{url}?token={token}"
    events = []
    async with websockets.connect(url, max_size=None) as ws:
        step = 16000 * 2 * chunk_ms // 1000  # байт на кадр

        async def receive():
            async for raw in ws:
                event = json.loads(raw)
                events.append(event)
                if event["type"] == "transcript.text.done":
                    return

        task = asyncio.create_task(receive())
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i : i + step])
            if realtime:
                await asyncio.sleep(chunk_ms / 1000)
        if commit:
            await ws.send(json.dumps({"type": "commit"}))
        await ws.send(json.dumps({"type": "close"}))
        await asyncio.wait_for(task, timeout=TIMEOUT)
    return events


def test_stream_returns_phrases(ws_url, token, pcm16):
    """Поток PCM -> несколько фраз текста + итоговое событие."""
    events = asyncio.run(_stream(ws_url, token, pcm16))
    kinds = [e["type"] for e in events]

    assert kinds[0] == "session.created", kinds[:3]
    assert kinds[-1] == "transcript.text.done", kinds[-3:]

    deltas = [e for e in events if e["type"] == "transcript.text.delta"]
    assert deltas, f"ни одной фразы не распознано: {kinds}"
    assert all(d["delta"].strip() for d in deltas), "пустая фраза в дельтах"

    # Нумерация фраз последовательная — по ней клиент собирает текст
    assert [d["seq"] for d in deltas] == list(range(1, len(deltas) + 1))

    done = events[-1]
    assert done["phrases"] == len(deltas)
    assert done["text"].strip(), "итоговый текст пуст"
    # Каждая дельта должна попасть в итог
    for d in deltas:
        assert d["delta"] in done["text"]


def test_stream_recognizes_speech(ws_url, token, pcm16):
    """Распознаётся именно речь из сэмпла, а не что попало."""
    events = asyncio.run(_stream(ws_url, token, pcm16))
    text = events[-1]["text"].lower()
    assert "проверяем" in text, f"ожидали слово из сэмпла, получили: {text[:200]}"


def test_stream_emits_text_before_the_end(ws_url, token, pcm16):
    """
    Главное свойство режима: первая фраза приходит ДО конца передачи.

    Аудио отправляется в темпе реального времени (как живой микрофон);
    первая дельта обязана прийти заметно раньше, чем уйдёт последний кадр.
    """
    async def run():
        url = f"{ws_url}?token={token}" if token else ws_url
        first_delta_at = None
        sent_all_at = None
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        async with websockets.connect(url, max_size=None) as ws:
            async def receive():
                nonlocal first_delta_at
                async for raw in ws:
                    e = json.loads(raw)
                    if e["type"] == "transcript.text.delta" and first_delta_at is None:
                        first_delta_at = loop.time() - t0
                    elif e["type"] == "transcript.text.done":
                        return

            task = asyncio.create_task(receive())
            step = 16000 * 2 // 10  # 100 мс
            for i in range(0, len(pcm16), step):
                await ws.send(pcm16[i : i + step])
                await asyncio.sleep(0.1)
            sent_all_at = loop.time() - t0
            await ws.send(json.dumps({"type": "close"}))
            await asyncio.wait_for(task, timeout=TIMEOUT)
        return first_delta_at, sent_all_at

    first, sent_all = asyncio.run(run())
    assert first is not None, "не пришло ни одной фразы"
    assert first < sent_all - 3, (
        f"первая фраза пришла на {first:.1f}с, передача кончилась на {sent_all:.1f}с — "
        "текст не опережает передачу, значит это не потоковый режим"
    )


def test_stream_rejects_bad_token(ws_url, token):
    """
    Неверный токен обрывает САМО рукопожатие: HTTP 403, сокет не поднимается.

    Именно отказ на апгрейде, а не закрытие уже открытого соединения —
    неаутентифицированный клиент не должен получить рабочий сокет даже на
    мгновение.
    """
    if not token:
        pytest.skip("авторизация выключена на этом инстансе")

    async def run():
        with pytest.raises(websockets.exceptions.InvalidStatus) as exc:
            async with websockets.connect(f"{ws_url}?token=obviously-wrong") as ws:
                await ws.recv()
        return exc.value

    err = asyncio.run(run())
    assert err.response.status_code == 403, err


def test_stream_without_token_is_refused(ws_url, token):
    """Отсутствующий токен — тоже 403, а не молчаливый пропуск."""
    if not token:
        pytest.skip("авторизация выключена на этом инстансе")

    async def run():
        with pytest.raises(websockets.exceptions.InvalidStatus) as exc:
            async with websockets.connect(ws_url) as ws:
                await ws.recv()
        return exc.value

    assert asyncio.run(run()).response.status_code == 403


def test_protocol_endpoint_describes_websocket(base_url, auth_headers):
    """GET на том же пути отдаёт протокол — WebSocket в OpenAPI не попадает."""
    r = requests.get(f"{base_url}/stt/stream", headers=auth_headers, timeout=30)
    assert r.status_code == 200, r.text
    spec = r.json()

    assert spec["transport"] == "websocket"
    assert spec["audio_format"]["sample_rate"] == 16000
    assert spec["audio_format"]["encoding"] == "pcm_s16le"
    # Интегратору нужно знать и про события, и про лимиты
    assert "transcript.text.delta" in spec["server_events"]
    assert "transcript.text.done" in spec["server_events"]
    assert spec["limits"]["max_sessions"] >= 1


def test_health_reports_stream_sessions(base_url):
    """/health показывает счётчик живых сессий отдельно от pending_requests."""
    health = requests.get(f"{base_url}/health", timeout=10).json()
    assert "stream_sessions" in health
    assert "stream_max_sessions" in health
    assert health["stream_sessions"] >= 0
