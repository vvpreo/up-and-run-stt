"""
Живая диктовка: WS /v1/realtime (OpenAI Realtime API, режим транскрипции) —
единственный потоковый эндпоинт сервиса.

Проверяется проводной протокол GA-диалекта: session.created / session.update,
input_audio_buffer.append (base64 PCM16 24 кГц), события фразы и их порядок,
авторизация заголовком и через subprotocol. Отдельный тест гоняет настоящий
клиент из официального SDK `openai`, если он установлен.
"""

import asyncio
import base64
import json
import subprocess
from urllib.parse import urlparse

import pytest
import requests
import websockets

T = "conversation.item.input_audio_transcription."
TIMEOUT = 120


@pytest.fixture(scope="module")
def rt_url(base_url):
    u = urlparse(base_url)
    return f"{'wss' if u.scheme == 'https' else 'ws'}://{u.netloc}/v1/realtime?intent=transcription"


@pytest.fixture(scope="module")
def pcm24(sample_path, tmp_path_factory):
    """Первые 20 с сэмпла: PCM s16le 24 кГц моно — формат Realtime API."""
    out = tmp_path_factory.mktemp("rt") / "audio24.raw"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(sample_path),
         "-t", "20", "-ar", "24000", "-ac", "1", "-f", "s16le", str(out)],
        check=True,
    )
    return out.read_bytes()


def _headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _run(url, token, pcm, *, realtime=False, update=None, tail=None):
    """Прогнать аудио и собрать события, пока сервер не замолчит на 8 с после конца передачи."""
    events = []
    async with websockets.connect(url, additional_headers=_headers(token), max_size=None) as ws:
        events.append(json.loads(await ws.recv()))
        last_event_at = asyncio.get_running_loop().time()

        # Читать надо параллельно с отправкой: иначе очередь входящих у клиента
        # переполняется и он перестаёт отвечать на ping сервера.
        async def receive():
            nonlocal last_event_at
            async for raw in ws:
                events.append(json.loads(raw))
                last_event_at = asyncio.get_running_loop().time()

        reader = asyncio.create_task(receive())
        await ws.send(json.dumps({"type": "session.update", "session": update or {
            "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "gpt-4o-transcribe", "language": "ru"},
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
            }},
        }}))
        step = 24000 * 2 // 10  # 100 мс
        for i in range(0, len(pcm), step):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i + step]).decode(),
            }))
            if realtime:
                await asyncio.sleep(0.1)
        for extra in tail or []:
            await ws.send(json.dumps(extra))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        last_event_at = asyncio.get_running_loop().time()
        while asyncio.get_running_loop().time() - last_event_at < 8:
            await asyncio.sleep(0.5)
        reader.cancel()
    return events


def test_session_handshake_and_update(rt_url, token, pcm24):
    events = asyncio.run(_run(rt_url, token, pcm24[:24000 * 2]))
    created = events[0]
    assert created["type"] == "session.created" and created["event_id"]
    s = created["session"]
    assert s["type"] == "transcription" and s["object"] == "realtime.transcription_session" and s["id"]
    assert s["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert s["audio"]["input"]["turn_detection"]["type"] == "server_vad"

    updated = next(e for e in events if e["type"] == "session.updated")
    i = updated["session"]["audio"]["input"]
    assert i["transcription"]["model"] == "gpt-4o-transcribe"  # чужое имя принимается
    assert i["transcription"]["language"] == "ru"
    assert i["turn_detection"]["silence_duration_ms"] == 500


def test_turn_events_in_order_and_transcript(rt_url, token, pcm24):
    events = asyncio.run(_run(rt_url, token, pcm24))
    assert all(e.get("event_id") for e in events), "у каждого серверного события есть event_id"
    assert not [e for e in events if e["type"] == "error"
                and e["error"]["code"] != "input_audio_buffer_commit_empty"], events

    # Фраза из одного шума закрывается completed с пустым transcript — их пропускаем
    completed = [e for e in events if e["type"] == T + "completed" and e["transcript"]]
    assert completed, [e["type"] for e in events]
    text = " ".join(e["transcript"] for e in completed).lower()
    assert "проверяем" in text, text[:200]

    types = [e["type"] for e in events]
    for c in completed:
        item = c["item_id"]
        assert c["content_index"] == 0 and c["usage"]["type"] == "duration"
        idx = lambda t: next(i for i, e in enumerate(events) if e["type"] == t and e.get("item_id") == item)
        # committed -> (delta)* -> completed для одной и той же фразы
        assert idx("input_audio_buffer.committed") < idx(T + "completed")
        deltas = [e for e in events if e["type"] == T + "delta" and e["item_id"] == item]
        assert deltas, "на фразу приходит хотя бы одна дельта"
        joined = "".join(d["delta"] for d in deltas)
        # Дельты — только дописывание; финал либо равен их склейке, либо
        # авторитетно заменяет её (перезапись чисел и т.п.)
        assert joined == c["transcript"] or c["transcript"], (joined, c["transcript"])
        assert all(events.index(d) < events.index(c) for d in deltas)

    assert "conversation.item.added" in types and "conversation.item.done" in types
    # Цепочка previous_item_id связывает фразы по порядку
    commits = [e for e in events if e["type"] == "input_audio_buffer.committed"]
    assert commits[0]["previous_item_id"] is None
    for prev, cur in zip(commits, commits[1:]):
        assert cur["previous_item_id"] == prev["item_id"]


def test_server_vad_events_carry_timing(rt_url, token, pcm24):
    events = asyncio.run(_run(rt_url, token, pcm24, realtime=True))
    started = [e for e in events if e["type"] == "input_audio_buffer.speech_started"]
    stopped = [e for e in events if e["type"] == "input_audio_buffer.speech_stopped"]
    assert started and stopped
    assert all(isinstance(e["audio_start_ms"], int) and e["item_id"] for e in started)
    assert stopped[0]["audio_end_ms"] > started[0]["audio_start_ms"]
    assert stopped[0]["item_id"] == started[0]["item_id"]


def test_unsupported_event_is_an_error_not_a_close(rt_url, token, pcm24):
    events = asyncio.run(_run(
        rt_url, token, pcm24[:24000 * 2 * 6],
        tail=[{"type": "response.create", "event_id": "evt_client_1"}],
    ))
    err = next(e for e in events if e["type"] == "error" and e["error"]["code"] == "unknown_event")
    assert err["error"]["type"] == "invalid_request_error"
    assert err["error"]["event_id"] == "evt_client_1"
    # Сокет остался жив: после ошибки пришла фраза
    assert [e for e in events if e["type"] == T + "completed"]


def test_key_via_browser_subprotocol(rt_url, token):
    """Браузер не может слать заголовки: ключ идёт в subprotocol, сервер выбирает `realtime`."""
    async def go():
        protos = ["realtime"] + ([f"openai-insecure-api-key.{token}"] if token else [])
        async with websockets.connect(rt_url, subprotocols=protos) as ws:
            assert ws.subprotocol == "realtime"
            return json.loads(await ws.recv())
    assert asyncio.run(go())["type"] == "session.created"


def test_bad_key_is_refused_at_handshake(rt_url, token):
    if not token:
        pytest.skip("authorization is disabled on this instance")

    async def go():
        async with websockets.connect(rt_url, additional_headers={"Authorization": "Bearer wrong"}):
            pass
    with pytest.raises(websockets.exceptions.InvalidStatus) as e:
        asyncio.run(go())
    assert e.value.response.status_code == 403


def test_official_openai_sdk_client(base_url, token, pcm24):
    """
    Настоящий клиент из SDK `openai` (GA Realtime) против нашего сервера.

    SDK в зависимости проекта не входит (он тянет старую версию websockets и
    понизил бы её в рантайм-образе), поэтому тест идёт только там, где SDK
    поставлен отдельно:
        uv run --with "openai[realtime]" pytest tests/test_realtime_ws.py
    """
    openai = pytest.importorskip("openai", reason="openai SDK is not installed")

    async def go():
        client = openai.AsyncOpenAI(api_key=token or "none", base_url=f"{base_url}/v1")
        finals, deltas = [], []
        async with client.realtime.connect(extra_query={"intent": "transcription"}) as conn:
            await conn.session.update(session={
                "type": "transcription",
                "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "gpt-4o-transcribe", "language": "ru"},
                    "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
                }},
            })
            for i in range(0, len(pcm24), 4800):
                await conn.input_audio_buffer.append(audio=base64.b64encode(pcm24[i:i + 4800]).decode())
            await conn.input_audio_buffer.commit()
            try:
                while True:
                    event = await asyncio.wait_for(conn.recv(), timeout=8)
                    if event.type == T + "delta":
                        deltas.append(event.delta)
                    elif event.type == T + "completed":
                        finals.append(event.transcript)
            except asyncio.TimeoutError:
                pass
        return finals, deltas

    finals, deltas = asyncio.run(go())
    assert finals and deltas
    assert "проверяем" in " ".join(finals).lower()


def test_tentative_extension_is_opt_in(rt_url, token, pcm24):
    """Хвост черновика — расширение: по умолчанию его нет, с ?tentative=true он есть."""
    plain = asyncio.run(_run(rt_url, token, pcm24[:24000 * 2 * 8], realtime=True))
    assert not [e for e in plain if e["type"].startswith("x.")], "расширения не должны приходить без запроса"

    ext = asyncio.run(_run(rt_url + "&tentative=true", token, pcm24[:24000 * 2 * 8], realtime=True))
    if not [e for e in ext if e["type"] == T + "delta"]:
        pytest.skip("no speech recognised in the first seconds")
    # Черновики идут только там, где они включены (на GPU по умолчанию)
    tent = [e for e in ext if e["type"] == "x.transcription.tentative"]
    for e in tent:
        assert e["item_id"] and "committed" in e and "tentative" in e and "rewrite" in e


def test_default_behaviour_is_strictly_standard(rt_url, token, pcm24):
    """
    Без параметров-расширений сервер шлёт только события стандарта OpenAI:
    никаких x.*, phrase.emotion и прочего — на это закладываются чужие клиенты.
    """
    standard = {
        "session.created", "session.updated", "error",
        "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed", "input_audio_buffer.cleared",
        "conversation.item.added", "conversation.item.done",
        T + "delta", T + "completed", T + "failed",
    }
    events = asyncio.run(_run(rt_url, token, pcm24, realtime=True))
    extra = {e["type"] for e in events} - standard
    assert not extra, f"нестандартные события без запроса: {extra}"


def test_text_arrives_before_the_audio_ends(rt_url, token, pcm24):
    """
    Главное свойство режима: текст приходит ДО конца передачи. Аудио идёт в
    темпе реального времени; первая завершённая фраза обязана прийти заметно
    раньше последнего кадра.
    """
    async def go():
        loop = asyncio.get_running_loop()
        first = None
        async with websockets.connect(rt_url, additional_headers=_headers(token), max_size=None) as ws:
            t0 = loop.time()

            async def receive():
                nonlocal first
                async for raw in ws:
                    e = json.loads(raw)
                    if e["type"] == T + "completed" and e["transcript"] and first is None:
                        first = loop.time() - t0

            task = asyncio.create_task(receive())
            step = 24000 * 2 // 10
            for i in range(0, len(pcm24), step):
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(pcm24[i:i + step]).decode()}))
                await asyncio.sleep(0.1)
            sent_all = loop.time() - t0
            task.cancel()
        return first, sent_all

    first, sent_all = asyncio.run(go())
    assert first is not None, "не пришло ни одной фразы"
    assert first < sent_all - 3, f"первая фраза на {first:.1f} с, передача кончилась на {sent_all:.1f} с"


def test_missing_key_is_refused_at_handshake(rt_url, token):
    if not token:
        pytest.skip("authorization is disabled on this instance")

    async def go():
        async with websockets.connect(rt_url):
            pass
    with pytest.raises(websockets.exceptions.InvalidStatus) as e:
        asyncio.run(go())
    assert e.value.response.status_code == 403


def test_partials_control_how_deltas_arrive(rt_url, token, pcm24):
    """
    ?partials=false — на фразу одна дельта с целой фразой (как whisper-1);
    ?partials=true — слова приходят по мере стабилизации, дельт на длинную
    фразу несколько, а их склейка — начало финального текста либо финал
    авторитетно её заменяет.
    """
    off = asyncio.run(_run(rt_url + "&partials=false", token, pcm24, realtime=True))
    for c in [e for e in off if e["type"] == T + "completed" and e["transcript"]]:
        deltas = [e["delta"] for e in off if e["type"] == T + "delta" and e["item_id"] == c["item_id"]]
        assert deltas == [c["transcript"]], (deltas, c["transcript"])

    on = asyncio.run(_run(rt_url + "&partials=true", token, pcm24, realtime=True))
    completed = [e for e in on if e["type"] == T + "completed" and e["transcript"]]
    assert completed
    per_item = [
        [e["delta"] for e in on if e["type"] == T + "delta" and e["item_id"] == c["item_id"]]
        for c in completed
    ]
    assert max(len(d) for d in per_item) > 1, "ни одна фраза не пришла по словам"
    # Дельты приходят раньше completed своей фразы
    for c in completed:
        ci = on.index(c)
        assert all(on.index(e) < ci for e in on if e["type"] == T + "delta" and e["item_id"] == c["item_id"])


def test_phrase_emotions_are_an_opt_in_extension(base_url, rt_url, token, pcm24):
    if not requests.get(f"{base_url}/health", timeout=10).json().get("emotions_enabled"):
        pytest.skip("emotions are disabled on this instance")

    events = asyncio.run(_run(rt_url + "&emotions=true&partials=false", token, pcm24))
    items = {e["item_id"] for e in events if e["type"] == T + "completed" and e["transcript"]}
    emos = [e for e in events if e["type"] == "phrase.emotion"]
    assert emos, "эмоции запрошены, но не пришли"
    for e in emos:
        assert e["item_id"] in items
        assert e["dominant"] in e["emotions"]
        assert abs(sum(e["emotions"].values()) - 1.0) < 0.05
        # Эмоция не обгоняет текст своей фразы
        done = next(x for x in events if x["type"] == T + "completed" and x["item_id"] == e["item_id"])
        assert events.index(done) < events.index(e)


def test_protocol_description_endpoint(base_url, auth_headers):
    """GET на том же пути отдаёт протокол — WebSocket в OpenAPI не попадает."""
    r = requests.get(f"{base_url}/v1/realtime", headers=auth_headers, timeout=30)
    assert r.status_code == 200, r.text
    spec = r.json()
    assert spec["transport"] == "websocket" and "OpenAI Realtime" in spec["standard"]
    assert "24000" in spec["audio"]["sample_rates"]
    assert T + "delta" in spec["server_events"] and T + "completed" in spec["server_events"]
    assert "x.transcription.tentative" in spec["extensions"]
    assert spec["limits"]["max_sessions"] >= 1


def test_old_native_endpoint_is_gone(base_url, auth_headers):
    """/stt/stream удалён: единственный живой эндпоинт — /v1/realtime."""
    assert requests.get(f"{base_url}/stt/stream", headers=auth_headers, timeout=10).status_code == 404


def test_health_reports_stream_sessions(base_url):
    """/health показывает счётчик живых сессий отдельно от pending_requests."""
    health = requests.get(f"{base_url}/health", timeout=10).json()
    assert health["stream_sessions"] >= 0 and health["stream_max_sessions"] >= 1


@pytest.fixture(scope="module")
def pcm24_long(sample_path, tmp_path_factory):
    """
    45 с плотной речи, PCM 24 кГц: из сэмпла вырезана тишина длиннее 0.3 с.
    Вместе с выключенным серверным VAD это речь без остановок. (Исходный сэмпл
    с его многосекундными паузами для этого не годится: на тишине модель
    бредит, слова не стабилизируются, и срабатывает жёсткий предохранитель.)
    """
    out = tmp_path_factory.mktemp("rt") / "long24.raw"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(sample_path),
         "-af", "silenceremove=stop_periods=-1:stop_duration=0.3:stop_threshold=-38dB",
         "-t", "45", "-ar", "24000", "-ac", "1", "-f", "s16le", str(out)],
        check=True,
    )
    return out.read_bytes()


NO_VAD = {
    "type": "transcription",
    "audio": {"input": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": {"language": "ru"},
        # Сервер не режет по паузам: поток непрерывен, как речь без остановок
        "turn_detection": None,
    }},
}


def test_nonstop_speech_is_cut_at_word_boundaries(base_url, rt_url, token, pcm24_long):
    """
    Речь без единой закрывающей паузы (серверный VAD выключен): фразы
    закрываются мягким срезом по границе зафиксированного слова задолго до
    жёсткого предохранителя, текст идёт непрерывно и не теряется. Мягкий срез
    опирается на черновики, поэтому проверяется там, где они включены.
    """
    auth = {"Authorization": f"Bearer {token}"} if token else {}
    limits = requests.get(f"{base_url}/v1/realtime", headers=auth, timeout=10).json()["limits"]
    if requests.get(f"{base_url}/health", timeout=10).json().get("device") != "cuda":
        pytest.skip("drafts are off by default on CPU; the soft cut needs them")

    events = asyncio.run(_run(rt_url, token, pcm24_long, realtime=True, update=NO_VAD))
    done = [e for e in events if e["type"] == T + "completed" and e["transcript"]]
    durations = [e["usage"]["seconds"] for e in done]
    assert len(done) >= 3, f"45 с без пауз должны разбиться на несколько фраз: {durations}"
    # Срез происходит около порога мягкого среза, до жёсткого предохранителя не доходит
    assert max(durations) < limits["max_phrase_sec"] - 1, durations
    # Мягкий срез закрывает не меньше половины накопленного (а копится от soft_cut_sec)
    assert max(durations) >= limits["soft_cut_sec"] * 0.5 - 0.5, durations
    # Текст на месте и без потерь на стыках: знакомые слова из разных мест записи
    text = " ".join(e["transcript"] for e in done).lower()
    for word in ("проверяем", "восемь", "десять"):
        assert word in text, (word, text[:300])
    assert not [e for e in events if e["type"] == "error"
                and e["error"]["code"] != "input_audio_buffer_commit_empty"]
