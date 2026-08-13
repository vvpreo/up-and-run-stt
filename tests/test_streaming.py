"""
SSE-стриминг: stream=true на /v1/audio/transcriptions отдаёт поток событий
OpenAI-формата (transcript.text.delta ... transcript.text.done).
"""

import json
import time

import requests

TIMEOUT = 300


def _stream_events(base_url, headers, path, extra=None):
    events, first_delta_at = [], None
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        with requests.post(
            f"{base_url}/v1/audio/transcriptions",
            headers=headers,
            files={"file": ("long.ogg", f)},
            data={"stream": "true", **(extra or {})},
            stream=True,
            timeout=TIMEOUT,
        ) as r:
            assert r.status_code == 200, r.text
            assert r.headers["content-type"].startswith("text/event-stream")
            for line in r.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                ev = json.loads(line[5:])
                if ev["type"] == "transcript.text.delta" and first_delta_at is None:
                    first_delta_at = time.perf_counter() - t0
                events.append(ev)
    return events, first_delta_at, time.perf_counter() - t0


def test_sse_stream_long_audio(base_url, auth_headers, sample_path):
    events, first_delta_at, total = _stream_events(base_url, auth_headers, sample_path)

    deltas = [e for e in events if e["type"] == "transcript.text.delta"]
    done = [e for e in events if e["type"] == "transcript.text.done"]

    # длинный файл (137 с) при 12-с чанках должен дать несколько дельт
    assert len(deltas) >= 3, f"expected multiple deltas, got {len(deltas)}"
    assert len(done) == 1

    full = done[0]["text"]
    assert "транскрипцию" in full.lower()
    # текст done — конкатенация дельт (с точностью до пробелов)
    assert full == " ".join(d["delta"].strip() for d in deltas).strip()
    assert done[0]["usage"]["type"] == "duration"

    # прогрессивность: первая дельта заметно раньше конца стрима
    assert first_delta_at is not None
    assert first_delta_at < total * 0.6, (
        f"first delta at {first_delta_at:.1f}s of {total:.1f}s — not progressive"
    )


def test_sse_stream_short_audio(base_url, auth_headers, short_wav):
    events, _, _ = _stream_events(base_url, auth_headers, short_wav)
    done = [e for e in events if e["type"] == "transcript.text.done"]
    assert len(done) == 1
    assert "транскрипцию" in done[0]["text"].lower()


def test_stream_false_is_regular_json(base_url, auth_headers, short_wav):
    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            headers=auth_headers,
            files={"file": ("a.wav", f)},
            data={"stream": "false", "response_format": "json"},
            timeout=TIMEOUT,
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert "text" in r.json()
