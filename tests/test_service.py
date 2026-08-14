"""
Интеграционные тесты работающего сервиса (docker compose up -d).

Покрывают: /health, авторизацию, оба транскрипционных эндпоинта, реестр
моделей, лимит размера загрузки, живость event loop во время транскрипции
и доступность очереди (429 при переполнении — см. test_limits).
"""

import concurrent.futures
import io
import time

import requests

TIMEOUT = 300


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health(base_url):
    r = requests.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["engine"] == "gigaam"
    assert isinstance(data["models"], dict) and data["models"]
    assert data["default_model"] in data["models"]
    assert "pending_requests" in data


def test_test_page_served(base_url):
    r = requests.get(f"{base_url}/", timeout=10)
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


# ---------------------------------------------------------------------------
# Авторизация
# ---------------------------------------------------------------------------

def test_transcription_requires_token(base_url, short_wav, token):
    if not token:
        import pytest
        pytest.skip("AUTH_TOKEN not set — auth disabled")
    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": ("a.wav", f)},
            timeout=TIMEOUT,
        )
    assert r.status_code == 401

    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            headers={"Authorization": "Bearer wrong-token"},
            files={"file": ("a.wav", f)},
            timeout=TIMEOUT,
        )
    assert r.status_code == 401


def test_health_needs_no_token(base_url):
    assert requests.get(f"{base_url}/health", timeout=10).status_code == 200


# ---------------------------------------------------------------------------
# Транскрипция: OpenAI-совместимый эндпоинт
# ---------------------------------------------------------------------------

def test_openai_endpoint_json(base_url, auth_headers, short_wav):
    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            headers=auth_headers,
            files={"file": ("a.wav", f)},
            data={"response_format": "json"},
            timeout=TIMEOUT,
        )
    assert r.status_code == 200
    text = r.json()["text"]
    assert "транскрипцию" in text.lower()


def test_openai_endpoint_text(base_url, auth_headers, short_wav):
    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            headers=auth_headers,
            files={"file": ("a.wav", f)},
            data={"response_format": "text"},
            timeout=TIMEOUT,
        )
    assert r.status_code == 200
    assert "транскрипцию" in r.text.lower()


# ---------------------------------------------------------------------------
# Транскрипция: нативный эндпоинт
# ---------------------------------------------------------------------------

def test_native_endpoint(base_url, auth_headers, short_wav):
    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/stt/asr?output=json&language=ru",
            headers=auth_headers,
            files={"audio_file": ("a.wav", f)},
            timeout=TIMEOUT,
        )
    assert r.status_code == 200
    assert "транскрипцию" in r.json()["text"].lower()


# ---------------------------------------------------------------------------
# Реестр моделей
# ---------------------------------------------------------------------------

def test_model_selection(base_url, auth_headers, short_wav):
    models = requests.get(f"{base_url}/health", timeout=10).json()["models"]
    for model_name in models:
        with open(short_wav, "rb") as f:
            r = requests.post(
                f"{base_url}/v1/audio/transcriptions",
                headers=auth_headers,
                files={"file": ("a.wav", f)},
                data={"model": model_name, "response_format": "json"},
                timeout=TIMEOUT,
            )
        assert r.status_code == 200, model_name


def test_unknown_model_falls_back_to_default(base_url, auth_headers, short_wav):
    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            headers=auth_headers,
            files={"file": ("a.wav", f)},
            data={"model": "whisper-1", "response_format": "json"},
            timeout=TIMEOUT,
        )
    assert r.status_code == 200


def test_known_but_disabled_model_is_400(base_url, auth_headers, short_wav):
    enabled = requests.get(f"{base_url}/health", timeout=10).json()["models"]
    disabled = {"v3_ctc", "v3_rnnt", "v3_e2e_ctc", "v3_e2e_rnnt"} - set(enabled)
    if not disabled:
        import pytest
        pytest.skip("all known variants are enabled on this instance")
    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            headers=auth_headers,
            files={"file": ("a.wav", f)},
            data={"model": sorted(disabled)[0]},
            timeout=TIMEOUT,
        )
    assert r.status_code == 400
    # /v1/* отдаёт ошибки в формате OpenAI: {"error": {message, type, ...}}
    assert "not enabled" in r.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Ops: лимит размера, живость event loop
# ---------------------------------------------------------------------------

def test_oversized_upload_is_413(base_url, auth_headers):
    """Файл больше MAX_UPLOAD_MB (200 МБ по умолчанию) отклоняется без OOM."""
    blob = io.BytesIO(b"\x00" * (201 * 1024 * 1024))
    r = requests.post(
        f"{base_url}/v1/audio/transcriptions",
        headers=auth_headers,
        files={"file": ("big.wav", blob)},
        timeout=TIMEOUT,
    )
    assert r.status_code == 413


def test_health_responsive_during_transcription(base_url, auth_headers, sample_path):
    """Пока идёт транскрипция, /health отвечает быстро (event loop жив)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        with open(sample_path, "rb") as f:
            fut = pool.submit(
                requests.post,
                f"{base_url}/v1/audio/transcriptions",
                headers=auth_headers,
                files={"file": ("long.ogg", f.read())},
                data={"response_format": "text"},
                timeout=TIMEOUT,
            )
            time.sleep(1.5)  # транскрипция 137-с файла точно ещё идёт
            t0 = time.perf_counter()
            r = requests.get(f"{base_url}/health", timeout=10)
            health_latency = time.perf_counter() - t0
        assert fut.result().status_code == 200

    assert r.status_code == 200
    assert health_latency < 2.0, f"/health отвечал {health_latency:.1f}s во время транскрипции"
