"""
GET /stats — загрузка сервиса для панели «Загрузка» в консоли.
"""

import pytest
import requests


@pytest.fixture(autouse=True)
def _stats_enabled(base_url):
    if requests.get(f"{base_url}/health", timeout=10).json().get("stats_enabled") is False:
        pytest.skip("stats are disabled on this instance (ENABLE_STATS=false)")


def test_stats_shape_and_public_access(base_url):
    r = requests.get(f"{base_url}/stats", timeout=10)  # без токена, как /health
    assert r.status_code == 200
    d = r.json()

    assert d["cpu"]["cores"] >= 1
    assert d["cpu"]["process_pct"] >= 0
    assert d["memory"]["process_rss_mb"] > 100, "модель загружена — RSS не может быть крошечным"
    assert d["memory"]["system_total_mb"] >= d["memory"]["process_rss_mb"]

    # GPU — либо null (CPU-образ), либо объект с именем карты
    assert d["gpu"] is None or d["gpu"]["name"]

    names = [m["name"] for m in d["models"]]
    health = requests.get(f"{base_url}/health", timeout=10).json()
    for model in health["models"]:
        assert model in names
    asr = next(m for m in d["models"] if m["name"] == health["default_model"])
    assert asr["loaded"] is True
    assert asr["device"] == health["device"]


def test_stats_attribute_time_to_the_model(base_url, auth_headers, short_wav):
    """После транскрипции в разбивке появляется время модели и VAD-счётчики растут."""
    default = requests.get(f"{base_url}/health", timeout=10).json()["default_model"]
    key = f"asr:{default}"
    before = requests.get(f"{base_url}/stats", timeout=10).json()["activity"].get(key, {})

    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            headers=auth_headers,
            files={"file": ("a.wav", f, "audio/wav")},
            data={"response_format": "text"},
            timeout=120,
        )
    assert r.status_code == 200

    after = requests.get(f"{base_url}/stats", timeout=10).json()["activity"]
    assert key in after, after.keys()
    a = after[key]
    assert a["calls_total"] > before.get("calls_total", 0)
    assert a["seconds_total"] > before.get("seconds_total", 0)
    assert a["audio_sec_total"] > before.get("audio_sec_total", 0)
    assert 0 <= a["busy_pct"] <= 100 * 8  # доля окна; несколько параллельных вызовов дают >100


def test_window_history_is_recorded_only_while_observed(base_url, auth_headers, short_wav):
    """
    Пооконная статистика ведётся только пока /stats кто-то опрашивает: первый
    запрос «включает» наблюдение, и транскрипция после него попадает в окно.
    """
    requests.get(f"{base_url}/stats", timeout=10)  # включить наблюдение
    default = requests.get(f"{base_url}/health", timeout=10).json()["default_model"]
    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions", headers=auth_headers,
            files={"file": ("a.wav", f, "audio/wav")}, data={"response_format": "text"}, timeout=120,
        )
    assert r.status_code == 200
    a = requests.get(f"{base_url}/stats", timeout=10).json()["activity"][f"asr:{default}"]
    assert a["calls"] >= 1 and a["avg_ms"] is not None
