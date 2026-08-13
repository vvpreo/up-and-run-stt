"""
OpenAI-совместимость: /v1/models, отказ translations, формат ошибок, usage.
"""

import pytest
import requests

TIMEOUT = 300


def test_models_list(base_url, auth_headers):
    r = requests.get(f"{base_url}/v1/models", headers=auth_headers, timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    health_models = requests.get(f"{base_url}/health", timeout=10).json()["models"]
    assert set(ids) == set(health_models)
    for m in data["data"]:
        assert m["object"] == "model" and m["created"] and m["owned_by"]


def test_models_retrieve_and_404(base_url, auth_headers):
    some = requests.get(f"{base_url}/v1/models", headers=auth_headers, timeout=10).json()["data"][0]["id"]
    r = requests.get(f"{base_url}/v1/models/{some}", headers=auth_headers, timeout=10)
    assert r.status_code == 200 and r.json()["id"] == some

    r = requests.get(f"{base_url}/v1/models/no-such-model", headers=auth_headers, timeout=10)
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "model_not_found"
    assert err["type"] == "invalid_request_error"


def test_translations_refused_with_envelope(base_url, auth_headers, short_wav):
    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/translations",
            headers=auth_headers,
            files={"file": ("a.wav", f)},
            timeout=TIMEOUT,
        )
    assert r.status_code == 400
    assert "not supported" in r.json()["error"]["message"]


def test_v1_auth_error_uses_envelope(base_url, token, short_wav):
    if not token:
        pytest.skip("AUTH_TOKEN not set — auth disabled")
    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": ("a.wav", f)},
            timeout=TIMEOUT,
        )
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_validation_error_is_400_envelope(base_url, auth_headers):
    # без файла: у OpenAI это 400 invalid_request_error, а не FastAPI 422
    r = requests.post(f"{base_url}/v1/audio/transcriptions", headers=auth_headers, timeout=10)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_bracketed_granularities_and_ignored_params(base_url, auth_headers, short_wav):
    """SDK-стиль: timestamp_granularities[] + stream/temperature не дают 422."""
    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            headers=auth_headers,
            files={"file": ("a.wav", f)},
            data=[
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "segment"),
                ("stream", "false"),
                ("temperature", "0"),
            ],
            timeout=TIMEOUT,
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("words"), "words must be present for bracketed granularities"
    assert data.get("usage", {}).get("type") == "duration"
