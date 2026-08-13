"""
Тесты /auth/check (гейт WebUI) и /gigaam/emotion (эмоции, ONNX-бэкенд).
"""

import pytest
import requests

TIMEOUT = 300


def test_auth_check_requires_token(base_url, token):
    if not token:
        pytest.skip("AUTH_TOKEN not set — auth disabled")
    assert requests.get(f"{base_url}/auth/check", timeout=10).status_code == 401
    assert requests.get(
        f"{base_url}/auth/check",
        headers={"Authorization": "Bearer wrong"},
        timeout=10,
    ).status_code == 401


def test_auth_check_accepts_valid_token(base_url, auth_headers):
    r = requests.get(f"{base_url}/auth/check", headers=auth_headers, timeout=10)
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_health_exposes_feature_flags(base_url):
    h = requests.get(f"{base_url}/health", timeout=10).json()
    assert "auth_required" in h
    assert "emotions_enabled" in h


def test_emotion_endpoint(base_url, auth_headers, short_wav):
    h = requests.get(f"{base_url}/health", timeout=10).json()
    if not h.get("emotions_enabled"):
        pytest.skip("emotions disabled on this instance")

    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/gigaam/emotion",
            headers=auth_headers,
            files={"audio_file": ("a.wav", f)},
            timeout=TIMEOUT,  # первый вызов лениво грузит модель
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert set(data["emotions"]) == {"angry", "sad", "neutral", "positive"}
    assert data["dominant"] in data["emotions"]
    assert abs(sum(data["emotions"].values()) - 1.0) < 0.02


def test_emotion_requires_token(base_url, short_wav, token):
    if not token:
        pytest.skip("AUTH_TOKEN not set — auth disabled")
    with open(short_wav, "rb") as f:
        r = requests.post(
            f"{base_url}/gigaam/emotion",
            files={"audio_file": ("a.wav", f)},
            timeout=TIMEOUT,
        )
    assert r.status_code == 401
