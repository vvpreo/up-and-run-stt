"""
Регрессия: субтитровые форматы непустые на ОБОИХ контрактах и на коротком
аудио (баг: OpenAI srt/vtt возвращали пустоту — движку заказывался text
вместо json, а короткое аудио не имело сегментов вовсе).
"""

import requests

TIMEOUT = 300


def _openai(base_url, headers, wav, fmt):
    with open(wav, "rb") as f:
        return requests.post(
            f"{base_url}/v1/audio/transcriptions",
            headers=headers,
            files={"file": ("a.wav", f)},
            data={"response_format": fmt},
            timeout=TIMEOUT,
        )


def _native(base_url, headers, wav, fmt):
    with open(wav, "rb") as f:
        return requests.post(
            f"{base_url}/stt/asr?output={fmt}",
            headers=headers,
            files={"audio_file": ("a.wav", f)},
            timeout=TIMEOUT,
        )


def test_openai_srt_not_empty(base_url, auth_headers, short_wav):
    r = _openai(base_url, auth_headers, short_wav, "srt")
    assert r.status_code == 200
    assert "-->" in r.text and len(r.text.strip()) > 30, f"empty srt: {r.text!r}"


def test_openai_vtt_not_empty(base_url, auth_headers, short_wav):
    r = _openai(base_url, auth_headers, short_wav, "vtt")
    assert r.status_code == 200
    assert r.text.startswith("WEBVTT")
    assert "-->" in r.text, f"vtt has no cues: {r.text!r}"


def test_native_srt_not_empty(base_url, auth_headers, short_wav):
    r = _native(base_url, auth_headers, short_wav, "srt")
    assert r.status_code == 200
    assert "-->" in r.text and len(r.text.strip()) > 30


def test_native_vtt_not_empty(base_url, auth_headers, short_wav):
    r = _native(base_url, auth_headers, short_wav, "vtt")
    assert r.status_code == 200
    assert "-->" in r.text


def test_native_tsv_not_empty(base_url, auth_headers, short_wav):
    r = _native(base_url, auth_headers, short_wav, "tsv")
    assert r.status_code == 200
    lines = [l for l in r.text.strip().splitlines() if l.strip()]
    assert len(lines) >= 2, f"tsv has no data rows: {r.text!r}"
