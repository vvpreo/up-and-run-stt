"""
Backpressure: при переполнении очереди сервис отвечает 429, а не копит запросы.

Тест шлёт заведомо больше параллельных запросов, чем MAX_PENDING_REQUESTS
(8 по умолчанию), и ожидает смесь 200 и 429. Чувствителен к таймингу
отправки, поэтому запросов сильно больше лимита.
"""

import concurrent.futures

import requests

TIMEOUT = 300
PARALLEL = 24


def _post(base_url, headers, payload):
    return requests.post(
        f"{base_url}/v1/audio/transcriptions",
        headers=headers,
        files={"file": ("clip.wav", payload)},
        data={"response_format": "text"},
        timeout=TIMEOUT,
    ).status_code


def test_queue_overflow_returns_429(base_url, auth_headers, short_wav):
    payload = short_wav.read_bytes()
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        codes = list(
            pool.map(lambda _: _post(base_url, auth_headers, payload), range(PARALLEL))
        )

    assert set(codes) <= {200, 429}, f"unexpected codes: {codes}"
    assert codes.count(200) >= 1, "no request succeeded"
    assert codes.count(429) >= 1, (
        f"no 429 among {PARALLEL} parallel requests — backpressure not working: {codes}"
    )
