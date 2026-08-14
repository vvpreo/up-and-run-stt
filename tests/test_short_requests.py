"""
Паттерн десктопной диктовки: клиент шлёт много коротких wav-запросов подряд
(на каждое нажатие «диктовать» — отдельный клип). Проверяем, что сервис держит
такой поток: каждый ответ быстрее длительности клипа, огрызки не роняют раут,
/health не голодает под нагрузкой.
"""

import subprocess
import time

import requests

TIMEOUT = 300


def _cut(sample_path, tmp_path, name, start, dur):
    out = tmp_path / f"{name}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(sample_path),
         "-t", str(dur), "-ar", "16000", "-ac", "1", str(out)],
        check=True,
    )
    return out


def _post_phrase(base_url, headers, path):
    with open(path, "rb") as f:
        return requests.post(
            f"{base_url}/v1/audio/transcriptions",
            headers=headers,
            files={"file": ("phrase.wav", f)},
            data={"response_format": "json"},
            timeout=TIMEOUT,
        )


def test_sequential_phrases_fast_and_ordered(base_url, auth_headers, sample_path, tmp_path):
    """Три коротких клипа подряд: каждый обрабатывается быстро и непустой."""
    phrases = [
        _cut(sample_path, tmp_path, "p1", 0, 3),
        _cut(sample_path, tmp_path, "p2", 12, 3),
        _cut(sample_path, tmp_path, "p3", 100, 3),
    ]

    texts, latencies = [], []
    for p in phrases:
        t0 = time.perf_counter()
        r = _post_phrase(base_url, auth_headers, p)
        latencies.append(time.perf_counter() - t0)
        assert r.status_code == 200, r.text
        texts.append(r.json()["text"].strip())

    assert all(texts), f"empty phrase transcription: {texts}"
    assert "проверяем" in texts[0].lower()
    # Диктовка осмысленна, только если клип обрабатывается быстрее, чем
    # произносится: 3-с клип должен уходить сильно быстрее 3 с.
    assert max(latencies) < 3.0, f"phrase latency too high for dictation: {latencies}"


def test_tiny_fragment_does_not_error(base_url, auth_headers, sample_path, tmp_path):
    """Огрызок в полсекунды (случайное нажатие) не должен ронять сервер."""
    tiny = _cut(sample_path, tmp_path, "tiny", 5.0, 0.5)
    r = _post_phrase(base_url, auth_headers, tiny)
    assert r.status_code == 200, r.text
    assert "text" in r.json()  # текст может быть пустым — главное, не 5xx


def test_phrase_requests_do_not_starve_health(base_url, auth_headers, sample_path, tmp_path):
    """Во время потока коротких запросов /health остаётся отзывчивым."""
    import concurrent.futures

    phrase = _cut(sample_path, tmp_path, "ph", 0, 3)
    payload = phrase.read_bytes()

    def spam():
        for _ in range(4):
            requests.post(
                f"{base_url}/v1/audio/transcriptions",
                headers=auth_headers,
                files={"file": ("phrase.wav", payload)},
                data={"response_format": "json"},
                timeout=TIMEOUT,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(spam)
        time.sleep(0.7)
        t0 = time.perf_counter()
        r = requests.get(f"{base_url}/health", timeout=10)
        health_latency = time.perf_counter() - t0
        fut.result()

    assert r.status_code == 200
    assert health_latency < 2.0, f"/health blocked during phrase stream: {health_latency:.1f}s"
