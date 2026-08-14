"""
VAD-чанкование: корректность пословных таймстемпов на длинном аудио,
переопределение per-request, пропуск тишины.
"""

import subprocess

import pytest
import requests

TIMEOUT = 300


def _words(base_url, headers, path, vad: bool):
    with open(path, "rb") as f:
        r = requests.post(
            f"{base_url}/stt/asr?output=json&word_timestamps=true&vad={str(vad).lower()}",
            headers=headers,
            files={"audio_file": ("long.ogg", f)},
            timeout=TIMEOUT,
        )
    assert r.status_code == 200, r.text
    data = r.json()
    words = [w for s in (data.get("segments") or []) for w in (s.get("words") or [])]
    return data, words


def test_health_reports_vad(base_url):
    h = requests.get(f"{base_url}/health", timeout=10).json()
    assert "vad_chunking" in h


def test_word_timestamps_correct_with_vad(base_url, auth_headers, sample_path):
    """Главный тест: слова при VAD-чанковании монотонны и покрывают весь файл."""
    data, words = _words(base_url, auth_headers, sample_path, vad=True)
    assert len(words) > 40, f"suspiciously few words: {len(words)}"

    # тайминги монотонно неубывающие, в пределах длительности файла (137.4 с)
    starts = [w["start"] for w in words]
    assert starts == sorted(starts), "word starts are not monotonic"
    for w in words:
        assert 0 <= w["start"] < w["end"] <= 138.5, f"bad word bounds: {w}"

    # слова покрывают весь файл, а не только первый чанк:
    assert starts[0] < 5, f"first word too late: {starts[0]}"
    assert starts[-1] > 120, f"last word too early: {starts[-1]} — offsets broken?"

    # сегменты тоже монотонны и не пересекаются
    segs = data["segments"]
    for a, b in zip(segs, segs[1:]):
        assert a["end"] <= b["start"] + 0.01, f"segments overlap: {a} / {b}"


def test_vad_and_fixed_words_agree(base_url, auth_headers, sample_path):
    """VAD и жёсткая нарезка дают близкие тайминги одних и тех же слов."""
    _, vad_words = _words(base_url, auth_headers, sample_path, vad=True)
    _, fixed_words = _words(base_url, auth_headers, sample_path, vad=False)

    # количество слов сопоставимо (VAD может чуть отличаться на стыках)
    assert abs(len(vad_words) - len(fixed_words)) <= max(5, len(fixed_words) // 10)

    # первые общие слова имеют практически одинаковые тайминги
    common = min(10, len(vad_words), len(fixed_words))
    for wv, wf in zip(vad_words[:common], fixed_words[:common]):
        assert abs(wv["start"] - wf["start"]) < 1.0, (wv, wf)


def test_vad_skips_silence(base_url, auth_headers, sample_path, tmp_path):
    """Файл: 8 с речи + 40 с тишины + 8 с речи — тишина не в сегментах."""
    speech = tmp_path / "speech8.wav"
    silence = tmp_path / "sil40.wav"
    combined = tmp_path / "combined.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(sample_path),
                    "-t", "8", "-ar", "16000", "-ac", "1", str(speech)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "anullsrc=r=16000:cl=mono", "-t", "40",
                    "-ar", "16000", "-ac", "1", str(silence)], check=True)
    # concat-ФИЛЬТР (протокол concat: ломает WAV — заголовки в середине данных)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(speech), "-i", str(silence), "-i", str(speech),
                    "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1",
                    str(combined)], check=True)

    data, words = _words(base_url, auth_headers, combined, vad=True)
    assert len(words) >= 8
    # вторая порция речи начинается на ~48-й секунде — слова должны быть и там
    assert any(w["start"] > 45 for w in words), "words after silence are missing"
    # ни одно слово не попадает в мёртвую тишину (12..44 с)
    assert not any(15 < w["start"] < 44 for w in words), "words inside silence"
    # сегменты не покрывают тишину целиком: суммарная длительность << 56 с
    total_seg = sum(s["end"] - s["start"] for s in data["segments"])
    assert total_seg < 30, f"segments cover silence: {total_seg:.1f}s"
