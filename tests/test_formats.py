"""
Входные форматы: каждый декодируется и транскрибируется (libsndfile + ffmpeg-фолбэк).
"""

import subprocess

import pytest
import requests

TIMEOUT = 300

FORMATS = {
    "wav": "pcm_s16le",
    "flac": "flac",
    "mp3": "libmp3lame",
    "ogg": "libvorbis",
    "opus": "libopus",
    "m4a": "aac",       # декодируется только ffmpeg-фолбэком
    "webm": "libopus",  # декодируется только ffmpeg-фолбэком
    "wma": "wmav2",     # декодируется только ffmpeg-фолбэком
}


@pytest.fixture(scope="module")
def format_files(tmp_path_factory, sample_path):
    out_dir = tmp_path_factory.mktemp("formats")
    files = {}
    for ext, codec in FORMATS.items():
        out = out_dir / f"clip.{ext}"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(sample_path),
             "-t", "6", "-c:a", codec, str(out)],
            check=True,
        )
        files[ext] = out
    return files


@pytest.mark.parametrize("ext", FORMATS)
def test_format_transcribes(base_url, auth_headers, format_files, ext):
    with open(format_files[ext], "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            headers=auth_headers,
            files={"file": (f"clip.{ext}", f)},
            data={"response_format": "text"},
            timeout=TIMEOUT,
        )
    assert r.status_code == 200, r.text
    assert len(r.text.strip()) > 10
