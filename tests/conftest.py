"""
Общие фикстуры тестов.

Интеграционные тесты ходят в работающий сервис (BASE_URL, по умолчанию
локальный docker-контейнер). Перед прогоном сервис должен быть поднят:
`docker compose up -d` и дождаться `model_loaded: true`.

Запуск: pytest tests/ -v
"""

import os
import subprocess
from pathlib import Path

import pytest

BASE_URL = os.getenv("STT_BASE_URL", "http://127.0.0.1:9007")
REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = REPO_ROOT / "sample-data" / "Sobolev_Andrey_1_0_00-2_17.ogg"


def _auth_token() -> str:
    """AUTH_TOKEN из env или из docker-compose.yml."""
    if os.getenv("STT_AUTH_TOKEN"):
        return os.environ["STT_AUTH_TOKEN"]
    compose = (REPO_ROOT / "docker-compose.yml").read_text()
    for line in compose.splitlines():
        if "AUTH_TOKEN:" in line:
            return line.split(":", 1)[1].strip()
    return ""


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def token() -> str:
    return _auth_token()


@pytest.fixture(scope="session")
def auth_headers(token) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


@pytest.fixture(scope="session")
def sample_path() -> Path:
    assert SAMPLE.exists(), f"Sample audio not found: {SAMPLE}"
    return SAMPLE


@pytest.fixture(scope="session")
def short_wav(tmp_path_factory, sample_path) -> Path:
    """Короткий (8 с) wav-клип из семпла — быстрые транскрипции в тестах."""
    out = tmp_path_factory.mktemp("audio") / "short.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(sample_path),
         "-t", "8", "-ar", "16000", "-ac", "1", str(out)],
        check=True,
    )
    return out
