# Базовый образ для прогона тестов в CI. Публикуется в GHCR
# (ghcr.io/vvpreo/up-and-run-stt-ci) воркфлоу .github/workflows/ci-image.yml.
#
# Зачем он: интеграционные тесты нарезают фикстуры внешним ffmpeg, а в
# раннерах GitHub Actions его нет ни на amd64, ни на arm64 — приходилось
# ставить apt'ом на каждом прогоне. Здесь он уже внутри, вместе с pytest и
# requests, поэтому шаг тестов не ставит вообще ничего: ни apt-пакетов, ни
# рантайм-зависимостей сервиса (onnxruntime/scipy/numpy, ~250 МБ), которые
# самим тестам не нужны — они ходят в сервис по HTTP.
#
# Тесты запускаются контейнером из этого образа в одной docker-сети с
# контейнером сервиса; адрес передаётся через STT_BASE_URL.

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="up-and-run-stt-ci" \
      org.opencontainers.image.description="Test runner image for up-and-run-stt: ffmpeg + pytest" \
      org.opencontainers.image.source="https://github.com/vvpreo/up-and-run-stt" \
      org.opencontainers.image.licenses="MIT"

# Тот же источник и та же версия ffmpeg, что и в основном образе, чтобы
# фикстуры кодировались тем же кодом, который потом их декодирует.
COPY --from=mwader/static-ffmpeg:7.1 /ffmpeg /usr/local/bin/ffmpeg

# Версии должны совпадать с группой dev в pyproject.toml (uv.lock).
RUN pip install --no-cache-dir \
    pytest==9.1.1 \
    requests==2.34.2 \
    websockets==17.0.1 \
    && ffmpeg -version | head -1 \
    && pytest --version

WORKDIR /workspace
