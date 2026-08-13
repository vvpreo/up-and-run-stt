# TODO

- [ ] [Переезд на ONNX Runtime: engine, эмоции, Dockerfile без torch, хостинг весов](docs/plans/public-image-roadmap.md) — совместимость таймстемпов и эмоций подтверждена экспериментально, фичерайзер готов

# PLANNED

# TO REVIEW

- [X] Ужать образ: 2.82 → 1.81 ГБ (−36%)
  - apt-пакет ffmpeg (с деревом библиотек ~550 МБ) заменён на статический бинарь johnvansickle (~80 МБ), скачивается при сборке stdlib-питоном.
  - librosa выпилена (−~250 МБ: numba/llvmlite/sklearn): ресемплинг переведён на `scipy.signal.resample_poly`; заодно убран git из apt.
  - Перепрогнаны все тесты: 8 форматов — 200; e2e 6.3–6.9 с (без изменений); декод 0.09–0.63 с; транскрипт бит-в-бит совпал с прежним (md5); RAM процесса ~2.5 ГБ с двумя моделями (без изменений).

- [X] ffmpeg-фолбэк декодирования: поддержка m4a/webm/wma и пр.
  - В образ добавлен `ffmpeg`; в [src/utils/audio.py](src/utils/audio.py) фолбэк: libsndfile (быстрый путь) → ffmpeg через временный файл (m4a с `moov atom` в конце не декодируется из pipe).
  - Проверены 8 форматов: wav, flac, mp3, ogg, opus, m4a, webm, wma — все 200.
  - Ресерч скорости по форматам (клип 137 с): декодирование 59–528 мс против ~6 с инференса, влияние формата < 8% запроса; таблица в README («Входные форматы аудио»).

- [X] Реестр моделей: набор инстанса + выбор модели полем `model` в запросе
  - `GIGAAM_MODELS` (список через запятую) — набор моделей инстанса: все грузятся при старте (веса докачиваются в том) и держатся в RAM (~1.4 ГБ каждая); первая — дефолт. `GIGAAM_MODEL` работает как алиас.
  - Выбор в запросе на обоих эндпоинтах: поле формы `model` (OpenAI) / query `model` (нативный). `whisper-1`/пусто/незнакомое → дефолт; известный, но не включённый вариант → 400 со списком.
  - Новый модуль [src/asr/registry.py](src/asr/registry.py); `/health` показывает `models` и `default_model`; на тестовой странице — селектор модели.
  - В текущем деплое включены `v3_e2e_ctc,v3_e2e_rnnt` (процесс ~2.4 ГБ RAM); проверено: разные модели дают разные транскрипты, 400 для невключённой.
  - Ограничение: при `MODEL_WORKERS > 1` пул работает только с дефолтной моделью.

- [X] Убрать `ASR_ENGINE`: движок фиксируется на уровне endpoint'а
  - Нативный endpoint переименован: `/asr` → `/gigaam/asr` (breaking change для тех, кто ходил на старый путь). OpenAI-совместимый `/v1/audio/transcriptions` не тронут.
  - Переменная `ASR_ENGINE` удалена из compose, Dockerfile и config.py; в `/health` и заголовках `Asr-Engine` — константа `gigaam`.
  - Выбор модели остался: `GIGAAM_MODEL` (дефолт в config.py исправлен на `v3_e2e_ctc` — был несуществующий для нативного пакета `ai-sage/GigaAM-v3`).
  - Заодно вычищены мёртвые переменные удалённых движков: `WHISPER_MODEL`, `WHISPERX_MODEL`, `GIGAAM_REVISION`, `GIGAAM_MLX_*`, `COMPUTE_TYPE`; заголовок Swagger переписан с OAITT/Whisper на gigaam-stt.
  - Тестовая страница и README обновлены на новый путь; сервис пересобран и проверен.

- [X] Вычистить пофайловые копирайт-шапки Андрея Соболева из исходников
  - Сняты docstring-заголовки «OAITT … Copyright Andrey Sobolev» в 29 py-файлах; код и работа сервиса не изменились (пересобран, проверен).
  - Атрибуция сохранена там, где обязана быть: `LICENSE` (MIT, копирайт Andrey Sobolev) и README (раздел «Лицензии и авторство» — явное упоминание, что база — haiodo/oaitt).

- [X] Уйти от Makefile и `.env`: сборка через `build.sh`, конфигурация — в `docker-compose.yml`
  - `Dockerfile.cpu` переименован в `Dockerfile` (публичная конвенция для Docker Hub).
  - Все переменные окружения перенесены в блок `environment` compose вместе с комментариями; `.env`/`.env.example`/`Makefile` удалены (бэкап в `.tmp/pruned-files-2026-08-13.tar`).
  - `build.sh` — сборка локального образа или с тегом для Docker Hub (`./build.sh myuser/gigaam-stt --push`).
  - Исправлен дефолт `AUTH_TOKEN` в config.py: было `"key"` (авторизация молча включалась с общеизвестным токеном при чистом `docker run`), стало пусто = отключена.

- [X] Высушить проект до рабочего минимума (убрать бенчмарки и пр.)
  - Удалены: `benchmark/`, `tests/`, `scripts/`, `docs/upstream/`, `src/batch_transcribe.py`, обучающие/демо-части `vendor/gigaam` (tests, train_utils, triton_scripts, assets, colab).
  - Из образа убраны неиспользуемые `transformers`, `onnx`, `onnxruntime` (проверено: не импортируются в рабочих путях); образ 2.19 ГБ.
  - `factory.py` сведён к gigaam-only; Makefile — только сервисные таргеты; `requirements.txt` соответствует реальным зависимостям.
  - Оставлены: `docs/RU_QUALITY.md` (отчёт о качестве), `sample-data/` (нужен для `make test`), тестовая веб-страница `src/static/index.html`.
  - Бэкап всего удалённого: `.tmp/pruned-backup-2026-08-13.tar.gz` и `.tmp/pruned-backup-vendor-2026-08-13.tar.gz`.

- [X] Опубликовать сервис + тестовую веб-страницу на `https://gigam-stt.dev.vvpreo.net` (через dev-nginx)
  - Тестовая страница (микрофон → WAV в браузере, загрузка файла) встроена в сервис: `GET /` ([src/static/index.html](src/static/index.html)).
  - Конфиг `nginx/conf.d/gigam-stt.conf` в dev-nginx; wildcard-сертификат покрывает домен (вместо `gigam.stt.*` выбран `gigam-stt.*` — TLS-wildcard действует на один уровень).
  - Включён `AUTH_TOKEN` в `.env`: транскрипционные эндпоинты требуют Bearer-токен (страница, `/health`, `/docs` открыты).
  - ⚠️ Прямой доступ по `:9007` теперь тоже требует токен — VoiceInk на Mac нужно прописать ключ.

- [X] Развернуть GigaAM STT-сервис в Docker (CPU) с OpenAI-совместимым API — см. [README.md](README.md)
  - Форк `haiodo/oaitt`, только движок GigaAM Native (CPU); MLX/WhisperX убраны.
  - `docker compose up -d` — одна команда; `restart: unless-stopped`; healthcheck `/health`.
  - Эндпоинты: `/v1/audio/transcriptions`, `/asr`, `/health`.
  - Веса в named volume `gigaam-models` (не запечены, не докачиваются при рестарте).
  - `AUTH_TOKEN` пуст → авторизация отключена; `MODEL_IDLE_TIMEOUT=0` (тёплая модель).
  - Бенчмарк WER/CER/RTF/RAM + отчёт [docs/RU_QUALITY.md](docs/RU_QUALITY.md).
  - Интеграция VoiceInk + OpenAI SDK задокументирована в README.
- [ ] Подтвердить транскрипцию с Mac через VoiceInk (скрин/лог) — требует действий на Mac-хосте.

# DONE
