# TODO

- [ ] Публикация на Docker Hub: осталось создать Access Token на Хабе и положить секреты `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` в репозиторий, затем запустить Actions → Release. Опционально: зеркало в GHCR, trivy-скан
- [ ] Метрики Prometheus: `/metrics` (счётчики запросов по эндпоинту/модели/статусу, гистограммы времени и RTF, секунды обработанного аудио, глубина очереди)
- [ ] Опционально: вариант образа с запечёнными весами (`-offline`, +~850 МБ) для air-gapped и независимости от HF
- [ ] Мелочь OpenAI-полноты: `language` в verbose_json полным английским словом («russian»), как у whisper-1
- [ ] Настоящий стриминг **запроса**: WS-эндпоинт (браузер шлёт PCM-кадры, сервер сам режет и отдаёт дельты) — инкрементальный `feed(frame)` вместо `speech_probs(audio)` в [src/asr/vad.py](src/asr/vad.py), протокол кадров, бэкпрешер, отмена при обрыве. Тогда VAD останется в одном месте, а клиент станет тупой трубой

# PLANNED

- [ ] [SSE-стриминг + VAD-чанкование](docs/plans/streaming-and-vad.md) — план с оценками производительности и ресурсов; порядок: VAD → стриминг → фразовые cue

# TO REVIEW

- [X] Фундамент для публикации образа + CI/CD на GitHub Actions
  - Зависимости переведены на **uv**: `pyproject.toml` + `uv.lock` (39 пакетов, точная резолюция включая транзитивные). Лок универсальный — `[tool.uv] environments` покрывает linux x86_64 и aarch64, поэтому обе архитектуры ставят идентичные версии. Сборка идёт через `uv sync --frozen` (падает, если лок разошёлся с pyproject). `requirements.txt`/`requirements-dev.txt` удалены — первый был мёртвый (ссылался на несуществующий `Dockerfile.cpu`, тянул torch, не упоминал onnxruntime).
  - Multi-arch: статический ffmpeg выбирается по `ARG TARGETARCH` вместо прибитого amd64-URL.
  - OCI-лейблы (source, revision, version, licenses и пр.) — на Docker Hub будет ссылка на исходники и видно, из какого коммита собран образ.
  - Dockerfile стал многостадийным: uv ставит зависимости в builder, в финальный образ едет только готовый venv. Размер 742 → **656 МБ**. По дороге поймал регресс до 1 ГБ — `chown -R` поверх скопированного venv дублировал 250 МБ в отдельный слой; чинится созданием пользователя ДО копирования и `COPY --chown`.
  - `.github/workflows/ci.yml` — на каждый push/PR: сборка и **47 интеграционных тестов на обеих архитектурах** (нативные раннеры `ubuntu-latest` и `ubuntu-24.04-arm`, бесплатные для публичных репо). Веса моделей кэшируются через `actions/cache`.
  - `.github/workflows/release.yml` — ручной запуск с выбором patch/minor/major: поднимает версию в `pyproject.toml` и `src/__init__.py`, коммитит, ставит тег `vX.Y.Z`, собирает обе архитектуры нативно, сшивает multi-arch манифест, публикует теги `X.Y.Z` / `X.Y` / `X` / `latest`, синкает README на Хаб и создаёт GitHub Release. Есть режим `dry_run` — сборка без публикации.
  - Статический ffmpeg теперь берётся из образа `mwader/static-ffmpeg:7.1`, а не качается с johnvansickle.com: тот отдаёт датацентровым IP HTML-заглушку с кодом 200, и сборка в CI падала на распаковке. Цена — бинарь 129 МБ вместо 80 МБ, образ 656 → **734 МБ**; 55 МБ можно вернуть, зеркалировав лёгкий ffmpeg в свой HF-репо (как уже сделано для silero-vad).
  - Тестам нужен ffmpeg **на раннере** (нарезка фикстур) — в образах GitHub Actions его нет ни на одной архитектуре, ставится отдельным шагом.
  - ✅ **CI зелёный на обеих архитектурах**: 47 тестов прошли и на amd64 (3:21), и на arm64 (4:15). Раннер `ubuntu-24.04-arm` доступен, образ на нём собирается из универсального uv-лока и работает.
  - ⚠️ Release-воркфлоу ещё не запускался — нужны секреты Docker Hub.

- [X] Переименование `gigaam-stt` → `up-and-run-stt` + обезличивание проекта
  - Имя продукта: 63 замены в 23 файлах (README, TODO, Dockerfile ×2, docker-compose, build.sh, main.py, src/app.py, WebUI, скрипты, тесты). Образ, compose-сервис и контейнер теперь `up-and-run-stt`.
  - Роуты: `/gigaam/asr` → `/stt/asr`, `/gigaam/emotion` → `/stt/emotion`. **Breaking change без алиасов** — старые пути отвечают 404 (проверено). OpenAI-совместимый `/v1/audio/transcriptions` не тронут, десктопные диктовщики не затронуты.
  - Намеренно НЕ переименованы (это про модель, а не про продукт): `GIGAAM_*` переменные, [src/asr/gigaam.py](src/asr/gigaam.py), `vendor/gigaam`, том `gigaam-models`, HF-репо `vvpreo/gigaam-v3-onnx`, путь токенизаторов `/app/data/gigaam/`.
  - Обезличивание: убраны все упоминания VoiceInk, DevBox, Mac, dev-nginx и домена деплоя из README, docker-compose, TODO, тестов и `docs/RU_QUALITY.md`. Раздел README «VoiceInk (Mac)» → «Desktop dictation clients» (любой клиент с настраиваемым OpenAI-эндпоинтом).
  - В TODO поправлена история, которую задела автозамена: запись про переименование `/asr` восстановлена как `/asr` → `/gigaam/asr` → `/stt/asr`.
  - Проверено: образ пересобран, контейнер пересоздан (том с весами цел, модели не перекачивались), `/stt/asr` отвечает, `/gigaam/asr` → 404, Swagger title `up-and-run-stt`, 47 тестов зелёные.

- [X] Уточнить, что `stream=true` — это стриминг **ответа**, и откатить живую диктовку в WebUI ([ревизия плана](docs/plans/streaming-and-vad.md))
  - Причина: живая диктовка имитировала стриминг запроса, которого в сервисе нет, и дублировала нарезку речи в браузере (энергетический VAD по RMS) поверх серверного silero-vad.
  - Из [src/static/index.html](src/static/index.html) убраны `liveFeed`/`flushPhrase`, детектор пауз, пофразная очередь и live-карточка (−~90 строк). Микрофон = обычный источник файла: запись целиком → «Стоп» → тот же `transcribe()`, что и drag-and-drop, с учётом тумблера стриминга.
  - Тумблер «Стриминг» → «Стриминг ответа», подсказки и tooltip переписаны; `UI_BUILD` поднят до `resp-stream-1` (сброс кэша страницы).
  - README: раздел «SSE streaming» начинается с явного предупреждения «response only» + сопоставление с OpenAI (их Realtime API — отдельный продукт); раздел «Live dictation» заменён на «Microphone in the WebUI».
  - `tests/test_live_dictation.py` → `tests/test_short_requests.py` — проверки те же (короткие последовательные запросы, огрызки, отзывчивость `/health`), переформулированы под внешнего клиента-диктовщика.
  - ⚠️ Не проверено вживую в браузере — нужен прогон на работающем сервисе.

- [X] Переезд на ONNX Runtime — завершён: движок (CTC+RNNT, слова, чанки, паритет с torch бит-в-бит), веса всех вариантов + emo на huggingface.co/vvpreo/gigaam-v3-onnx (конвертация `scripts/convert_onnx.py`), образ 733 МБ без torch (`Dockerfile.torch` — для конвертации), эмоции `/stt/emotion`, WebUI-консоль с токен-гейтом и всеми фичами, OpenAI-совместимость (`/v1/models`, формат ошибок, отказ translations, терпимость к параметрам SDK, `usage`), 32 интеграционных теста

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
  - Нативный endpoint переименован: `/asr` → `/gigaam/asr` (breaking change для тех, кто ходил на старый путь; позже переименован ещё раз, в `/stt/asr`). OpenAI-совместимый `/v1/audio/transcriptions` не тронут.
  - Переменная `ASR_ENGINE` удалена из compose, Dockerfile и config.py; в `/health` и заголовках `Asr-Engine` — константа `gigaam`.
  - Выбор модели остался: `GIGAAM_MODEL` (дефолт в config.py исправлен на `v3_e2e_ctc` — был несуществующий для нативного пакета `ai-sage/GigaAM-v3`).
  - Заодно вычищены мёртвые переменные удалённых движков: `WHISPER_MODEL`, `WHISPERX_MODEL`, `GIGAAM_REVISION`, `GIGAAM_MLX_*`, `COMPUTE_TYPE`; заголовок Swagger переписан с OAITT/Whisper на имя проекта.
  - Тестовая страница и README обновлены на новый путь; сервис пересобран и проверен.

- [X] Вычистить пофайловые копирайт-шапки Андрея Соболева из исходников
  - Сняты docstring-заголовки «OAITT … Copyright Andrey Sobolev» в 29 py-файлах; код и работа сервиса не изменились (пересобран, проверен).
  - Атрибуция сохранена там, где обязана быть: `LICENSE` (MIT, копирайт Andrey Sobolev) и README (раздел «Лицензии и авторство» — явное упоминание, что база — haiodo/oaitt).

- [X] Уйти от Makefile и `.env`: сборка через `build.sh`, конфигурация — в `docker-compose.yml`
  - `Dockerfile.cpu` переименован в `Dockerfile` (публичная конвенция для Docker Hub).
  - Все переменные окружения перенесены в блок `environment` compose вместе с комментариями; `.env`/`.env.example`/`Makefile` удалены (бэкап в `.tmp/pruned-files-2026-08-13.tar`).
  - `build.sh` — сборка локального образа или с тегом для Docker Hub (`./build.sh myuser/up-and-run-stt --push`).
  - Исправлен дефолт `AUTH_TOKEN` в config.py: было `"key"` (авторизация молча включалась с общеизвестным токеном при чистом `docker run`), стало пусто = отключена.

- [X] Высушить проект до рабочего минимума (убрать бенчмарки и пр.)
  - Удалены: `benchmark/`, `tests/`, `scripts/`, `docs/upstream/`, `src/batch_transcribe.py`, обучающие/демо-части `vendor/gigaam` (tests, train_utils, triton_scripts, assets, colab).
  - Из образа убраны неиспользуемые `transformers`, `onnx`, `onnxruntime` (проверено: не импортируются в рабочих путях); образ 2.19 ГБ.
  - `factory.py` сведён к gigaam-only; Makefile — только сервисные таргеты; `requirements.txt` соответствует реальным зависимостям.
  - Оставлены: `docs/RU_QUALITY.md` (отчёт о качестве), `sample-data/` (нужен для `make test`), тестовая веб-страница `src/static/index.html`.
  - Бэкап всего удалённого: `.tmp/pruned-backup-2026-08-13.tar.gz` и `.tmp/pruned-backup-vendor-2026-08-13.tar.gz`.

- [X] Публикация за реверс-прокси + встроенная веб-консоль
  - Консоль (микрофон → WAV в браузере, загрузка файла) встроена в сервис: `GET /` ([src/static/index.html](src/static/index.html)).
  - HTTPS обязателен для записи с микрофона: `getUserMedia` работает только в secure context, поэтому за прокси нужен валидный сертификат.
  - `AUTH_TOKEN` включает Bearer-авторизацию транскрипционных эндпоинтов; `/`, `/health` и `/docs` остаются открытыми.
  - ⚠️ Токен требуется и при прямом доступе по `:9007` — клиентам нужно прописать тот же ключ.

- [X] Развернуть GigaAM STT-сервис в Docker (CPU) с OpenAI-совместимым API — см. [README.md](README.md)
  - Форк `haiodo/oaitt`, только движок GigaAM Native (CPU); MLX/WhisperX убраны.
  - `docker compose up -d` — одна команда; `restart: unless-stopped`; healthcheck `/health`.
  - Эндпоинты: `/v1/audio/transcriptions`, `/asr`, `/health` (нативный путь позже стал `/stt/asr`).
  - Веса в named volume `gigaam-models` (не запечены, не докачиваются при рестарте).
  - `AUTH_TOKEN` пуст → авторизация отключена; `MODEL_IDLE_TIMEOUT=0` (тёплая модель).
  - Бенчмарк WER/CER/RTF/RAM + отчёт [docs/RU_QUALITY.md](docs/RU_QUALITY.md).
  - Интеграция с десктопными диктовщиками и OpenAI SDK задокументирована в README.

# DONE
