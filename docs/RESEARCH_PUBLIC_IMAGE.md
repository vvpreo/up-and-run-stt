# Ресерч: gigaam-stt как публичный Docker-образ

Дата: 2026-08-13. Три направления: конкуренты (self-hosted STT-серверы),
полнота OpenAI-совместимости, практики публикации на Docker Hub.
Выводы сведены в приоритеты в [plans/public-image-roadmap.md](plans/public-image-roadmap.md).

> **Статус на конец 2026-08-13 — уже реализовано:** переезд на ONNX Runtime
> (образ 733 МБ без torch, веса на [vvpreo/gigaam-v3-onnx](https://huggingface.co/vvpreo/gigaam-v3-onnx),
> паритет с torch бит-в-бит), пословные таймстемпы (оба контракта),
> эмоции (`/gigaam/emotion`), ops-фиксы (event loop, 413/429-лимиты,
> non-root, CORS, timing-safe auth), токен-гейт WebUI с показом всех фич,
> тестовый набор 26 тестов. Из пробелов ниже остаются актуальными:
> SSE-стриминг, `/v1/models`, OpenAI-формат ошибок, VAD-чанкование,
> ARM64/CI/GHCR, метрики.

---

## 1. Конкуренты: сравнение фич

Рассмотрены: ahmetoner/whisper-asr-webservice (~3.3k ⭐), speaches (быв.
faster-whisper-server, ~3.6k ⭐), LocalAI (~47k ⭐ на весь проект),
whisper.cpp server, vosk-server (~1.3k ⭐), wyoming-faster-whisper
(интеграция Home Assistant), hwdsl2/docker-whisper (ближайший по формату
«один образ на Docker Hub»).

| Фича | У большинства конкурентов | У нас |
|---|---|---|
| OpenAI-совместимый endpoint | да (кроме vosk/whisper.cpp) | **да** |
| `/v1/audio/translations` | да | нет (модель не переводит — нужен корректный отказ) |
| Стриминг (SSE / WebSocket) | у половины (speaches, vosk, LocalAI) | **нет** |
| Пословные таймстемпы | да (5 из 6) | в работе (ONNX-движок уже умеет) |
| SRT/VTT/verbose_json | да | да (verbose_json без слов — доделывается) |
| Диаризация | только WhisperX-стеки | нет (не table stakes) |
| VAD / длинное аудио | да | чанкование есть, VAD нет |
| `/v1/models` + выбор модели | у продвинутых | выбор есть, `/v1/models` нет |
| GPU-вариант образа | да | нет (CPU-only — осознанная ниша) |
| Multi-arch ARM64 | да | нет |
| Веб-UI | у половины | **да** (тестовая страница) |
| Авторизация | у меньшинства | **да** (Bearer) — наше преимущество |
| Метрики | speaches, LocalAI | нет |
| Non-root контейнер | почти ни у кого | **да** — наше преимущество |

**Ранжированные пробелы** (что есть у большинства, а у нас нет):

1. **Таймстемпы + честные субтитры** — №1 реальный сценарий self-hosted
   STT (субтитрование); клиенты ждут `response_format=srt|vtt|verbose_json`.
2. **SSE-стриминг** (`stream=true`) — главная жалоба на batch-only серверы;
   OpenAI SDK уже поддерживает.
3. **`/v1/models`** — GUI-клиенты (Open WebUI, LibreChat) пробуют его для
   выпадашек моделей; у нас реестр уже есть, endpoint тривиален.
4. **VAD-чанкование длинного аудио** — пользователи публичного образа
   принесут часовые подкасты.
5. **ARM64** — Apple Silicon и ARM-VPS — большая часть аудитории
   self-hosted; после переезда на onnxruntime становится дёшево.
6. GPU-тег — не блокер (CPU-ниша легитимна), диаризация — nice to have.

**Контрпозиционирование**: Bearer-авторизация, non-root, лимиты
аплоада/очереди — этого нет у большинства конкурентов; выносить в README
на Docker Hub как преимущества.

---

## 2. OpenAI Audio API: полнота совместимости

Актуальная поверхность OpenAI (2025–2026): модели `whisper-1`,
`gpt-4o(-mini)-transcribe`, `gpt-4o-transcribe-diarize`.

**Параметры `/v1/audio/transcriptions`**: `file`, `model`, `language`,
`prompt`, `response_format` (`json|text|srt|verbose_json|vtt` +
`diarized_json`), `temperature`, `timestamp_granularities[]` (именно с
квадратными скобками в multipart-имени!), `stream` (SSE, только gpt-4o-модели),
`include[]=logprobs`, `chunking_strategy` (`auto` | server_vad-объект).

**verbose_json** — точные поля: `task`, `language` (полное английское
название языка, «russian»), `duration`, `text`, `segments[]`
(`id, seek, start, end, text, tokens, temperature, avg_logprob,
compression_ratio, no_speech_prob`), `words[]` (`word, start, end`),
`usage` (`{"type":"duration","seconds":n}` у whisper-1).

**SSE-стриминг**: `data: {"type":"transcript.text.delta","delta":"..."}` …
завершение — `transcript.text.done` с полным текстом и `usage`.

**Формат ошибок**, который парсят все SDK:
`{"error": {"message", "type", "param", "code"}}`; OpenAI отдаёт 400 (не
422) на ошибки валидации. Наш FastAPI-формат `{"detail": ...}` ломает
клиентов, которые смотрят на `error.code`.

**Ранжированные пробелы для drop-in совместимости:**

1. **Формат ошибок** (envelope + 400 вместо 422) — максимум пользы за
   минимум усилий: пара exception-handler'ов.
2. **`GET /v1/models` (+ `/v1/models/{id}`)** — самый «прощупываемый»
   endpoint; отдать из готового реестра тривиально.
3. **Терпимость к неизвестным параметрам** — принимать
   `timestamp_granularities[]` (скобки!), `stream`, `include[]`,
   `chunking_strategy`, `temperature` без 422 (принять-и-игнорировать).
4. **Точность полей verbose_json** (см. выше; нейтральные значения там,
   где GigaAM данных не даёт; `language` — полным словом).
5. **SSE `stream=true`** — псевдо-стриминг по готовым сегментам достаточен.
6. `usage` в ответах — дёшево, новые SDK ждут опционально.
7. `/v1/audio/translations` — корректный отказ в OpenAI-формате
   (модель русскоязычная, перевод не поддержан).
8. `include[]`/`chunking_strategy`/diarized_json/Realtime WS — принять и
   игнорировать; полноценно почти никто из self-hosted не делает.

---

## 3. Docker Hub: публикация

**Ключевые факты:**
- onnxruntime имеет официальные aarch64-wheels → после переезда на ONNX
  ARM64-образ становится дешёвым (Apple Silicon, ARM-VPS).
- GitHub Actions: нативные arm64-раннеры бесплатны для публичных реп —
  multi-arch без QEMU.
- Docker Hub: анонимный лимит остаётся 100 pulls/6ч (ужесточение до
  10/час анонсировано, но не введено). Хедж — зеркало на GHCR (без
  лимитов для публичных образов).
- Веса качаются с CDN Сбера (`cdn.chatwm.opensmodel.sberdevices.ru`) —
  анонимно, без SLA; риск достижимости вне РФ. Митигация — зеркало весов
  на HF Hub (решено: `vvpreo/gigaam-v3-onnx`) и/или вариант образа с
  запечёнными весами.

**Найденные блокеры в текущих файлах:**
- Статический ffmpeg захардкожен на amd64 (`...-amd64-static.tar.xz`) —
  сломает arm64-сборку; нужен `ARG TARGETARCH` + пин версии + sha256.
- `torch==2.11.0+cpu` — x86-only пин (уходит вместе с переездом на ONNX;
  multi-arch делать ПОСЛЕ переезда).
- Нет GitHub-remote — CI, GHCR, README-sync, OCI-лейбл `image.source`
  предполагают публичный репозиторий; перед публикацией вычистить
  инфраструктурные детали (домены vvpreo.net, токены).

**Чеклист:**
- Теги: `latest` + семверы из git-тегов (`docker/metadata-action`);
  торч-линейку оставить как `1.x`, ONNX — `2.0.0+`; GPU-тег — не делать
  (CPU-only by design, задокументировать).
- OCI-лейблы (`org.opencontainers.image.*`), пин версий зависимостей
  (lock-файл вместо дублирования Dockerfile/requirements), `.dockerignore`
  почистить от наследия oaitt.
- CI: PR → build amd64 + pytest + smoke (`/health`); тег `v*` →
  multi-arch build + push Hub+GHCR + trivy-скан + README-sync
  (`peter-evans/dockerhub-description`).
- Публичный README для Hub: англоязычный (+русская секция), one-liner
  `docker run`, таблица env, объяснение тома, теги, лицензии (MIT код +
  MIT веса GigaAM).
- В публичном compose-примере — образ с Hub, а не локальный build.

**Порядок исполнения:** GitHub-репо → ffmpeg-arch + лейблы + пины
(вместе с ONNX-миграцией) → CI amd64 → arm64 + двойной пуш → README/Hub +
trivy → опционально вариант с запечёнными весами.
