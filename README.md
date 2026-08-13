# gigaam-stt — локальный STT-сервис (GigaAM, CPU) с OpenAI-совместимым API

Локальный сервис распознавания речи (speech-to-text) на базе модели
**GigaAM** (SberDevices, MIT) для **русского языка**, упакованный в один
самодостаточный Docker-сервис с **OpenAI-совместимым HTTP API**.

Развёрнут на **DevBox** (Ubuntu, только CPU — GPU нет). Обслуживает нескольких
потребителей одновременно:

1. **VoiceInk** (десктоп-диктовка на Mac) — через Custom-модель с
   OpenAI-совместимым эндпоинтом.
2. **Внутренние сервисы** — как drop-in замена OpenAI Whisper API.

Основа — форк [`haiodo/oaitt`](https://github.com/haiodo/oaitt) (MIT). Оставлен
только движок **GigaAM Native** под CPU; MLX/WhisperX/Transformers-варианты
убраны за ненадобностью.

> **Приоритет — качество русского.** Замеры WER/CER и выводы о пригодности —
> в [`docs/RU_QUALITY.md`](docs/RU_QUALITY.md). Английский вне фокуса (GigaAM v3
> на нём заведомо слабее).

---

## Быстрый старт

```bash
docker compose up -d          # сборка (первый раз) + запуск
docker compose logs -f        # первый старт скачивает веса модели (~420 МБ)
```

Вся конфигурация — в блоке `environment` файла `docker-compose.yml`
(с комментариями по каждой переменной). Отдельного `.env` нет.

Либо без compose — готовым образом (сборка: `./build.sh [тег] [--push]`):

```bash
docker run -d --name gigaam-stt -p 9007:9007 \
  -v gigaam-models:/app/data \
  -e AUTH_TOKEN=<секрет> \
  gigaam-stt:cpu
```

Значения по умолчанию всех переменных зашиты в образ (ENV в `Dockerfile`),
так что для старта достаточно проброса порта и тома под веса. Образ — **~0.7 ГБ**
(инференс на ONNX Runtime, без PyTorch); веса моделей в образ не входят —
скачиваются при первом старте с [vvpreo/gigaam-v3-onnx](https://huggingface.co/vvpreo/gigaam-v3-onnx)
(зеркало официальных весов, конвертация — `scripts/convert_onnx.py`).
PyTorch-вариант для конвертации весов и сверки качества — `Dockerfile.torch`.

Проверка готовности:

```bash
curl -s http://localhost:9007/health | python3 -m json.tool
```

Ожидаемо: `"model_loaded": true`, `"engine": "gigaam"`.

Сервис слушает `0.0.0.0:9007` и виден по сети (с Mac и из других контейнеров).
После `reboot` поднимается сам (`restart: unless-stopped`).

---

## API

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | OpenAI-совместимый (поле `file`). Для VoiceInk и OpenAI SDK. |
| `POST` | `/gigaam/asr` | Нативный (поле `audio_file`), расширенный ответ: сегменты, слова, метрики. |
| `POST` | `/gigaam/emotion` | Эмоции речи (GigaAMEmo): angry / sad / neutral / positive. |
| `GET`  | `/health` | Статус, модели, очередь, флаги фич, память. |
| `GET`  | `/` | WebUI-консоль: вход по `AUTH_TOKEN`, микрофон/файл, оба контракта, все форматы, слова, эмоции ([src/static/index.html](src/static/index.html)). |

Также есть Swagger UI: `http://localhost:9007/docs`.

### Публичный доступ и авторизация

Сервис опубликован на **https://gigam-stt.dev.vvpreo.net** через реверс-прокси
`dev-nginx` (конфиг `nginx/conf.d/gigam-stt.conf` в проекте `dev-nginx`,
wildcard-сертификат `*.dev.vvpreo.net`). По HTTPS работает и запись с
микрофона на тестовой странице (secure context).

Транскрипционные эндпоинты (`/v1/audio/transcriptions`, `/gigaam/asr`) защищены
Bearer-токеном: ключ задан переменной `AUTH_TOKEN` в `docker-compose.yml` (см. значение
там). Страница `/`, `/health` и `/docs` открыты. Без валидного токена API
отвечает `401`.

```bash
curl -s https://gigam-stt.dev.vvpreo.net/v1/audio/transcriptions \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -F "file=@audio.wav" -F "response_format=json"
```

> ⚠️ Токен требуется **и при прямом доступе** по `:9007` (LAN/туннель) — в
> VoiceInk и других потребителях нужно указать этот же ключ как API key.

### `POST /v1/audio/transcriptions` (OpenAI-совместимый)

Поле файла — **`file`** (как у OpenAI). `response_format`: `json` (по умолчанию),
`text`, `verbose_json`, `srt`, `vtt`.

```bash
# json → {"text": "..."}
curl -s http://localhost:9007/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=json"

# text → просто текст
curl -s http://localhost:9007/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "response_format=text"
```

### `POST /gigaam/asr` (нативный)

Поле файла — **`audio_file`**. Query-параметры: `output` (`json`/`text`/`srt`/`vtt`/`tsv`),
`language`, `word_timestamps`.

```bash
curl -s "http://localhost:9007/gigaam/asr?output=json&language=ru" \
  -F "audio_file=@audio.wav"
```

Ответ (`output=json`) содержит `text`, `language`, при длинном аудио — `segments`,
а также `confidence`/`chars_per_second` (диагностика).

### Входные форматы аудио

Принимаются практически все распространённые форматы: **wav, flac, mp3,
ogg/vorbis, opus, m4a/aac, webm, wma** и другие. Декодирование двухступенчатое:
сначала `libsndfile` (wav/flac/ogg/opus/mp3 — быстрый путь), для остального —
фолбэк на статический бинарь `ffmpeg` в образе; ресемплинг — `scipy`
(polyphase). «Сырые» PCM-байты без заголовка не принимаются — нужен контейнер
(хотя бы WAV). Частота и число каналов входного файла не важны: всё
автоматически приводится к 16 кГц моно.

Влияние формата на скорость — **пренебрежимо мало**. Замер на одном и том же
клипе 137 с (i7-8750H, `v3_e2e_ctc`, лучший из прогонов):

| Формат | Размер | Чистое декодирование | Полный запрос (localhost) |
|---|---|---|---|
| m4a | 1.2 МБ | 89 мс | 6.5 с |
| wma | 2.5 МБ | 113 мс | 6.9 с |
| wav | 12.9 МБ | 157 мс | 6.4 с |
| mp3 | 1.0 МБ | 213 мс | 6.7 с |
| flac | 5.5 МБ | 224 мс | 6.3 с |
| ogg | 0.8 МБ | 241 мс | 6.4 с |
| webm | 1.1 МБ | 266 мс | 6.8 с |
| opus | 1.1 МБ | 631 мс | 6.9 с |

Декодирование занимает 0.09–0.6 с против ~6 с инференса (< 10% запроса);
разброс полного времени между форматами сравним с шумом между прогонами.
Практический вывод: **формат выбирайте по размеру передачи, а не по скорости
декодирования** — по сети сжатый opus/mp3 (~1 МБ) выгоднее wav (~13 МБ),
локально разницы нет.

---

## Конфигурация (переменные окружения)

Все параметры задаются переменными окружения в рантайме — в блоке
`environment` файла `docker-compose.yml` (там же комментарии) или через
`docker run -e`. Ключевые:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `GIGAAM_MODELS` | `v3_e2e_ctc` | Набор моделей инстанса, через запятую (см. ниже). Первая — дефолт. Меняется без пересборки. |
| `DEVICE` | `cpu` | Устройство инференса. GPU нет. |
| `DEFAULT_LANGUAGE` | `ru` | Язык по умолчанию. |
| `HOST` / `PORT` | `0.0.0.0` / `9007` | Bind внутри контейнера. |
| `MODEL_CACHE_DIR` | `/app/data` | Кэш весов (том `gigaam-models`). |
| `MODEL_IDLE_TIMEOUT` | `0` | `0` = не выгружать модель (держим «тёплой»). |
| `MODEL_WORKERS` | `1` | Число инстансов модели в процессе. |
| `AUTH_TOKEN` | *(пусто)* | Пусто = авторизация отключена. **В текущем compose задан** — Bearer-токен обязателен для транскрипционных эндпоинтов. |
| `GIGAAM_MAX_SHORT_AUDIO_SEC` | `25.0` | Длиннее — режется на чанки. |
| `GIGAAM_CHUNK_SEC` / `GIGAAM_MIN_CHUNK_SEC` | `30` / `5` | Размер чанков для длинного аудио. |
| `OMP_NUM_THREADS` | *(не задан)* | Ограничение потоков torch/OpenMP (по умолчанию = число физ. ядер). |
| `MAX_UPLOAD_MB` | `200` | Лимит размера загружаемого аудио; сверх — `413`. `0` = без лимита. |
| `MAX_PENDING_REQUESTS` | `8` | Лимит одновременных/ожидающих транскрипций; сверх — `429`. `0` = без лимита. |
| `CORS_ORIGINS` | *(пусто)* | CORS-origin'ы через запятую (`*` = все). Пусто = CORS выключен. |
| `ENABLE_DOCS` | `true` | `false` — скрыть `/docs`, `/redoc`, `/openapi.json`. |
| `LOG_LEVEL` | `INFO` | Уровень логирования (DEBUG/INFO/WARNING/ERROR). |
| `ENABLE_EMOTIONS` | `true` | `false` — отключить `/gigaam/emotion` (модель эмоций стоит ~1 ГБ RAM, грузится лениво при первом запросе). |
| `INFERENCE_BACKEND` | `onnx` | `torch` — PyTorch-бэкенд (только образ из `Dockerfile.torch`). |

### Модели: набор инстанса и выбор в запросе

Инстанс обслуживает набор моделей из `GIGAAM_MODELS` (например,
`v3_e2e_ctc,v3_e2e_rnnt`): все они загружаются при старте (веса докачиваются
в том по необходимости) и держатся «тёплыми» в RAM — **~1.4 ГБ на каждую**.

Конкретная модель выбирается **полем `model` в запросе** — как принято в
OpenAI-совместимых API:

- имя из набора (`v3_e2e_rnnt`, ...) → эта модель;
- `whisper-1`, пусто или незнакомое имя → модель по умолчанию (первая в списке);
- известный вариант GigaAM, не включённый на инстансе → `400` со списком доступных.

```bash
# OpenAI-эндпоинт: поле формы model
curl -s http://localhost:9007/v1/audio/transcriptions \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -F "file=@audio.wav" -F "model=v3_e2e_rnnt"

# Нативный эндпоинт: query-параметр model
curl -s "http://localhost:9007/gigaam/asr?model=v3_e2e_rnnt&output=json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -F "audio_file=@audio.wav"
```

Набор моделей и дефолт видны в `/health` (`models`, `default_model`).
При `MODEL_WORKERS > 1` пул работает только с моделью по умолчанию.

#### Варианты моделей

| Значение | Пунктуация | Комментарий |
|---|---|---|
| `v3_e2e_ctc` | да | **По умолчанию.** End-to-end, нормализация текста, самый быстрый. |
| `v3_e2e_rnnt` | да | End-to-end, заявлено лучшее качество, медленнее. |
| `v3_ctc` | нет | Без пунктуации, быстрый — запасной для интерактивной диктовки. |
| `v3_rnnt` | нет | Без пунктуации, RNNT. |

Смена набора **без пересборки образа**: правим `GIGAAM_MODELS` в
`docker-compose.yml`, затем `docker compose up -d --force-recreate`.

---

## Персистентность моделей

Веса **не запечены в образ** и **не докачиваются при каждом старте**. Они лежат
в именованном Docker-томе `gigaam-models`, смонтированном в `/app/data`
(`MODEL_CACHE_DIR`). Скачиваются один раз при первом использовании модели и
переживают рестарты и пересборки образа.

- Внутри контейнера: `/app/data/onnx/<model>.onnx` (+ `.yaml`); токенизаторы
  e2e-моделей — `/app/data/gigaam/<model>_tokenizer.model`.
- Источник весов: HF Hub [vvpreo/gigaam-v3-onnx](https://huggingface.co/vvpreo/gigaam-v3-onnx)
  (переопределяется `GIGAAM_ONNX_BASE_URL`); токенизаторы — CDN SberDevices.
  Всё обычным HTTPS, без токенов.
- Размер: fp32-модель ≈ 845 МБ (int8-варианты ≈ 215 МБ — `GIGAAM_ONNX_VARIANT=.int8`).

Посмотреть содержимое тома:

```bash
docker run --rm -v gigaam-models:/data alpine ls -lh /data/gigaam
```

> Контейнер работает от непривилегированного пользователя (uid 1000). Если том
> с весами был создан старой (root) версией образа, один раз выполните:
> `docker run --rm -v gigaam-models:/data alpine chown -R 1000:1000 /data`.

---

## Производительность (этот CPU)

Хост: **Intel Core i7-8750H** (6 ядер / 12 потоков, 2.2 ГГц), 64 ГБ RAM, без GPU.

<!-- PERF_TABLE_START -->
Замерено на 30 клипах FLEURS `ru_ru` (6.2 мин чистой читаной речи). Подробности,
анализ ошибок и оговорки — в [`docs/RU_QUALITY.md`](docs/RU_QUALITY.md).

| Модель | WER | CER | RTF | Скорость | RAM (загружена) |
|---|---|---|---|---|---|
| `v3_e2e_rnnt` | **4.9 %** | 1.3 % | 0.111 | 9.0× | 1.44 GiB |
| `v3_e2e_ctc` *(дефолт)* | 6.2 % | 1.3 % | 0.088 | 11.3× | 1.41 GiB |
| `v3_ctc` | 8.3 % | 2.2 % | 0.091 | 10.9× | 1.42 GiB |

**Рекомендация:** для лучшего качества русского — `v3_e2e_rnnt` (запаса скорости
на CPU достаточно: клип 5–15 с обрабатывается за ~0.5–1.7 с). Дефолтный
`v3_e2e_ctc` — чуть быстрее и почти так же точен.
<!-- PERF_TABLE_END -->

**RTF** (realtime factor) = время обработки / длительность аудио; меньше — лучше
(`0.1` ≈ в 10 раз быстрее реального времени). Для коротких диктовочных клипов
(2–15 с) задержка складывается из RTF·длительность + сетевой оверхед.

### Масштабирование по ядрам (один worker)

Скорость инференса растёт по ядрам **нелинейно** — отдача быстро затухает
(мелкие операции не параллелятся, ядра упираются в общую шину памяти).
Замер: тот же i7-8750H, `v3_e2e_ctc`, клип 20 с, `torch.set_num_threads(N)`,
лучший из 3 прогонов:

| Потоков CPU | Время | Ускорение | Скорость | КПД на ядро |
|---|---|---|---|---|
| 1 | 2.37 с | ×1.0 | 8.5× | 8.5× |
| 2 | 1.37 с | ×1.73 | 14.6× | 7.3× |
| 3 | 1.05 с | ×2.26 | 19.0× | 6.3× |
| 4 | 0.90 с | ×2.63 | 22.3× | 5.6× |
| 6 | 0.83 с | ×2.86 | 24.2× | 4.0× |

Практические выводы для настройки `MODEL_WORKERS` × `OMP_NUM_THREADS`:

- **Минимальная задержка одиночного запроса** (интерактивная диктовка,
  один пользователь): 1 worker со всеми ядрами — текущий дефолт.
  При этом 3–4 ядра дают почти тот же результат, что и 6.
- **Максимальная суммарная пропускная способность** (несколько параллельных
  клиентов): выгоднее несколько worker'ов по 2–3 ядра — например,
  3 worker'а × 2 потока дают в сумме ~44× реального времени против 24× у
  «1 worker × 6 потоков» на тех же ядрах. Цена: +~1.4 ГБ RAM на каждый
  дополнительный worker (своя копия модели) и более медленный каждый
  отдельный запрос; реальная сумма чуть ниже теоретической из-за общей
  шины памяти.

На другом железе абсолютные цифры будут другими, но форма кривой
(затухающая отдача после 2–4 ядер) — типична для CPU-инференса torch.

---

## Интеграция

### VoiceInk (Mac)

VoiceInk → **AI Models → Custom**:

- **API Endpoint:** `http://<адрес-сервиса>:9007/v1/audio/transcriptions`
- **API Key:** значение `AUTH_TOKEN` из `docker-compose.yml` (авторизация включена)
- **Model Name:** `whisper-1` (для OpenAI-эндпоинта поле игнорируется)

Важно: именно `http://` (не `https`), и адрес DevBox должен быть достижим с Mac
(один LAN — прямой IP; удалённый DevBox — SSH port-forward `-L 9007:localhost:9007`).
Проверьте сеть/фаервол. Конкретные реквизиты — см. отдельное сообщение / раздел ниже.

### Другой проект — OpenAI Python SDK

Сервис — drop-in замена OpenAI Whisper API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://<адрес-сервиса>:9007/v1/", api_key="<AUTH_TOKEN>")

with open("audio.wav", "rb") as f:
    r = client.audio.transcriptions.create(
        model="whisper-1",          # игнорируется
        file=f,
        language="ru",
        response_format="json",     # или "text", "verbose_json"
    )
print(r.text)
```

Прямой `curl` — см. раздел [API](#api) выше (и `/v1/...`, и `/gigaam/asr`).

---

## Оценка качества русского

Замеры WER/CER + RTF + RAM по русскому набору и сравнение вариантов модели —
[`docs/RU_QUALITY.md`](docs/RU_QUALITY.md). Бенчмарк-инструментарий из проекта
убран (проект высушен до рабочего минимума); результаты сохранены в отчёте.

---

## Управление

```bash
docker compose up -d / down / logs -f / ps       # сервис
./build.sh                                       # сборка локального образа
./build.sh myuser/gigaam-stt --push              # сборка + публикация на Docker Hub
curl -s localhost:9007/health | python3 -m json.tool   # статус

# быстрый прогон на семпле (токен — из docker-compose.yml):
curl -s localhost:9007/v1/audio/transcriptions \
  -H "Authorization: Bearer <AUTH_TOKEN>" \
  -F "file=@sample-data/Sobolev_Andrey_1_0_00-2_17.ogg" -F "response_format=text"
```

## Тесты

Интеграционные тесты ходят в работающий сервис (`docker compose up -d`,
дождаться `model_loaded: true`):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest tests/ -v
```

Покрытие: /health, авторизация, оба эндпоинта, реестр моделей, все входные
форматы, лимит размера (`413`), backpressure (`429`), живость event loop во
время транскрипции. Другой адрес сервиса — через `STT_BASE_URL` / `STT_AUTH_TOKEN`.

## Troubleshooting

- **`/health` долго не `model_loaded:true`** — первый старт скачивает веса;
  смотрите `docker compose logs -f`. `start_period` healthcheck = 300 с.
- **С Mac не достучаться** — проверьте, что порт `9007` виден с Mac (LAN/SSH),
  используется `http://`, фаервол не режет.
- **Медленно на длинном аудио** — это CPU; для интерактивной диктовки берите
  короткие клипы и при необходимости `v3_ctc` (быстрее, без пунктуации).

## Лицензии и авторство

- Проект базируется на работе **Андрея Соболева** ([haiodo/oaitt](https://github.com/haiodo/oaitt)) —
  оригинального автора сервера OAITT. Этот форк и `haiodo/oaitt` — MIT
  (см. `LICENSE`, копирайт Andrey Sobolev сохранён).
- `gigaam` (vendored в `vendor/gigaam`) — MIT, SberDevices.
