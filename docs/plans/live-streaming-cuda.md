# План: живая диктовка «буквы текут» + CUDA (DGX Spark)

Дата: 2026-09-24. Статус: **research завершён, план согласовывается, реализация не начата.**

Заказчик функции — соседний проект (голосовой ввод в веб-форме: иконка микрофона
у курсора, текст появляется по мере речи). Требования оттуда: браузер, 2–3
одновременных диктовки, сервис на DGX Spark (GB10, aarch64, 128 ГБ UMA),
лицензия MIT/Apache (допустима некоммерческая).

Ответы пользователя (2026-09-24):

1. **Качество — главный критерий**: минимизировать ручные правки в текстовом
   поле, даже ценой ресурсов. **Смешанная RU/EN речь и IT-термины нужны.**
2. Замеры на Spark запускать.
3. Псевдо-стриминг с GigaAM в другом продукте понравился (качество, ресурсы,
   UX); беспокоит только английский. Рассматриваются варианты: две модели
   одновременно либо «всё закрыть одним Whisper».

Цель в **этом** проекте: доработать сервис так, чтобы (1) он работал на CUDA
в инфраструктуре NVIDIA и (2) в живой диктовке текст появлялся во время речи.
Модель при этом становится **подключаемой**: слой стриминга (VAD, окно,
стабилизация, протокол) один, движков может быть несколько.

---

## 1. Что есть сейчас и в чём разрыв

| | Сейчас | Нужно |
|---|---|---|
| Вход | `WS /stt/stream`, PCM s16le 16 кГц, серверный silero-VAD режет на фразы | то же |
| Выход | текст **готовой фразы** через ~1 с после паузы | промежуточный текст **во время** фразы |
| Железо | CPU, ONNX Runtime `CPUExecutionProvider` | CUDA на DGX Spark (aarch64, sm_121) |
| Языки | русский (GigaAM v3) | RU + EN-термины внутри русской речи |
| Модель | GigaAM v3, офлайновая (нужен законченный отрезок) | потоковая модель или псевдо-стриминг поверх офлайновой |

GigaAM v3 — Conformer с полным self-attention, офлайновая; потокового варианта у
SberDevices нет ни в v3 (11.2025), ни в GigaAM Multilingual (06.2026)
([issue #18](https://github.com/salute-developers/GigaAM/issues/18)).
Word boosting / hotwords тоже нет
([issue #31](https://github.com/salute-developers/GigaAM/issues/31)).

---

## 2. Результаты research

### 2.1 CUDA на DGX Spark — штатный путь есть

- **ONNX Runtime**: с `onnxruntime-gpu` **1.29.0** (08.2026) на PyPI есть
  wheel'ы `manylinux_2_34_aarch64` (py3.11–3.14) с SASS для `sm_121a`; ORT 1.30
  добавил тюнинг под SM121. Нужны CUDA 13.0 + cuDNN 9. Сборка из исходников не
  нужна ([release 1.29](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0),
  [issue #27944](https://github.com/microsoft/onnxruntime/issues/27944)).
  Грабли: INT8-веса на CUDA не брать; `onnxruntime` и `onnxruntime-gpu`
  взаимоисключающи; `TensorrtExecutionProvider` в aarch64-wheel не подтверждён.
- **PyTorch**: `torch>=2.9` из индекса `cu130` (aarch64), sm_121 через
  совместимость с sm_120; без `torch.compile`/flash-attn, SDPA. NGC
  `pytorch:25.10`/`25.11` — «GB10 fully supported». NeMo на Spark работает.
- **whisper.cpp CUDA** на Spark собирается
  (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120;121`, контейнер Ubuntu 24.04)
  ([тред NVIDIA](https://forums.developer.nvidia.com/t/running-whisper-cpp-stt-server-on-dgx-spark-gb10-arm64-cuda-13-via-docker/371803)).
  **CTranslate2 (faster-whisper) официальных aarch64+CUDA wheel не имеет** —
  только сторонние бинарники
  ([assix/ctranslate2-aarch64-cuda13-binaries](https://github.com/assix/ctranslate2-aarch64-cuda13-binaries)).
- **vLLM** на GB10 работает (nightly cu130 в NGC-контейнере), нужен явный
  `--gpu-memory-utilization`, иначе захватит UMA
  ([блог vLLM](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark)).
- **Docker**: `--gpus all --ipc=host --ulimit memlock=-1`, база
  `nvidia/cuda:13.0.x-runtime-ubuntu24.04` (arm64). **cgroups не ограничивают
  CUDA-аллокации** (UMA) — лимиты только в приложении; OOM вешает бокс
  ([known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)).
- **Скорость GB10** (память 273 ГБ/с ≈ ¼ RTX 4090, уровень RTX 3060/4060).
  Публичный бенчмарк, batch 1: parakeet-ctc-0.6b 458× realtime, whisper
  large-v3 ~20–25×
  ([форум](https://forums.developer.nvidia.com/t/running-parakeet-speech-to-text-on-spark/356353/13)).
  Оценки: GigaAM v3 (~220M CTC) ~300–600×; Whisper turbo single-stream ~12–20×,
  батчем заметно больше. **Замеров turbo на Spark в сети нет — мерить самим.**

### 2.2 Бенчмарк, который совпадает с нашей задачей

Habr, 04.08.2026, «бенчмарк 23 ASR-нейросетей для русской айтишной диктовки»
([статья](https://habr.com/ru/articles/1066528/)): два диктора, начитка ~2100
слов + 37 спонтанных диктовок, 24 из них с английскими терминами
(«задеплоили feature в production», «смёржи branch»). Оценка
**Q = 0.65·(100−WER) + 0.10·пунктуация + 0.25·EPI**, где EPI — доля терминов,
сохранённых латиницей (GitHub → 1.0, транслит «гитхаб» → 0). Верхушка
статистически не разделена.

| Модель | Q | WER | EPI | ×RT (5070 Ti) | Примечание |
|---|---|---|---|---|---|
| Breeze-ASR-25 (MediaTek, ft large-v2) | 90.7 | 7.8 | 85 | ×21 | обучена на code-switching zh/en, навык переносится на ru/en |
| **Whisper large-v3-turbo + промпт** | **89.6** | 8.1 | 76 | ×53 | без промпта — без пунктуации, промпт даёт +6.2 Q |
| Whisper turbo RU code-switch (coriollon) | 89.5 | 8.0 | 77 | ×52 | LoRA на синтетике RU+EN-термины |
| Whisper medium | 88.7 | 8.8 | 82 | ×27 | |
| Whisper turbo RU-файнтюны (bond005, antony66) | 84–86 | | | | лучше на чистом RU, **хуже на терминах** |
| Voxtral Mini 4B | 83.4 | ~10.5 | ~70 | ×1.4 | |
| Qwen3-ASR 1.7B | 79.6 | ~12 | ~65 | ×7 | |
| Parakeet-TDT-0.6B-v3 | 75.2 | ~17 | ~50 | ×12 | |
| **GigaAM v3** | **71.8** | ~18.5 | ~40 | ×18 | «SOTA на чистом русском, на терминах — каша»: Gemini → «Jemni» |
| T-one | — | | | | дисквалифицирована: нет пунктуации |
| gpt-4o-mini-transcribe (облако) | 92.1 | 5.7 | 83 | | для калибровки |

Вывод: **в нашей постановке (термины внутри русской речи) семейство Whisper
выигрывает у GigaAM с большим отрывом**, и это единственный публичный замер
именно такого сценария. На чистом русском без терминов GigaAM v3 по-прежнему
точнее всех (наш замер: WER 4.9 % на FLEURS).

Дополнительно по Whisper:
- Промпт с глоссарием работает и измерен (+6.2 Q); faster-whisper имеет
  `initial_prompt`/`hotwords`, whisper.cpp — `--prompt`, WhisperLiveKit —
  `--static-init-prompt` и per-session контекст. У GigaAM и у Voxtral Realtime
  словаря нет.
- Провалы: галлюцинации на тишине («Редактор субтитров А. Семкин»), петли
  повторов у large-v3 (~20 % прогонов без ограничителей). Лечится: внешний
  silero-VAD (уже есть у нас), `condition_on_previous_text=false`,
  `temperature=0`, пороги `no_speech`/`compression_ratio`, отбрасывать окна
  < 0.5 с. Всё это у нас уже частично реализовано в сегментаторе.
- Псевдо-стриминг Whisper через LocalAgreement даёт committed-текст с WER
  +0.2–0.6 к офлайну, задержка 2–3 с из-за дорогого декодера (см. 2.5).

### 2.3 Voxtral и Nemotron простыми словами

**Voxtral Mini 4B Realtime** (Mistral AI, Франция, 02.2026, Apache-2.0).
Это «речевая LLM»: аудио-энкодер 1B + текстовый декодер 3.4B (из Ministral).
Обучена выдавать текст с заданной задержкой (80 мс … 2.4 с, рекомендовано
480 мс), поэтому стримит по-настоящему, слово за словом. 13 языков, включая
русский и английский, язык определяет сама.
- Русский хорош: FLEURS 6.0 % при 480 мс, 5.6 % при 960 мс (Whisper large
  офлайн 5.1 %). Английский 4.9 %.
- Но в бенчмарке диктовки с терминами — Q 83.4, ниже Whisper turbo.
- Риски для нашего кейса: при смене языка внутри потока может **перевести**
  фразу вместо записи (Mistral признал, фикса нет,
  [discussion #21](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/discussions/21));
  галлюцинирует на шуме; словаря терминов в realtime нет; в vLLM
  зафиксирован отказ на 3-й одновременной сессии (закрыт «not planned»,
  [issue #35863](https://github.com/vllm-project/vllm/issues/35863)) и
  зависания длинных сессий.
- Ресурсы: ~9 ГБ весов в bf16 + KV-кэш, только через vLLM (nightly cu130 на
  Spark, без замеров); на 5070 Ti всего ×1.4 realtime — для трёх пользователей
  на GB10 скорее всего не хватит.

**Nemotron 3.5 ASR Streaming 0.6B** (NVIDIA, 06.2026, OpenMDW-1.1).
Классическая потоковая ASR (не LLM): cache-aware FastConformer + RNN-T, 600M
параметров, один чекпойнт на 40 языков, язык задаётся подсказкой на сессию или
`auto`. Чанки 80–1120 мс, пунктуация и регистр встроены, не «сочиняет» на шуме.
- Русский: FLEURS 9.6 % при 560 мс, 9.2 % при 1120 мс — примерно в 1.5–1.8 раза
  больше ошибок, чем у Voxtral/GigaAM. Английский ~8 %.
- На терминах и символах слабее Voxtral (в стороннем тесте 0.77 vs 0.92).
- Плюсы: крошечная (<1 ГБ), ~10 мс вычислений на 100 мс аудио, есть word
  boosting в NeMo (для streaming-пути не подтверждён).
- На Spark официального запуска 3.5 не найдено; путь через NeMo cu130 aarch64
  существует, но хрупкий.

**Лицензия OpenMDW-1.1** — публикует Linux Foundation (05.2026), NVIDIA
переводит на неё Nemotron/Cosmos/GR00T. Пермиссивная «в духе MIT»:
коммерческое использование, изменение и распространение без ограничений,
выходы модели (транскрипты) не ограничены; условие — сохранять текст лицензии
и уведомления при передаче весов; есть оговорка о прекращении лицензии, если
подать иск о нарушении патента/копирайта в связи с моделью. Не одобрена OSI.
Практически — так же свободно, как Apache-2.0
([текст](https://openmdw.ai/license/1-1/), [FAQ](https://openmdw.ai/faq/)).

Для нашей задачи обе уступают Whisper: Voxtral — по терминам, стабильности
vLLM и стоимости; Nemotron — по качеству русского. Остаются как запасные
кандидаты на A/B, если хватит времени.

### 2.4 Гибриды «две модели»

- **Роутинг по языку фразы** (детектор языка → GigaAM для RU, Whisper для EN)
  не решает задачу: термины стоят **внутри русской фразы**, вся фраза
  «русская». Gladia измерила: при смене языка внутри фразы WER ~41 %
  ([статья](https://dev.to/gladia-io/building-real-time-multilingual-asr-with-code-switching-3561)).
  Русские десктопные тулзы (VoiceSwitch, «Писарь») держат GigaAM и Whisper,
  но переключают вручную.
- **Две модели параллельно, выбор по уверенности** — реализаций для RU/EN не
  найдено; на терминах GigaAM «уверенно» выдаёт транслит.
- **ASR + LLM-посткоррекция с глоссарием** — академически работает (дообученные
  корректоры), но zero-shot LLM на диктовке: +1–5 с на фразу, over-correction
  («правит правильный текст»), количественных доказательств снижения правок
  для RU/EN нет. Автор Habr-бенчмарка от неё отказался. Дешёвая альтернатива —
  **словарь алиасов** («пул реквест» → pull request) на постобработке; в одной
  из статей сработал лучше глоссария в промпте.
- **GigaAM-партиалы + Whisper-финал**: партиалы будут показывать GigaAM-транслит,
  финал его перепишет — мерцание ровно на тех словах, ради которых всё
  затевалось. Не имеет смысла.

### 2.5 Псевдо-стриминг поверх офлайновой модели

«Перекодировать растущее окно + LocalAgreement-2» (Liu et al. 2020,
[whisper_streaming](https://github.com/ufal/whisper_streaming),
[WhisperPipe 2026](https://arxiv.org/pdf/2604.25611)): committed-префикс не
меняется, деградация WER +0.2–0.6 абс. Для CTC (GigaAM) приём естественнее,
нестабильность только в последних ~0.3–0.6 с окна. Для Whisper дороже: декодер
авторегрессионный, окно 10 с ≈ 0.5–0.8 с на GB10 (оценка), поэтому такт
1–2 с и задержка коммита 2–3 с. Параметры: окно от последнего среза +
left-context 2–3 с, кап 15–20 с, hop 300–400 мс (CTC) / 1–2 с (Whisper),
LocalAgreement-2 по нормализованным словам, «опасная зона» 500 мс у правого
края, пунктуация фиксируется только финалом фразы.

---

## 3. Рекомендация

**Главная гипотеза: один Whisper large-v3-turbo с промптом-глоссарием и
псевдо-стримингом на CUDA.** Это единственный вариант, у которого есть
измеренное преимущество именно на «русский + IT-термины», есть механизм
глоссария, MIT-лицензия и подтверждённая сборка на Spark (whisper.cpp CUDA).
GigaAM остаётся **вторым профилем** для чистого русского (точнее, в 10–20 раз
дешевле, псевдо-стриминг отзывчивее) — если пользователь без терминов, он
может выбрать его. Voxtral/Nemotron — только если останется время на A/B.

Что должно подтвердить решение:

1. ~~Throughput turbo на GB10~~ — **замерено 2026-09-24, хватает с запасом**
   (см. [`docs/SPARK_BENCHMARK.md`](../SPARK_BENCHMARK.md)): окно 10 с —
   0.12–0.24 с, ~8 окон/с на одном whisper-server; для 3 пользователей с тактом
   0.5–1 с нужно 3–6 окон/с. GigaAM на CUDA: фраза 35 мс, 20 сессий без роста
   задержки, GPU 0–4 %; ограничитель — CPU ≈ 0.1 ядра на сессию.
2. **Реальное число правок** на своей диктовке (RU + термины), а не WER из
   статей: turbo vs turbo-codeswitch vs Breeze vs GigaAM.
3. UX псевдо-стриминга Whisper (коммит через ~1–2 с при такте 0.5–1 с) против
   GigaAM (~0.6–0.7 с после паузы) — приемлемо ли для поля ввода.

---

## 4. План доработки сервиса

### Этап A — CUDA-бэкенд для GigaAM (ONNX Runtime CUDA EP) — **сделано (2026-09-24)**

Реализовано в рамках замеров: выбор провайдера по `DEVICE`, кап
`CUDA_MEM_LIMIT_MB`, поле `device` в `/health`, `Dockerfile.cuda`, compose-профиль
`cuda`, устойчивый загрузчик весов. Проверено на Spark (sm_121, ORT 1.30.0).
Не сделано: fp16-вариант весов (на GPU и fp32 хватает с запасом) и второй
lock-профиль (пока подмена пакета внутри Dockerfile.cuda).


- `src/asr/onnx_engine.py`: провайдеры по `DEVICE` (`auto|cuda|cpu`), для CUDA
  `cudnn_conv_algo_search=HEURISTIC` (как в vendored `_providers_list`);
  фактический провайдер в `/health` (`device`).
- fp16-веса: `scripts/convert_onnx.py --dtype fp16`, суффикс `.fp16` на HF,
  `GIGAAM_ONNX_VARIANT=.fp16`. INT8 на CUDA не использовать.
- `Dockerfile.cuda` (arm64+amd64): база `nvidia/cuda:13.0.x-runtime-ubuntu24.04`
  + python 3.11 + `onnxruntime-gpu>=1.29`; в pyproject extras `cuda`
  (второй lock или маркеры); compose-профиль `cuda` (`--gpus all`, `ipc: host`),
  контейнер `up-and-run-stt-cuda`.
- Память — на уровне приложения (`MODEL_WORKERS=1`), в `/health` RSS и
  `/proc/meminfo`, не `cudaMemGetInfo`.
- Тесты: сюита без изменений; проверка `device == cuda` вручную на Spark
  (в GitHub нет aarch64-GPU раннеров).

Оценка: 1–2 дня.

### Этап B — второй движок: Whisper

- `src/asr/whisper_engine.py` за тем же интерфейсом `ASRModel` (фабрика уже
  является точкой расширения). Рантайм: **faster-whisper (CTranslate2)** —
  собран и замерен на Spark 2026-09-25 (`benchmark/streaming/Dockerfile.fasterwhisper`,
  [`SPARK_BENCHMARK.md` §3.1](../SPARK_BENCHMARK.md)): окно 10 с — 0.05–0.09 с,
  в 1.5–2 раза быстрее whisper.cpp, Python-API встаёт в сервис напрямую, есть
  `initial_prompt`/`hotwords`. Батчинг окон на GB10 выигрыша не даёт (энкодер
  насыщает GPU одним окном) — один воркер, окна по очереди. Запасной —
  whisper.cpp (`whisper-server` отдельным контейнером).
- Модели: `large-v3-turbo` (MIT), опционально `coriollon/…-turbo-russian-codeswitch`
  (Apache-2.0) и `Breeze-ASR-25` (Apache-2.0) как профили `WHISPER_MODEL`.
- Промпт: `WHISPER_PROMPT` (дефолт — билингвальный с пунктуацией по образцу
  promptv4) + `WHISPER_GLOSSARY` (список терминов, до ~50) + per-request
  `prompt` в запросе. Антигаллюцинационные пороги в env.
- Выбор движка на запрос: параметр `model` (`v3_e2e_ctc` → GigaAM,
  `whisper-turbo` → Whisper), `/v1/models` отдаёт оба. Уже существующая
  семантика `model` сохраняется.
- Постобработка: словарь алиасов (`ALIASES_FILE`, «пул реквест» → pull request)
  — применяется к любому движку.

Оценка: 2–3 дня (без учёта сборки whisper.cpp на Spark — ещё ~1 день).

### Этап C — партиалы в `WS /stt/stream` (движок-агностично)

Протокол (обратно совместимо, по умолчанию выключено):
- query `partials=true`; событие `transcript.text.partial`
  `{committed, tentative, seq_phrase}` во время фразы; `transcript.text.delta`
  по-прежнему финал фразы (может переписать пунктуацию/регистр committed-слов,
  не сами слова); `session.created` сообщает `partials`, `hop_ms`.

Алгоритм (`src/services/partial_decoder.py`, без знания о сокете и модели):
окно от последнего среза + left-context 2–3 с, кап 15–20 с; hop из настроек
движка (CTC 400 мс, Whisper 1000–1500 мс) + внеочередной прогон на VAD-конец;
coalescing при занятом GPU; пословные таймстемпы от движка; LocalAgreement-2
по нормализованным словам; «опасная зона» 500 мс; один инференс-воркер с
батчингом окон разных сессий и семафором. На CPU партиалы по умолчанию
выключены.

WebUI: tentative-хвост серым, committed обычным; тумблер «Промежуточный текст».

Оценка: 2–3 дня + день замеров.

### Этап D — исследовательский стенд

`benchmark/streaming/`: compose для Spark, скрипт прогона, метрики (§5).
Вне образа сервиса.

Порядок: A → B → замер throughput turbo (решает всё) → C → сравнение.

---

## 5. Протокол исследования на Spark

### Конфигурации

| | Модель / режим | Как поднимаем | Зачем |
|---|---|---|---|
| G0 | GigaAM v3 e2e_ctc/rnnt, по VAD-фразам, CUDA | наш `Dockerfile.cuda` | baseline качества/стоимости; валидация CUDA EP на sm_121 |
| G1 | GigaAM v3 e2e_ctc + партиалы | наш образ | «нравится в другом продукте» — точка отсчёта UX |
| W0 | Whisper turbo + промпт, по VAD-фразам | whisper.cpp CUDA | качество на терминах, throughput |
| W1 | Whisper turbo + партиалы (такт 1–2 с) | наш образ + whisper.cpp | главная гипотеза |
| W2 | turbo-russian-codeswitch, Breeze-ASR-25 | то же, смена модели | «качественный» профиль, если хватит throughput |
| V (опц.) | Voxtral Realtime 480/960 мс | vLLM nightly cu130, `--gpu-memory-utilization` ≤ 0.3 | настоящий стриминг, проверка 3 сессий |
| N (опц.) | Nemotron 3.5 Streaming, 560 мс, `ru-RU` | NeMo cu130 | дешёвый настоящий стриминг |

### Данные

- **Своя диктовка — главный набор.** Сценарий `benchmark/read-aloud/`
  дополнить блоком «IT-термины внутри русской речи» (30–40 фраз в стиле
  «задеплоили feature в production», названия продуктов, аббревиатуры) и
  блоком спонтанной диктовки; записать хотя бы сессию 1 + этот блок. Эталон —
  с терминами латиницей.
- FLEURS `ru_ru` (те же 30 клипов, что в `RU_QUALITY.md`) — чистый русский,
  сравнимость с CPU-замерами. FLEURS `en_us` 30 клипов — английский.
- Пословные таймстемпы эталона для задержек — forced alignment одним способом
  для всех систем.

### Метрики

1. **Ручные правки**: число операций редактирования (слова: вставить/удалить/
   заменить) от гипотезы до эталона на своей диктовке — прямой критерий
   пользователя. Плюс WER/CER и **доля терминов латиницей** (как EPI).
2. WER committed-текста в потоке против офлайна — цена стриминга.
3. Задержки: first-partial, word-commit (p50/p95), phrase-final.
4. Стабильность: доля partial-событий, изменивших показанный текст.
5. Нагрузка: 1 / 3 / 5 сессий — p95 задержек, RTF, `nvidia-smi dmon`,
   `/proc/meminfo`.
6. Ресурсы: память, старт, образ.

### Порядок и критерии

1. Этап A → G0: CUDA EP работает, RTF и память зафиксированы.
2. Этап B → W0: **замер throughput turbo на GB10** (single и батч ×3, окно
   10 с). Порог для партиалов: ≥ 15 с аудио/с при 3 параллельных окнах.
3. Своя диктовка через G0 и W0 (+W2): таблица правок/WER/терминов. Ожидание по
   литературе: Whisper заметно лучше на терминах, GigaAM лучше на чистом
   русском.
4. Этап C → G1 и W1: задержки и стабильность; порог — WER committed ≤ офлайн
   + 1 абс., commit p95 ≤ 1.5 с (GigaAM) / ≤ 3 с (Whisper), 3 сессии без
   роста p95 > 2×.
5. V и N — если остаётся время; те же данные и метрики.
6. Решение по таблице «правки на своей диктовке / commit-latency / GPU на
   3 сессии / лицензия»; сводка — в `docs/STREAMING_QUALITY.md`.

### Правила на Spark

- Только по явному указанию пользователя, по скиллу `cody-use-spark`: GPU-лок
  «одна нагрузка за раз», капы памяти в приложении (для vLLM — обязательно
  `--gpu-memory-utilization`), `free -h` до и после.
- Контейнеры `gigam-serving-<function>` (`-stt-cuda`, `-whisper`, `-voxtral`,
  `-nemotron`, `-bench`), удаляются после серии.
- Артефакты в `benchmark/results/` (git-ignored).

---

## 6. Риски

- **Throughput Whisper на GB10** неизвестен; если не хватит на 3 сессии с
  партиалами — такт 2–3 с, `medium`, или партиалы только на GigaAM-профиле.
- CTranslate2 на aarch64+CUDA без официальных wheel — собираем сами
  (Dockerfile готов, ~2 мин компиляции); поддержка сборки на нас.
- Галлюцинации Whisper на тишине — закрыты серверным VAD и порогами, но
  проверить на своей диктовке (пауза в середине фразы).
- Псевдо-стриминг Whisper — коммит через 2–3 с; если UX неприемлем, партиалы
  показывать только с GigaAM, а Whisper давать финал (мерцание на терминах —
  осознанная цена).
- Цифры псевдо-стриминга для CTC-Conformer в литературе отсутствуют — мерить.
- Один `uv.lock` на CPU- и CUDA-образ: extras с маркерами или второй lock.
- Voxtral (если дойдёт): eager-режим на sm_121, баг 3-й сессии, захват UMA.

## 7. Ключевые источники

- Habr-бенчмарк диктовки RU/EN, 23 модели (08.2026): https://habr.com/ru/articles/1066528/ ; апрельская статья того же автора: https://habr.com/ru/articles/1024634/
- Whisper RU-файнтюны: https://huggingface.co/coriollon/whisper-large-v3-turbo-russian , https://huggingface.co/landco11/whisper-large-v3-turbo-russian-codeswitch , https://huggingface.co/bond005/whisper-podlodka-turbo ; Breeze-ASR-25: https://huggingface.co/MediaTek-Research/Breeze-ASR-25
- Промптинг Whisper: https://cookbook.openai.com/examples/whisper_prompting_guide ; WhisperLiveKit: https://github.com/QuentinFuxa/WhisperLiveKit ; галлюцинации: https://arxiv.org/abs/2402.08021
- whisper.cpp на Spark: https://forums.developer.nvidia.com/t/running-whisper-cpp-stt-server-on-dgx-spark-gb10-arm64-cuda-13-via-docker/371803 ; CT2 aarch64 CUDA: https://github.com/assix/ctranslate2-aarch64-cuda13-binaries
- ORT 1.29/1.30 aarch64+CUDA13: https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0 , https://github.com/microsoft/onnxruntime/issues/27944
- PyTorch на Spark: https://discuss.pytorch.org/t/nvidia-dgx-spark-support/223677 ; NGC 25.11: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-11.html
- Docker/память на Spark: https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html , https://docs.nvidia.com/dgx/dgx-spark/known-issues.html ; vLLM на Spark: https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
- Бенчмарк ASR на GB10: https://forums.developer.nvidia.com/t/running-parakeet-speech-to-text-on-spark/356353/13
- GigaAM v3 / Multilingual: https://github.com/salute-developers/GigaAM , https://huggingface.co/ai-sage/GigaAM-Multilingual , https://habr.com/ru/companies/sberdevices/articles/973160/
- Voxtral Mini 4B Realtime: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602 , https://arxiv.org/abs/2602.11298 , перевод при смене языка: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/discussions/21 , 3-я сессия в vLLM: https://github.com/vllm-project/vllm/issues/35863
- Nemotron 3.5 ASR Streaming: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b ; OpenMDW-1.1: https://openmdw.ai/license/1-1/ , https://openmdw.ai/faq/
- Code-switching и LID-роутинг: https://dev.to/gladia-io/building-real-time-multilingual-asr-with-code-switching-3561
- LocalAgreement / псевдо-стриминг: https://arxiv.org/abs/2005.11185 , https://arxiv.org/html/2307.14743v2 , https://arxiv.org/pdf/2604.25611 , https://arxiv.org/abs/2406.10052 ; NeMo buffered inference: https://docs.nvidia.com/nemo/speech/nightly/asr/inference.html
