# DGX Spark в этом проекте

Общие правила работы на споке — в скилле `cody-use-spark` (лок GPU, капы памяти,
восстановление). Здесь — проектная специфика и история инцидентов.

## Что и как запускается

- Код: `rsync` с девбокса в `sparky@192.168.1.188:code/gigam-serving/`
  (исключая `.git .venv .tmp __pycache__ data .hf_token`). На споке код не правят.
- CUDA-образ: `docker build -f Dockerfile.cuda -t gigam-serving-stt-cuda:dev .`
  (сборка на самом споке, arm64 нативно, ~5 мин, 3.7 ГБ).
  База `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04`, `onnxruntime-gpu==1.30.0`
  с PyPI — содержит ядра для sm_121, работает из коробки.
- Веса: том `gigam-serving-models`. Первый раз проще перелить с девбокса
  (`docker run --rm -v gigaam-models:/data alpine tar c ... | ssh sparky docker run -i ... tar x`),
  чем качать с HF (см. инцидент 1).
- Сервис (замеры):

  ```bash
  docker run -d --name gigam-serving-stt-cuda --gpus all --ipc=host -p 127.0.0.1:9007:9007 \
    -v gigam-serving-models:/app/data --memory 24g --memory-swap 24g \
    -e DEVICE=cuda -e CUDA_MEM_LIMIT_MB=8192 -e GIGAAM_MODELS=v3_e2e_ctc,v3_e2e_rnnt \
    -e AUTH_TOKEN=bench -e STREAM_MAX_SESSIONS=64 -e MAX_PENDING_REQUESTS=32 \
    gigam-serving-stt-cuda:dev
  ```

  `CUDA_MEM_LIMIT_MB` — кап арены ORT на GPU (cgroup GPU-память не ловит).
- Замеры под GPU-локом: держатель `flock -n /home/sparky/.cody/gpu.lock -c "timeout 3h sleep infinity"`,
  по завершении убить. Стенд whisper.cpp — `benchmark/streaming/Dockerfile.whispercpp`,
  скрипты — `benchmark/streaming/`. Результаты — `~/code/gigam-serving/.tmp/results-*/`
  на споке, сводка — `docs/STREAMING_QUALITY.md`.

## Инциденты и грабли

1. **2026-09-24, обрезанная загрузка весов.** При первом старте на споке
   `v3_e2e_ctc.onnx` скачался с HF на 138 МБ из 845 и был закэширован; сервис
   падал на `INVALID_PROTOBUF` при каждом старте. Загрузчик переписан: пишет в
   `.part`, сверяет с `Content-Length`, переименовывает только целый файл.
   Лечение старого тома: удалить битый файл из `/app/data/onnx/`.
2. Память GPU в `nvidia-smi` — «Not Supported»; смотреть `free -g` /
   `/proc/meminfo` до и после старта контейнера. Замерено: контейнер GigaAM
   с двумя моделями — 4.0 ГиБ, whisper-server turbo — 2.4 ГиБ.
3. **Entrypoint образов nvidia/cuda** печатает лицензионный баннер в stdout при
   каждом `docker run` — ломает любой скрипт, читающий JSON из контейнера.
   В `Dockerfile.cuda` стоит `ENTRYPOINT []`; для чужих образов —
   `--entrypoint`.
4. **whisper.cpp под CUDA 13 на arm64**: линковка падает на
   `undefined reference to cuGetErrorString` — нужен стаб
   `/usr/local/cuda/lib64/stubs/libcuda.so` (`LIBRARY_PATH` + `-lcuda`);
   собранные `.so` лежат в `build/bin`, а не `build/src`; бинари без
   `--gpus all` не стартуют (нет `libcuda.so.1`). Всё учтено в
   `benchmark/streaming/Dockerfile.whispercpp`.
5. Компиляция whisper.cpp (`-j 8`) на споке ~5 мин, память в норме; во время
   замеров её лучше не запускать — CPU-колонки поплывут.
6. **`pkill -f` через ssh убивает саму ssh-сессию**, если шаблон встречается в
   командной строке `ssh ... 'pkill -f ...'` (exit 255 без вывода). Чистку и
   остановку делать файлом-скриптом на споке (`bash ~/code/<project>/.tmp/cleanup.sh`)
   или через `pgrep`-цикл с шаблоном, которого нет в строке вызова.
7. Соседний сервис `faces-and-persons-worker` держит CUDA-контекст (1.3 ГиБ) и
   простаивает (0 % GPU) — при планировании памяти учитывать, на замеры не влияет,
   пока не обрабатывает задачи; проверять `nvidia-smi --query-compute-apps`.
8. В bash-сценариях с бесконечным фоновым циклом нагрузки голый `wait` ждёт и
   его — ждать только нужные PID (`wait "${pids[@]}"`). И `local a=1 b=$a` в
   одной строке разворачивает `$a` до присвоения: объявлять по одной.
9. **CTranslate2 (faster-whisper) под CUDA 13 на arm64 собирается**, но: (а) он
   использует устаревший `find_package(CUDA)` и свой флаг `CUDA_ARCH_LIST`,
   `CMAKE_CUDA_ARCHITECTURES` игнорирует; «Auto» без GPU в контейнере
   подставляет Jetson 5.3 → `nvcc: Unsupported gpu architecture 'compute_53'`;
   (б) CMake 3.28 не принимает «12.0/12.1» (регулярка на однозначный major) —
   патч модуля в `benchmark/streaming/patch_cmake_arch.py`. Сборка v4.6.0 при
   `-j 10` — всего ~2 мин. Всё в `Dockerfile.fasterwhisper`.
10. **Фичи Whisper нельзя дополнять нулями**: `feature_extractor(w)` без
    `padding=True` возвращает лог-мел только на длину окна; ноль в лог-мел
    пространстве — не тишина, модель на таком входе галлюцинирует («Ролиз.»×28) и
    декодирует втрое дольше. Дополнять надо аудио (`padding=True`).
