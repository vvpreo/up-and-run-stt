#!/usr/bin/env bash
# Замер whisper.cpp (CUDA) на окнах псевдо-стриминга: сколько времени занимает
# декод окна 5/10/15 с для large-v3-turbo (и опционально других моделей).
#   bash benchmark/streaming/whisper_bench.sh <image> <audio16k.wav> <out-dir> [models...]
# Модели качаются в том gigam-serving-whisper-models при первом запуске.
set -euo pipefail
image="$1"; audio="$2"; out="$3"; shift 3
models=("$@"); [ ${#models[@]} -eq 0 ] && models=(large-v3-turbo)
mkdir -p "$out"
audio_dir="$(cd "$(dirname "$audio")" && pwd)"; audio_name="$(basename "$audio")"
vol=gigam-serving-whisper-models

for m in "${models[@]}"; do
  docker run --rm --name gigam-serving-whisper-tmp -e LD_LIBRARY_PATH=/usr/local/bin:/usr/local/lib -v "$vol:/models" "$image" bash -c \
    "test -f /models/ggml-$m.bin || /opt/whisper.cpp/models/download-ggml-model.sh $m /models" >&2
done

# Окна: вырезаем из начала файла (там речь) 5/10/15 с — как окно псевдо-стриминга
docker run --rm --name gigam-serving-whisper-tmp -e LD_LIBRARY_PATH=/usr/local/bin:/usr/local/lib -v "$audio_dir:/audio:ro" -v "$out:/out" "$image" bash -c '
  for w in 5 10 15 30; do ffmpeg -v error -y -ss 3 -t $w -i /audio/'"$audio_name"' -ac 1 -ar 16000 -sample_fmt s16 /out/win_$w.wav; done' >&2

# Конфигурации: greedy на окнах 5/10/15/30 с (энкодер whisper.cpp всегда
# паддит до 30 с, поэтому длина окна влияет только на декод); окно 10 с с
# обрезанным audio-ctx (-ac 500 = 10/30 энкодера — известный способ не платить
# за паддинг) и окно 10 с с beam 5 (качество как в бенчмарках).
configs=(
  "win=5 ac=0 bs=1"
  "win=10 ac=0 bs=1"
  "win=15 ac=0 bs=1"
  "win=30 ac=0 bs=1"
  "win=10 ac=500 bs=1"
  "win=10 ac=0 bs=5"
)
for m in "${models[@]}"; do
  for cfg in "${configs[@]}"; do
    eval "$cfg"  # win, ac, bs
    for run in 1 2 3; do
      docker run --rm --name gigam-serving-whisper-tmp -e LD_LIBRARY_PATH=/usr/local/bin:/usr/local/lib --gpus all -v "$vol:/models:ro" -v "$out:/out:ro" "$image" \
        whisper-cli -m /models/ggml-$m.bin -f /out/win_$win.wav -l ru -t 4 -bs $bs -bo $bs -ac $ac -nt 2>&1 \
        | grep -E "total time|encode time|decode time|load time|sample time|batchd time|prompt time|mel time" \
        | sed "s/^/$m $cfg run=$run /" | tee -a "$out/whisper_cli_times.txt" >&2
    done
  done
done

# Пропускная способность сервера при параллельных запросах: whisper-server +
# N одновременных POST окна 10 с (server обрабатывает запросы по очереди —
# это потолок последовательного throughput).
docker rm -f gigam-serving-whisper-server 2>/dev/null || true
docker run -d -e LD_LIBRARY_PATH=/usr/local/bin:/usr/local/lib --name gigam-serving-whisper-server --gpus all -p 127.0.0.1:9008:8080 \
  -v "$vol:/models:ro" "$image" whisper-server -m /models/ggml-${models[0]}.bin -l ru -t 4 --host 0.0.0.0 --port 8080 -nt >&2
sleep 8
for n in 1 3 6; do
  t0=$(date +%s.%N)
  for i in $(seq 1 $n); do
    curl -s -o /dev/null -w "%{time_total}\n" -F "file=@$out/win_10.wav" -F "response_format=text" localhost:9008/inference &
  done | sort -n | awk -v n=$n '{a[NR]=$1} END {printf "server win=10s parallel=%d per-request max=%.3fs\n", n, a[NR]}' | tee -a "$out/whisper_server_times.txt" >&2
  wait
  t1=$(date +%s.%N)
  echo "server win=10s parallel=$n wall=$(echo "$t1 - $t0" | bc)s" | tee -a "$out/whisper_server_times.txt" >&2
  sleep 2
done
docker rm -f gigam-serving-whisper-server >/dev/null
echo "done: $out" >&2
