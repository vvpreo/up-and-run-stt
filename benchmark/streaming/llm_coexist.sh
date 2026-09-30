#!/usr/bin/env bash
# Соседство STT с LLM на одной GPU (DGX Spark, ollama). Запускается на самом споке
# под GPU-локом:
#   bash benchmark/streaming/llm_coexist.sh <stt-image> <whisper-image> <audio.wav> <out-dir> <llm-model> [llm-model2...]
# Сценарий: baseline STT без LLM -> LLM в одиночку (ток/с) -> STT при непрерывной
# генерации LLM (задержки STT + ток/с LLM в это время) -> whisper-server 3 окна
# с LLM и без. Память: LLM должна влезать в кап контейнера ollama.
set -uo pipefail
stt_image="$1"; whisper_image="$2"; audio="$3"; out="$4"; shift 4
llms=("$@")
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
audio_dir="$(cd "$(dirname "$audio")" && pwd)"; audio_name="$(basename "$audio")"
OLLAMA=http://127.0.0.1:11434

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

# ---- LLM-нагрузка: последовательные запросы с длинной генерацией, пока есть стоп-файл
llm_loop() {  # llm_loop <model> <tag>  -> глобальная LLM_PID
  local model="$1"
  local tag="$2"
  local stop="$out/.llm_run_$tag"
  : > "$stop"
  # stdout/stderr фонового цикла закрыты явно: иначе он держал бы пайп
  # вызывающей оболочки.
  (
    while [ -f "$stop" ]; do
      curl -s -m 600 "$OLLAMA/api/generate" -d "{\"model\":\"$model\",\"prompt\":\"Напиши очень длинный подробный рассказ о путешествии по Сибири, не менее 600 слов, без заголовков.\",\"stream\":false,\"options\":{\"num_predict\":400,\"temperature\":0.8},\"keep_alive\":\"15m\"}" \
        | python3 "$here/llm_tok.py" >> "$out/llm_${tag}.txt"
    done
  ) >/dev/null 2>&1 &
  LLM_PID=$!
}
llm_stop() { rm -f "$out/.llm_run_$1"; wait "$LLM_PID" 2>/dev/null || true; }
llm_summary() { python3 - "$1" <<'EOF'
import sys
v=[float(l.split()[0]) for l in open(sys.argv[1]) if not l.startswith("err")]
n=[int(l.split()[1]) for l in open(sys.argv[1]) if not l.startswith("err")]
print(f"requests={len(v)} tok/s mean={sum(v)/len(v):.1f} min={min(v):.1f} max={max(v):.1f} tokens={sum(n)}" if v else "no llm samples")
EOF
}

stt_run() {  # stt_run <label> <sessions>
  local label="$1" n="$2"
  bash "$here/sysmon.sh" gigam-serving-stt-cuda "$out/$label.sys.csv" & local mon=$!
  sleep 2
  docker run --rm --name gigam-serving-bench-tmp --network host --entrypoint python3 -v "$here:/bench:ro" -v "$audio_dir:/audio:ro" "$stt_image" \
    /bench/loadgen.py ws --audio "/audio/$audio_name" --token bench --sessions "$n" --model v3_e2e_ctc --stagger 1.3 > "$out/$label.json"
  kill $mon 2>/dev/null; wait $mon 2>/dev/null
  python3 - "$out/$label.json" <<'EOF'
import json,sys; r=json.load(open(sys.argv[1])); L=r["latency_end_to_text_sec"]; I=r["inference_sec"]
print(f'{sys.argv[1]}: sessions={r["sessions"]} phrases={r["phrases_total"]} overflow={r["overflow_total"]} lat p50={L["p50"]} p95={L["p95"]} max={L["max"]} | inf p50={I.get("p50")} p95={I.get("p95")}')
EOF
}

whisper_run() {  # whisper_run <label>: 5 повторов по 3 параллельных окна 10 с
  local label="$1"
  : > "$out/$label.txt"
  for rep in 1 2 3 4 5; do
    t0=$(date +%s.%N)
    # ждать только свои curl'ы: голый `wait` дождался бы и бесконечного LLM-цикла
    local pids=()
    for i in 1 2 3; do curl -s -m 120 -o /dev/null -F "file=@$out/win_10.wav" -F "response_format=text" localhost:9008/inference & pids+=($!); done
    wait "${pids[@]}"
    echo "$(echo "$(date +%s.%N) - $t0" | bc)" >> "$out/$label.txt"
    sleep 1
  done
  python3 -c "import sys; v=[float(x) for x in open('$out/$label.txt')]; print('$label: 3 windows wall sec:', ' '.join(f'{x:.2f}' for x in v), f'| mean {sum(v)/len(v):.2f}')" >&2
}

# ---- подготовка
log "start STT"
docker rm -f gigam-serving-stt-cuda >/dev/null 2>&1 || true
docker run -d --name gigam-serving-stt-cuda --gpus all --ipc=host -p 127.0.0.1:9007:9007 \
  -v gigam-serving-models:/app/data --memory 24g --memory-swap 24g \
  -e DEVICE=cuda -e CUDA_MEM_LIMIT_MB=4096 -e GIGAAM_MODELS=v3_e2e_ctc,v3_e2e_rnnt \
  -e AUTH_TOKEN=bench -e MODEL_IDLE_TIMEOUT=0 -e STREAM_MAX_SESSIONS=64 -e MAX_PENDING_REQUESTS=32 "$stt_image" >/dev/null
for i in $(seq 1 40); do sleep 3; curl -s -m 3 localhost:9007/health | grep -q '"model_loaded": *true' && break; done
docker run --rm --name gigam-serving-whisper-tmp -e LD_LIBRARY_PATH=/usr/local/bin:/usr/local/lib -v "$audio_dir:/audio:ro" -v "$out:/out" "$whisper_image" \
  ffmpeg -v error -y -ss 3 -t 10 -i "/audio/$audio_name" -ac 1 -ar 16000 -sample_fmt s16 /out/win_10.wav
docker rm -f gigam-serving-whisper-server >/dev/null 2>&1 || true
docker run -d --name gigam-serving-whisper-server -e LD_LIBRARY_PATH=/usr/local/bin:/usr/local/lib --gpus all -p 127.0.0.1:9008:8080 \
  -v gigam-serving-whisper-models:/models:ro "$whisper_image" whisper-server -m /models/ggml-large-v3-turbo.bin -l ru -t 4 --host 0.0.0.0 --port 8080 -nt >/dev/null
sleep 10
curl -s -o /dev/null -F "file=@$out/win_10.wav" localhost:9008/inference  # прогрев
free -g | head -2 >&2
# фоновый монитор соседнего GPU-жильца (faces-and-persons): память на GPU, CPU, общая загрузка GPU
faces_pid=$(docker inspect -f "{{.State.Pid}}" faces-and-persons-worker-1 2>/dev/null || echo 0)
: > "$out/.monitor"
( echo "ts,faces_gpu_mib,faces_cpu_pct,gpu_util" > "$out/faces_activity.csv"
  while [ -f "$out/.monitor" ]; do
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F", " -v p="$faces_pid" "\$1==p {print \$2}")
    c=$(docker stats --no-stream --format "{{.CPUPerc}}" faces-and-persons-worker-1 2>/dev/null | tr -d %)
    g=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d " ")
    echo "$(date +%s),${m:-0},${c:-0},$g" >> "$out/faces_activity.csv"; sleep 5
  done ) >/dev/null 2>&1 &
MON_PID=$!

# ---- baseline без LLM
log "baseline STT (LLM idle)"
stt_run base_ws3 3
whisper_run base_whisper3

for llm in "${llms[@]}"; do
  tag="${llm//[:\/]/_}"
  log "load LLM $llm"
  curl -s -m 900 "$OLLAMA/api/generate" -d "{\"model\":\"$llm\",\"prompt\":\"Привет\",\"stream\":false,\"options\":{\"num_predict\":8},\"keep_alive\":\"15m\"}" >/dev/null
  docker exec ollama ollama ps >&2; free -g | head -2 >&2

  log "LLM alone 60s"
  llm_loop "$llm" "${tag}_alone"; sleep 60; llm_stop "${tag}_alone"
  echo "LLM $llm alone: $(llm_summary "$out/llm_${tag}_alone.txt")" >&2

  log "STT ws3 under LLM load"
  llm_loop "$llm" "${tag}_ws3"
  sleep 5; stt_run "ws3_under_${tag}" 3
  llm_stop "${tag}_ws3"
  echo "LLM $llm during STT ws3: $(llm_summary "$out/llm_${tag}_ws3.txt")" >&2

  log "whisper under LLM load"
  llm_loop "$llm" "${tag}_whisper"
  sleep 5; whisper_run "whisper3_under_${tag}"
  llm_stop "${tag}_whisper"
  echo "LLM $llm during whisper: $(llm_summary "$out/llm_${tag}_whisper.txt")" >&2

  docker exec ollama ollama stop "$llm" >/dev/null 2>&1 || true
  sleep 5
done

log "cleanup"
rm -f "$out/.monitor"; wait "$MON_PID" 2>/dev/null || true
docker rm -f gigam-serving-whisper-server gigam-serving-stt-cuda >/dev/null 2>&1 || true
echo "done: $out" >&2
