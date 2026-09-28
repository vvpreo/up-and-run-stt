#!/usr/bin/env bash
# Матрица замеров сервиса на GPU-хосте (DGX Spark). Запускается на самом хосте:
#   bash benchmark/streaming/run_matrix.sh <image> <container> <audio.wav> <out-dir> [token]
# Каждый прогон: sysmon в фоне (GPU %, MemAvailable, CPU/RSS контейнера) + loadgen
# в контейнере образа сервиса (там есть websockets/numpy). Результаты — JSON/CSV в out-dir.
set -euo pipefail
image="$1"; container="$2"; audio="$3"; out="$4"; token="${5:-bench}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
audio_dir="$(cd "$(dirname "$audio")" && pwd)"; audio_name="$(basename "$audio")"

run() {  # run <label> <loadgen args...>
  local label="$1"; shift
  echo "=== $label" >&2
  bash "$here/sysmon.sh" "$container" "$out/$label.sys.csv" & local mon=$!
  sleep 2
  # --entrypoint: базовый образ nvidia/cuda печатает лицензионный баннер в
  # stdout через свой entrypoint — он попал бы в JSON.
  docker run --rm --name gigam-serving-bench-tmp --network host --entrypoint python3 -v "$here:/bench:ro" -v "$audio_dir:/audio:ro" "$image" \
    /bench/loadgen.py "$@" --audio "/audio/$audio_name" --token "$token" > "$out/$label.json"
  kill $mon 2>/dev/null || true; wait $mon 2>/dev/null || true
  python3 - "$out/$label.json" "$out/$label.sys.csv" <<'EOF'
import json, sys, csv
r = json.load(open(sys.argv[1]))
rows = list(csv.DictReader(open(sys.argv[2])))
def col(k, f=float):
    v = [f(x[k]) for x in rows if x.get(k) not in (None, "")]
    return v
gpu = col("gpu_util"); cpu = col("cpu_pct"); rss = col("rss_mib"); avail = col("mem_avail_gib")
sysline = (f"gpu_util avg {sum(gpu)/len(gpu):.0f}% max {max(gpu):.0f}% | cpu avg {sum(cpu)/len(cpu):.0f}% max {max(cpu):.0f}% | "
           f"rss max {max(rss):.0f} MiB | mem_avail min {min(avail):.1f} GiB") if rows and gpu else "no sys samples"
if r["mode"] == "ws":
    L = r["latency_end_to_text_sec"]; I = r["inference_sec"]
    print(f'{sys.argv[1]}: sessions={r["sessions"]} phrases={r["phrases_total"]} overflow={r["overflow_total"]} '
          f'lat p50={L.get("p50")} p95={L.get("p95")} max={L.get("max")} | inf p50={I.get("p50")} p95={I.get("p95")} | {sysline}')
else:
    print(f'{sys.argv[1]}: req={r["requests"]} rtf p50={r["rtf_per_request"]["p50"]} max={r["rtf_per_request"]["max"]} '
          f'aggregate={r["aggregate_speed_x_realtime"]}x | {sysline}')
EOF
  sleep 3
}

# Прогрев
run warmup ws --sessions 1 --limit-sec 15 > /dev/null 2>&1 || true

# Офлайн: RTF одного запроса и агрегированная скорость при 3 и 6 параллельных
for m in v3_e2e_ctc v3_e2e_rnnt; do
  run "http_${m}_1" http --sessions 1 --model "$m" --url http://localhost:9007/v1/audio/transcriptions
  run "http_${m}_3" http --sessions 3 --model "$m" --url http://localhost:9007/v1/audio/transcriptions
done
run "http_v3_e2e_ctc_6" http --sessions 6 --model v3_e2e_ctc --url http://localhost:9007/v1/audio/transcriptions

# Живые сессии в реальном времени: масштабирование по числу сессий
for n in 1 3 5 10 20; do
  run "ws_v3_e2e_ctc_${n}" ws --sessions "$n" --model v3_e2e_ctc --stagger 1.3
done
for n in 1 3 10; do
  run "ws_v3_e2e_rnnt_${n}" ws --sessions "$n" --model v3_e2e_rnnt --stagger 1.3
done
echo "done: $out" >&2
