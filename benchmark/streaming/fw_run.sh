#!/usr/bin/env bash
# Прогон замера faster-whisper на Spark (под GPU-локом, запускать на самом споке):
#   flock -n /home/sparky/.cody/gpu.lock -c "bash benchmark/streaming/fw_run.sh <image> <audio.wav> <out-dir> [llm-model]"
# 1) fw_bench: fp16 и int8_float16, батчи 1/3/6, полный клип;
# 2) при заданной LLM — батч 3 (fp16) на фоне непрерывной генерации, с ток/с LLM.
set -uo pipefail
image="$1"; audio="$2"; out="$3"; llm="${4:-}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
audio_dir="$(cd "$(dirname "$audio")" && pwd)"; audio_name="$(basename "$audio")"
vol=gigam-serving-fasterwhisper-models
OLLAMA=http://127.0.0.1:11434
log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

run_bench() {  # run_bench <label> <extra args...>
  local label="$1"; shift
  docker run --rm --name gigam-serving-fw-tmp --gpus all --ipc=host --memory 24g --memory-swap 24g \
    -v "$vol:/models" -v "$here:/bench:ro" -v "$audio_dir:/audio:ro" -v "$out:/out" "$image" \
    python3 /bench/fw_bench.py --audio "/audio/$audio_name" --out "/out/$label.json" "$@" > /dev/null
}

log "sanity"
docker run --rm --name gigam-serving-fw-tmp --gpus all "$image" python3 -c "import ctranslate2 as c; print('ct2', c.__version__, 'cuda', c.get_cuda_device_count(), c.get_supported_compute_types('cuda'))" 2>&1 | tail -1 >&2
free -g | head -2 >&2

log "bench without LLM"
run_bench fw_base --compute float16,int8_float16 --batches 1,3,6 --runs 5

if [ -n "$llm" ]; then
  tag="${llm//[:\/]/_}"
  log "load LLM $llm"
  curl -s -m 900 "$OLLAMA/api/generate" -d "{\"model\":\"$llm\",\"prompt\":\"Привет\",\"stream\":false,\"options\":{\"num_predict\":8},\"keep_alive\":\"15m\"}" >/dev/null
  docker exec ollama ollama ps >&2; free -g | head -2 >&2
  stop="$out/.llm_run"; : > "$stop"; : > "$out/llm_${tag}.txt"
  ( while [ -f "$stop" ]; do
      curl -s -m 600 "$OLLAMA/api/generate" -d "{\"model\":\"$llm\",\"prompt\":\"Напиши очень длинный подробный рассказ о путешествии по Сибири, не менее 600 слов, без заголовков.\",\"stream\":false,\"options\":{\"num_predict\":400,\"temperature\":0.8},\"keep_alive\":\"15m\"}" \
        | python3 "$here/llm_tok.py" >> "$out/llm_${tag}.txt"
    done ) >/dev/null 2>&1 &
  LLM_PID=$!
  sleep 8
  log "bench under LLM"
  run_bench "fw_under_${tag}" --compute float16 --batches 1,3,6 --runs 5 --skip-full
  rm -f "$stop"; wait "$LLM_PID" 2>/dev/null || true
  python3 - "$out/llm_${tag}.txt" <<'EOF' >&2
import sys
v=[float(l.split()[0]) for l in open(sys.argv[1]) if not l.startswith("err")]
print(f"LLM during fw bench: requests={len(v)} tok/s mean={sum(v)/len(v):.1f} min={min(v):.1f}" if v else "no llm samples")
EOF
  docker exec ollama ollama stop "$llm" >/dev/null 2>&1 || true
fi

python3 - "$out" <<'EOF' >&2
import json, glob, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "fw_*.json"))):
    r = json.load(open(f)); print("==", os.path.basename(f), r["model"], "ct2", r["ct2"])
    for ct, res in r["results"].items():
        print(f"  [{ct}] load {res['load_sec']}s mem {res.get('mem_after_load_gib')} GiB")
        for k, v in res.items():
            if k.startswith("batch"):
                print(f"    {k}: total p50 {v['total_sec']['p50']}s  per-window {v['per_window_sec']}s  {v['windows_per_sec']} win/s  feats {v['features_sec']['p50']}s")
        if "full_clip" in res:
            fc = res["full_clip"]; print(f"    full clip: {fc['wall_sec']}s RTF {fc['rtf']} | {fc['text_head'][:90]}")
EOF
echo "done: $out" >&2
