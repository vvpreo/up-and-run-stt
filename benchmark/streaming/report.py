#!/usr/bin/env python3
"""
Сводит результаты run_matrix.sh (loadgen JSON + sysmon CSV) в markdown-таблицы.
    python3 benchmark/streaming/report.py <results-dir>
"""

import csv
import glob
import json
import os
import sys


def sys_stats(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path)))
    def col(k):
        return [float(r[k]) for r in rows if r.get(k) not in (None, "")]
    gpu, cpu, rss, avail = col("gpu_util"), col("cpu_pct"), col("rss_mib"), col("mem_avail_gib")
    if not gpu:
        return {}
    return {
        "gpu_avg": sum(gpu) / len(gpu), "gpu_max": max(gpu),
        "cpu_avg": sum(cpu) / len(cpu) if cpu else 0, "cpu_max": max(cpu) if cpu else 0,
        "rss_max": max(rss) if rss else 0, "avail_min": min(avail) if avail else 0,
    }


def main(d: str) -> None:
    ws, http = [], []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        label = os.path.basename(f)[:-5]
        if label == "warmup":
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        s = sys_stats(os.path.join(d, label + ".sys.csv"))
        (ws if r["mode"] == "ws" else http).append((label, r, s))

    if http:
        print("| Прогон | Параллельных запросов | RTF на запрос p50 / max | Суммарно ×RT | CPU avg / max | RSS max |")
        print("|---|---|---|---|---|---|")
        for label, r, s in http:
            print(f"| `{label}` | {r['requests']} | {r['rtf_per_request']['p50']} / {r['rtf_per_request']['max']} | "
                  f"{r['aggregate_speed_x_realtime']}× | {s.get('cpu_avg', 0):.0f}% / {s.get('cpu_max', 0):.0f}% | "
                  f"{s.get('rss_max', 0):.0f} MiB |")
        print()
    if ws:
        print("| Прогон | Сессий | Фраз | Overflow | Задержка конец-фразы→текст p50 / p95 / max, с | Инференс p50 / p95, с | GPU avg / max | CPU avg / max | RSS max | MemAvail min |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for label, r, s in ws:
            L, I = r["latency_end_to_text_sec"], r["inference_sec"]
            print(f"| `{label}` | {r['sessions']} | {r['phrases_total']} | {r['overflow_total']} | "
                  f"{L.get('p50')} / {L.get('p95')} / {L.get('max')} | {I.get('p50')} / {I.get('p95')} | "
                  f"{s.get('gpu_avg', 0):.0f}% / {s.get('gpu_max', 0):.0f}% | {s.get('cpu_avg', 0):.0f}% / {s.get('cpu_max', 0):.0f}% | "
                  f"{s.get('rss_max', 0):.0f} MiB | {s.get('avail_min', 0):.1f} GiB |")
            if r.get("errors"):
                print(f"|  | errors: {r['errors'][:3]} |")


if __name__ == "__main__":
    main(sys.argv[1])
