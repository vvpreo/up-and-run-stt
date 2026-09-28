#!/usr/bin/env bash
# Сэмплер нагрузки хоста во время замеров: раз в секунду пишет CSV-строку
#   ts, gpu_util_%, sm_clock_MHz, mem_available_GiB, <container> cpu_%, <container> rss_MiB
# Использование: sysmon.sh <container-name> <out.csv>   (остановить — kill/Ctrl-C)
#
# Память GPU на DGX Spark берётся из /proc/meminfo, а не из nvidia-smi:
# на единой памяти nvidia-smi показывает "Not Supported", а cudaMemGetInfo врёт.
set -u
name="$1"; out="$2"
echo "ts,gpu_util,sm_mhz,mem_avail_gib,cpu_pct,rss_mib" > "$out"
while true; do
  ts=$(date +%s)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' ' ')
  avail=$(awk '/MemAvailable/ {printf "%.2f", $2/1048576}' /proc/meminfo)
  stats=$(docker stats --no-stream --format "{{.CPUPerc}} {{.MemUsage}}" "$name" 2>/dev/null | head -1)
  cpu=$(echo "$stats" | awk '{gsub("%","",$1); print $1}')
  rss=$(echo "$stats" | awk '{v=$2; if (v ~ /GiB/) {gsub("GiB","",v); printf "%.0f", v*1024} else {gsub("MiB","",v); printf "%.0f", v}}')
  echo "$ts,${gpu// /,},$avail,${cpu:-},${rss:-}" >> "$out"
  sleep 1
done
