"""
GET /stats — текущая загрузка сервиса для панели «Загрузка» в консоли.

Отдаёт CPU и память процесса, GPU (загрузка, видеопамять, температура — если
в контейнере есть nvidia-smi), список моделей с устройством и памятью, и
разбивку времени по видам работы за последние секунды (src/services/meter.py).
Рассчитан на опрос раз в секунду: тяжёлые замеры кэшируются.

Ничего не работает в фоне: CPU/память/GPU снимаются в момент запроса, а
пооконный учёт времени включается запросом и сам выключается через 10 с
после последнего. Измеренная цена на i7-8750H + GTX 1050 Ti: запрос ~3 мс,
nvidia-smi ~15 мс раз в секунду, около 1 % одного ядра при открытой панели.
ENABLE_STATS=false отключает ручку совсем.

Открыт без токена, как и /health: в ответе только цифры нагрузки.
"""

import asyncio
import os
import shutil
import subprocess
import threading
import time

import psutil
from fastapi import APIRouter, HTTPException

from src.asr.registry import list_models
from src.config import ENABLE_STATS, GIGAAM_MODELS
from src.routes.stream import active_sessions as active_stream_sessions
from src.services.limits import pending_count
from src.services.meter import meter

router = APIRouter(tags=["Health"])

_proc = psutil.Process(os.getpid())
_lock = threading.Lock()
_cpu_cache = {"t": 0.0, "value": None}
_gpu_cache = {"t": 0.0, "value": None}
_NVIDIA_SMI = shutil.which("nvidia-smi")

# Первый вызов cpu_percent() только запоминает точку отсчёта
_proc.cpu_percent(None)
psutil.cpu_percent(None)


def _num(text: str):
    try:
        return float(text)
    except ValueError:
        return None  # "[N/A]" / "Not Supported" (видеопамять на DGX Spark)


def _cpu() -> dict:
    """CPU процесса и хоста с прошлого замера; не чаще раза в 0.5 с."""
    with _lock:
        now = time.monotonic()
        if _cpu_cache["value"] is None or now - _cpu_cache["t"] >= 0.5:
            cores = psutil.cpu_count() or 1
            proc_pct = _proc.cpu_percent(None)  # 100 = одно ядро целиком
            mem = _proc.memory_info()
            vm = psutil.virtual_memory()
            _cpu_cache.update(
                t=now,
                value={
                    "cpu": {
                        "cores": cores,
                        "process_pct": round(proc_pct, 1),
                        "process_pct_of_total": round(proc_pct / cores, 1),
                        "system_pct": round(psutil.cpu_percent(None), 1),
                        "threads": _proc.num_threads(),
                    },
                    "memory": {
                        "process_rss_mb": round(mem.rss / 2**20),
                        "system_used_mb": round((vm.total - vm.available) / 2**20),
                        "system_total_mb": round(vm.total / 2**20),
                    },
                },
            )
        return _cpu_cache["value"]


def _gpu():
    """GPU через nvidia-smi (вся карта, не только наш процесс); кэш 0.9 с."""
    if not _NVIDIA_SMI:
        return None
    now = time.monotonic()
    if _gpu_cache["value"] is not None and now - _gpu_cache["t"] < 0.9:
        return _gpu_cache["value"]
    try:
        out = subprocess.run(
            [
                _NVIDIA_SMI,
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()[0]
        name, util, used, total, temp = [x.strip() for x in out.split(",")]
        value = {
            "name": name,
            "util_pct": _num(util),
            "mem_used_mb": _num(used),
            "mem_total_mb": _num(total),
            "temp_c": _num(temp),
        }
    except Exception:
        value = None
    _gpu_cache.update(t=now, value=value)
    return value


def _models() -> list:
    limit_mb = int(os.getenv("CUDA_MEM_LIMIT_MB", "0"))
    per_model = limit_mb // max(1, len(GIGAAM_MODELS)) if limit_mb > 0 else None
    out = []
    for name, m in list_models().items():
        info = m.get_info()
        device = info.get("device") or "cpu"
        out.append(
            {
                "name": name,
                "loaded": bool(info.get("loaded")),
                "device": device,
                # прирост RSS процесса при загрузке модели
                "ram_mb": info.get("memory", {}).get("model_memory_mb"),
                # потолок GPU-арены этой модели (доля CUDA_MEM_LIMIT_MB)
                "gpu_budget_mb": per_model if device == "cuda" else None,
            }
        )
    from src.routes.emotion import emotions_available

    if emotions_available():
        from src.asr.onnx_emo import emo_model

        out.append(
            {
                "name": "emo",
                "loaded": emo_model.is_loaded(),
                "device": getattr(emo_model, "device", None) or "cpu",
                "ram_mb": None,
                "gpu_budget_mb": None,
            }
        )
    return out


@router.get("/stats")
async def stats() -> dict:
    """Загрузка сервиса: CPU, память, GPU, модели, время по видам работы."""
    if not ENABLE_STATS:
        raise HTTPException(status_code=404, detail="stats are disabled (ENABLE_STATS=false)")
    # Запрос и есть «наблюдатель»: с него начинается запись истории по окнам
    meter.observe()
    base = _cpu()
    gpu = await asyncio.to_thread(_gpu)
    return {
        "ts": round(time.time(), 2),
        **base,
        "gpu": gpu,
        "models": _models(),
        # доля занятости за последние 5 с по видам работы (см. services/meter.py)
        "activity_window_sec": 5,
        "activity": meter.snapshot(5.0),
        "sessions": {"stream": active_stream_sessions(), "pending_requests": pending_count()},
    }
