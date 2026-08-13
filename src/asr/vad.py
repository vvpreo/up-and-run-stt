"""
Silero-VAD (v5, ONNX) — детекция речи для умного чанкования.

Минимальный самодостаточный враппер (без pip-пакета silero-vad и torch):
модель ~2.3 МБ запечена в образ (/app/vad/silero_vad.onnx, зеркало —
vvpreo/gigaam-v3-onnx), онна обрабатывает окна по 512 сэмплов (32 мс @16кГц)
с рекуррентным состоянием и отдаёт вероятность речи на окно.

Модель: snakers4/silero-vad, MIT.
"""

import logging
import os
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple

import numpy as np

from src.config import SAMPLE_RATE

logger = logging.getLogger(__name__)

# Путь к запечённой в образ модели; фолбэк — том (докачивается движком)
VAD_MODEL_PATH = os.getenv("VAD_MODEL_PATH", "/app/vad/silero_vad.onnx")

# Порог вероятности речи (ниже — тишина)
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))

_FRAME = 512    # сэмплов на окно (32 мс при 16 кГц)
_CONTEXT = 64   # v5 требует 64 сэмпла контекста перед окном (вход = 576)


class SileroVad:
    """Ленивая обёртка silero_vad.onnx: audio -> вероятности речи по окнам."""

    def __init__(self) -> None:
        self.session = None
        self._lock = Lock()

    def available(self) -> bool:
        return Path(VAD_MODEL_PATH).exists()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self.session is not None:
                return
            import onnxruntime as rt

            opts = rt.SessionOptions()
            opts.log_severity_level = 3
            opts.intra_op_num_threads = 1  # модель крошечная, потоки не нужны
            self.session = rt.InferenceSession(
                VAD_MODEL_PATH, providers=["CPUExecutionProvider"], sess_options=opts
            )
            logger.info(f"Silero VAD loaded from {VAD_MODEL_PATH}")

    def speech_probs(self, audio: np.ndarray) -> np.ndarray:
        """Вероятность речи для каждого 512-сэмплового окна (float32)."""
        self._ensure_loaded()
        n_frames = len(audio) // _FRAME
        if n_frames == 0:
            return np.zeros(0, dtype=np.float32)

        probs = np.empty(n_frames, dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(_CONTEXT, dtype=np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        audio = np.asarray(audio, dtype=np.float32)
        with self._lock:
            for i in range(n_frames):
                frame = audio[i * _FRAME : (i + 1) * _FRAME]
                inp = np.concatenate([context, frame])[None]  # (1, 576)
                out, state = self.session.run(
                    None, {"input": inp, "state": state, "sr": sr}
                )
                probs[i] = out[0, 0]
                context = frame[-_CONTEXT:]
        return probs

    def speech_regions(
        self,
        audio: np.ndarray,
        threshold: Optional[float] = None,
        min_silence_sec: float = 0.4,
        pad_sec: float = 0.15,
    ) -> List[Tuple[int, int]]:
        """
        Границы речевых участков в сэмплах: [(start, end), ...].

        Соседние участки, разделённые тишиной короче min_silence_sec,
        сливаются; каждый участок расширяется на pad_sec с обеих сторон.
        """
        threshold = threshold if threshold is not None else VAD_THRESHOLD
        probs = self.speech_probs(audio)
        speech = probs >= threshold

        regions: List[Tuple[int, int]] = []
        start = None
        for i, is_speech in enumerate(speech):
            if is_speech and start is None:
                start = i
            elif not is_speech and start is not None:
                regions.append((start, i))
                start = None
        if start is not None:
            regions.append((start, len(speech)))

        # окна -> сэмплы, с паддингом
        pad = int(pad_sec * SAMPLE_RATE)
        sampled = [
            (max(0, s * _FRAME - pad), min(len(audio), e * _FRAME + pad))
            for s, e in regions
        ]

        # слить участки с короткой тишиной между ними
        min_gap = int(min_silence_sec * SAMPLE_RATE)
        merged: List[Tuple[int, int]] = []
        for s, e in sampled:
            if merged and s - merged[-1][1] < min_gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        return merged


# Единственный ленивый инстанс на процесс
silero_vad = SileroVad()
