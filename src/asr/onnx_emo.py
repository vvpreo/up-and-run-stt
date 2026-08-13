"""
GigaAMEmo на ONNX Runtime — распознавание эмоций в речи.

Модель выдаёт вероятности 4 классов (angry / sad / neutral / positive).
Веса скачиваются с того же зеркала, что и ASR (GIGAAM_ONNX_BASE_URL).
Аудио длиннее EMO_CHUNK_SEC анализируется чанками, вероятности
усредняются со взвешиванием по длительности.
"""

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import numpy as np

from src.asr.features import NumpyFeatureExtractor
from src.asr.onnx_engine import ONNX_BASE_URL, ONNX_VARIANT, _download
from src.config import MODEL_CACHE_DIR, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Максимальная длина куска для одного прохода emo-модели, сек
EMO_CHUNK_SEC = float(os.getenv("EMO_CHUNK_SEC", "25.0"))


class GigaAMOnnxEmo:
    """Ленивая обёртка emo.onnx: load() при первом использовании."""

    def __init__(self) -> None:
        self.session = None
        self.featurizer: Optional[NumpyFeatureExtractor] = None
        self.labels: List[str] = []
        self._lock = Lock()

    def is_loaded(self) -> bool:
        return self.session is not None

    def ensure_loaded(self) -> None:
        with self._lock:
            if self.session is not None:
                return
            import onnxruntime as rt
            from omegaconf import OmegaConf

            onnx_dir = Path(MODEL_CACHE_DIR) / "onnx"
            model_path = _download(
                f"{ONNX_BASE_URL}/emo{ONNX_VARIANT}.onnx",
                onnx_dir / f"emo{ONNX_VARIANT}.onnx",
            )
            cfg_path = _download(f"{ONNX_BASE_URL}/emo.yaml", onnx_dir / "emo.yaml")
            cfg = OmegaConf.load(cfg_path)

            self.labels = list(cfg.id2name)
            self.featurizer = NumpyFeatureExtractor(
                **{k: v for k, v in cfg.preprocessor.items() if k != "_target_"}
            )

            opts = rt.SessionOptions()
            opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            threads = int(os.getenv("OMP_NUM_THREADS", "0"))
            if threads > 0:
                opts.intra_op_num_threads = threads
            opts.log_severity_level = 3
            self.session = rt.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"], sess_options=opts
            )
            logger.info(f"Emo ONNX model loaded ({model_path.stat().st_size >> 20} MB)")

    def classify(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Args:
            audio: float32 numpy, mono, 16 kHz.

        Returns:
            {label: probability}, суммируется в 1.
        """
        self.ensure_loaded()
        audio = np.asarray(audio, dtype=np.float32)
        chunk = int(EMO_CHUNK_SEC * SAMPLE_RATE)

        acc = np.zeros(len(self.labels), dtype=np.float64)
        total = 0
        with self._lock:
            for start in range(0, len(audio), chunk):
                piece = audio[start : start + chunk]
                if len(piece) < SAMPLE_RATE // 4:  # <0.25 c — шумовой хвост
                    continue
                feats = self.featurizer(piece)
                (probs,) = self.session.run(
                    None,
                    {
                        "features": feats[None],
                        "feature_lengths": np.array([feats.shape[1]], dtype=np.int64),
                    },
                )
                acc += probs[0].astype(np.float64) * len(piece)
                total += len(piece)

        if total == 0:
            raise ValueError("Audio is too short for emotion recognition")
        acc /= total
        return {label: round(float(p), 4) for label, p in zip(self.labels, acc)}


# Единственный ленивый инстанс на процесс
emo_model = GigaAMOnnxEmo()
