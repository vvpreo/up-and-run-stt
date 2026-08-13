"""
ONNX Runtime движок GigaAM — инференс без PyTorch.

Конвейер: numpy log-mel фичи (src/asr/features.py) -> ONNX encoder ->
numpy CTC-декодирование (с кадрами токенов) -> текст + пословные таймстемпы.

Поддерживается CTC-семейство (v3_ctc, v3_e2e_ctc). RNNT-варианты требуют
трёх сессий и пошагового декодера — планируются следующим этапом.

Файлы модели ({name}.onnx + {name}.yaml, раскладка vendored `to_onnx`)
ищутся в MODEL_CACHE_DIR/onnx; при отсутствии скачиваются с
GIGAAM_ONNX_BASE_URL. Токенизатор (sentencepiece, только для e2e-моделей)
скачивается с CDN SberDevices, как и в torch-движке.
"""

import logging
import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from src.asr.base import ASRModel
from src.asr.features import NumpyFeatureExtractor
from src.config import (
    DEFAULT_MODEL,
    GIGAAM_CHUNK_SEC,
    GIGAAM_MAX_SHORT_AUDIO_SEC,
    GIGAAM_MIN_CHUNK_SEC,
    MODEL_CACHE_DIR,
    SAMPLE_RATE,
    VAD_CHUNKING,
)
from src.models.schemas import Segment, TranscriptionResponse, WordTimestamp

logger = logging.getLogger(__name__)

# Откуда скачивать .onnx-веса (без завершающего /). Файлы: {name}.onnx, {name}.yaml
# По умолчанию — наше зеркало (конвертация: scripts/convert_onnx.py).
ONNX_BASE_URL = os.getenv(
    "GIGAAM_ONNX_BASE_URL",
    "https://huggingface.co/vvpreo/gigaam-v3-onnx/resolve/main",
)
# Суффикс варианта весов: "" (fp32) или ".int8" (квантизованные)
ONNX_VARIANT = os.getenv("GIGAAM_ONNX_VARIANT", "")

_SBER_CDN = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"

SUPPORTED_ONNX_MODELS = {"v3_ctc", "v3_e2e_ctc", "v3_rnnt", "v3_e2e_rnnt"}

# Максимум эмитов на кадр в greedy RNNT-декоде (как в vendored onnx_utils)
_MAX_LETTERS_PER_FRAME = 3


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    logger.info(f"Downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as src, open(tmp, "wb") as out:
        while chunk := src.read(1 << 20):
            out.write(chunk)
    tmp.rename(dest)
    return dest


class _Tokenizer:
    """Мини-токенизатор: sentencepiece (e2e) или посимвольный словарь."""

    def __init__(self, model_path: Optional[str], vocab: Optional[List[str]]):
        if model_path:
            from sentencepiece import SentencePieceProcessor

            self.sp = SentencePieceProcessor()
            self.sp.load(model_path)
            self.vocab = None
        else:
            assert vocab, "charwise tokenizer requires vocabulary"
            self.sp = None
            self.vocab = list(vocab)

    def __len__(self) -> int:
        return self.sp.get_piece_size() if self.sp else len(self.vocab)

    def decode(self, ids: List[int]) -> str:
        if self.sp:
            return self.sp.decode(ids)
        return "".join(self.vocab[i] for i in ids)

    def id_to_str(self, token_id: int) -> str:
        if self.sp:
            return self.sp.id_to_piece(token_id)
        return self.vocab[token_id]


def _group_words(
    tokenizer: _Tokenizer,
    token_ids: List[int],
    token_frames: List[int],
    frame_shift: float,
    offset_sec: float,
) -> List[WordTimestamp]:
    """
    Группирует токены в слова с таймстемпами (порт vendored
    `timestamps_utils.frames_to_words`, без torch-зависимостей).
    """
    words: List[WordTimestamp] = []
    chars: List[str] = []
    frames: List[int] = []

    def commit() -> None:
        if not chars:
            return
        text = "".join(chars).strip()
        if text:
            words.append(
                WordTimestamp(
                    word=text,
                    start=round(offset_sec + frames[0] * frame_shift, 3),
                    end=round(offset_sec + (frames[-1] + 1) * frame_shift, 3),
                )
            )
        chars.clear()
        frames.clear()

    for token_id, frame in zip(token_ids, token_frames):
        piece = tokenizer.id_to_str(token_id)
        if piece.startswith("▁"):
            commit()
            piece = piece[1:]
        elif piece == " ":
            commit()
            continue
        chars.append(piece)
        frames.append(frame)

    commit()
    return words


class GigaAMOnnxASR(ASRModel):
    """ASR-движок на ONNX Runtime (интерфейс ASRModel, как у torch-движка)."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        super().__init__()
        self.model = None  # rt.InferenceSession; None = не загружена
        self.model_name = model_name or DEFAULT_MODEL
        if self.model_name not in SUPPORTED_ONNX_MODELS:
            raise ValueError(
                f"ONNX backend supports {sorted(SUPPORTED_ONNX_MODELS)}; "
                f"got '{self.model_name}'."
            )
        self.tokenizer: Optional[_Tokenizer] = None
        self.featurizer: Optional[NumpyFeatureExtractor] = None
        self._blank_id: int = 0
        self._pred_sess = None
        self._joint_sess = None

    # ------------------------------------------------------------------ load

    @property
    def _is_rnnt(self) -> bool:
        return "rnnt" in self.model_name

    def load_model(self) -> None:
        import onnxruntime as rt
        from omegaconf import OmegaConf

        onnx_dir = Path(MODEL_CACHE_DIR) / "onnx"
        name = self.model_name
        if self._is_rnnt:
            # RNNT: три сессии (encoder / decoder / joint)
            parts = {}
            for part in ("encoder", "decoder", "joint"):
                parts[part] = _download(
                    f"{ONNX_BASE_URL}/{name}_{part}{ONNX_VARIANT}.onnx",
                    onnx_dir / f"{name}_{part}{ONNX_VARIANT}.onnx",
                )
            model_path = parts["encoder"]
        else:
            model_path = _download(
                f"{ONNX_BASE_URL}/{name}{ONNX_VARIANT}.onnx",
                onnx_dir / f"{name}{ONNX_VARIANT}.onnx",
            )
        cfg_path = _download(f"{ONNX_BASE_URL}/{name}.yaml", onnx_dir / f"{name}.yaml")
        cfg = OmegaConf.load(cfg_path)

        self.featurizer = NumpyFeatureExtractor(
            **{k: v for k, v in cfg.preprocessor.items() if k != "_target_"}
        )

        # Токенизатор: e2e — sentencepiece с CDN Сбера; иначе словарь из yaml
        if "e2e" in name:
            tok_path = _download(
                f"{_SBER_CDN}/{name}_tokenizer.model",
                Path(MODEL_CACHE_DIR) / "gigaam" / f"{name}_tokenizer.model",
            )
            self.tokenizer = _Tokenizer(str(tok_path), None)
        else:
            vocab = list(cfg.decoding.get("vocabulary") or [])
            self.tokenizer = _Tokenizer(None, vocab)
        self._blank_id = len(self.tokenizer)

        opts = rt.SessionOptions()
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        # По умолчанию — все физические ядра; переопределяется OMP_NUM_THREADS
        threads = int(os.getenv("OMP_NUM_THREADS", "0"))
        if threads > 0:
            opts.intra_op_num_threads = threads
        opts.log_severity_level = 3

        self.model = rt.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=opts
        )
        if self._is_rnnt:
            self._pred_sess = rt.InferenceSession(
                str(parts["decoder"]), providers=["CPUExecutionProvider"], sess_options=opts
            )
            self._joint_sess = rt.InferenceSession(
                str(parts["joint"]), providers=["CPUExecutionProvider"], sess_options=opts
            )
            self._pred_hidden = int(cfg.head.decoder.pred_hidden)
            self._pred_layers = int(cfg.head.decoder.pred_rnn_layers)
        logger.info(f"ONNX model '{name}{ONNX_VARIANT}' loaded ({model_path.stat().st_size >> 20} MB)")

    def _cleanup_model(self) -> None:
        self.model = None
        self.tokenizer = None
        self._pred_sess = None
        self._joint_sess = None

    # ------------------------------------------------------------- inference

    def _infer_chunk(
        self, audio: np.ndarray
    ) -> Tuple[str, List[int], List[int], float]:
        """
        Прогоняет кусок аудио (<= GIGAAM_MAX_SHORT_AUDIO_SEC) через ONNX.

        Returns:
            (text, token_ids, token_frames, frame_shift)
        """
        feats = self.featurizer(audio)
        if self._is_rnnt:
            token_ids, token_frames, enc_frames = self._rnnt_decode(feats)
        else:
            token_ids, token_frames, enc_frames = self._ctc_decode(feats)

        frame_shift = (len(audio) / SAMPLE_RATE) / max(enc_frames, 1)
        return self.tokenizer.decode(token_ids), token_ids, token_frames, frame_shift

    def _ctc_decode(self, feats: np.ndarray) -> Tuple[List[int], List[int], int]:
        """CTC: encoder выдаёт log_probs; схлопывание с сохранением кадров."""
        log_probs, enc_lens = self.model.run(
            None,
            {
                "features": feats[None],
                "feature_lengths": np.array([feats.shape[1]], dtype=np.int64),
            },
        )
        labels = log_probs.argmax(axis=-1)[0][: int(enc_lens[0])]

        token_ids: List[int] = []
        token_frames: List[int] = []
        prev = None
        for frame, label in enumerate(labels.tolist()):
            if label != self._blank_id and label != prev:
                token_ids.append(label)
                token_frames.append(frame)
            prev = label
        return token_ids, token_frames, int(enc_lens[0])

    def _rnnt_decode(self, feats: np.ndarray) -> Tuple[List[int], List[int], int]:
        """
        Greedy RNNT-декод (порт vendored `onnx_utils._decode_rnnt_batch`
        для batch=1, без torch; дополнительно запоминает кадр каждого токена).
        """
        encoded, enc_lens = self.model.run(
            None,
            {
                "audio_signal": feats[None],
                "length": np.array([feats.shape[1]], dtype=np.int64),
            },
        )
        t_max = int(enc_lens[0])
        dtype = encoded.dtype

        token_ids: List[int] = []
        token_frames: List[int] = []
        label = np.full((1, 1), self._blank_id, dtype=np.int64)
        h = np.zeros((self._pred_layers, 1, self._pred_hidden), dtype=dtype)
        c = np.zeros_like(h)

        for t in range(t_max):
            f = encoded[:, :, t : t + 1]
            for _ in range(_MAX_LETTERS_PER_FRAME):
                dec, ho, co = self._pred_sess.run(None, {"x": label, "hi": h, "ci": c})
                (joint,) = self._joint_sess.run(
                    None, {"enc": f, "dec": dec.swapaxes(1, 2)}
                )
                k = int(joint[0, 0, 0, :].argmax())
                if k == self._blank_id:
                    break
                token_ids.append(k)
                token_frames.append(t)
                label = np.array([[k]], dtype=np.int64)
                h, c = ho, co
        return token_ids, token_frames, t_max

    def _chunk_bounds(self, num_samples: int) -> List[Tuple[int, int]]:
        """Жёсткие границы чанков (хвост < min сливается с предыдущим)."""
        chunk = int(GIGAAM_CHUNK_SEC * SAMPLE_RATE)
        min_chunk = int(GIGAAM_MIN_CHUNK_SEC * SAMPLE_RATE)
        bounds: List[Tuple[int, int]] = []
        pos = 0
        while pos < num_samples:
            end = min(pos + chunk, num_samples)
            if num_samples - pos < min_chunk and bounds:
                bounds[-1] = (bounds[-1][0], num_samples)
                break
            bounds.append((pos, end))
            pos = end
        return bounds

    def _chunk_bounds_vad(self, audio: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Границы чанков по паузам речи (silero-vad).

        Чанк набирает речевые участки, пока укладывается в GIGAAM_CHUNK_SEC;
        резка проходит по тишине между участками, чисто тихие промежутки
        между чанками не транскрибируются вовсе. Границы остаются в исходной
        временной шкале — таймстемпы слов/сегментов не смещаются.

        Returns:
            Список границ; [] — речи нет; None — VAD недоступен (фолбэк).
        """
        from src.asr.vad import silero_vad

        if not silero_vad.available():
            logger.warning("VAD model not found; falling back to fixed chunking")
            return None

        regions = silero_vad.speech_regions(audio)
        if not regions:
            return []

        chunk_max = int(GIGAAM_CHUNK_SEC * SAMPLE_RATE)
        min_chunk = int(GIGAAM_MIN_CHUNK_SEC * SAMPLE_RATE)

        # Группируем речевые участки в чанки до chunk_max по общему охвату
        grouped: List[Tuple[int, int]] = []
        cur_s, cur_e = regions[0]
        for s, e in regions[1:]:
            if e - cur_s <= chunk_max:
                cur_e = e
            else:
                grouped.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        grouped.append((cur_s, cur_e))

        # Непрерывная речь длиннее chunk_max — дорезаем жёстко внутри
        bounds: List[Tuple[int, int]] = []
        for s, e in grouped:
            if e - s <= chunk_max:
                bounds.append((s, e))
                continue
            pos = s
            while pos < e:
                end = min(pos + chunk_max, e)
                if e - pos < min_chunk and bounds:
                    bounds[-1] = (bounds[-1][0], e)
                    break
                bounds.append((pos, end))
                pos = end
        return bounds

    def transcribe(
        self,
        audio: np.ndarray,
        task: str,
        language: Optional[str],
        word_timestamps: bool,
        output: str,
        options: Optional[dict] = None,
    ) -> Union[TranscriptionResponse, str]:
        self.update_activity()
        self.ensure_model_loaded()

        if task == "translate":
            logger.warning("GigaAM does not support translation; transcribing instead")

        audio = np.asarray(audio, dtype=np.float32)
        duration = len(audio) / SAMPLE_RATE

        # per-request переопределение VAD (options.vad), иначе — env-дефолт
        use_vad = VAD_CHUNKING
        if options and options.get("vad") is not None:
            use_vad = bool(options["vad"])

        with self.model_lock:
            if duration > GIGAAM_MAX_SHORT_AUDIO_SEC:
                bounds = self._chunk_bounds_vad(audio) if use_vad else None
                if bounds is None:
                    bounds = self._chunk_bounds(len(audio))
            else:
                bounds = [(0, len(audio))]

            texts: List[str] = []
            segments: List[Segment] = []
            for idx, (start, end) in enumerate(bounds):
                offset = start / SAMPLE_RATE
                text, token_ids, token_frames, frame_shift = self._infer_chunk(
                    audio[start:end]
                )
                if not text.strip():
                    continue
                texts.append(text.strip())
                seg = Segment(
                    id=idx,
                    start=round(offset, 3),
                    end=round(end / SAMPLE_RATE, 3),
                    text=text.strip(),
                )
                if word_timestamps:
                    seg.words = _group_words(
                        self.tokenizer, token_ids, token_frames, frame_shift, offset
                    )
                segments.append(seg)

        full_text = " ".join(texts).strip()
        if output == "text":
            return full_text

        resp = TranscriptionResponse(text=full_text, language=language or "ru")
        # Сегменты прикладываем всегда (даже один): из них собираются
        # srt/vtt/tsv и verbose_json — без сегментов субтитры пустые.
        if segments:
            resp.segments = segments
        if duration > 0:
            resp.chars_per_second = round(len(full_text) / duration, 4)
        return resp
