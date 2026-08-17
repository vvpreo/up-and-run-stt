"""
Сегментация живого аудиопотока на фразы.

Здесь живёт вся логика «стриминга запроса»: клиент шлёт сырой PCM по мере
того, как говорит, а сервер решает, где кончается фраза, и отдаёт готовые
куски на инференс. Клиент при этом ничего не решает — ни где резать, ни что
считать тишиной.

Почему резать вообще приходится: GigaAM офлайновая, она принимает
законченный отрезок и возвращает текст, достраивать гипотезу по мере
поступления сэмплов не умеет. Поэтому «стриминг» на входе означает не
непрерывное декодирование, а то, что нарезка переехала на сервер и делается
нейросетевым VAD вместо эвристики в браузере.

Класс не знает ни про WebSocket, ни про модель: на вход байты, на выход
готовые фразы. Так его можно тестировать без сети и без инференса.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.asr.vad import FRAME_SAMPLES, VAD_THRESHOLD, silero_vad
from src.config import (
    SAMPLE_RATE,
    STREAM_MAX_PHRASE_SEC,
    STREAM_MIN_PHRASE_SEC,
    STREAM_SILENCE_MS,
)

logger = logging.getLogger(__name__)

_FRAME_MS = FRAME_SAMPLES / SAMPLE_RATE * 1000  # 32 мс при 16 кГц

# Сколько тишины оставлять перед первым словом и после последнего.
#
# Подобрано замером на сэмпле 137 с (32 фразы), качество мерилось долей
# словаря, совпавшего с транскрипцией целого файла:
#   без подрезки (137 с аудио в модель)  8.9 с CPU, 58/70 слов
#   pad 500 мс   ( 69 с аудио в модель)  5.0 с CPU, 63/70 слов  <- выбрано
#   pad 250 мс   ( 56 с аудио в модель)  4.3 с CPU, 61/70 слов
#
# Подрезка не только экономит инференс, но и УЛУЧШАЕТ качество: длинные
# участки тишины модель склонна достраивать мусором. Слишком агрессивная
# подрезка (250 мс) начинает обрубать концы слов — «один» превращалось
# в «идеадин», — поэтому 500 мс.
_PREROLL_SAMPLES = int(0.5 * SAMPLE_RATE)
_TRAIL_PAD_MS = 500.0


@dataclass
class Phrase:
    """Готовый к распознаванию отрезок речи."""

    audio: np.ndarray          # float32 в [-1, 1], 16 кГц моно
    start_sec: float           # смещение от начала сессии
    duration_sec: float
    forced: bool               # закрыта по лимиту длины, а не по паузе


class PhraseSegmenter:
    """
    Накопитель аудио, режущий поток на фразы по паузам речи.

    Использование:
        seg = PhraseSegmenter()
        for phrase in seg.feed(pcm_bytes):
            ...                      # фраза готова к инференсу
        tail = seg.flush()           # хвост при закрытии соединения
    """

    def __init__(
        self,
        threshold: float = VAD_THRESHOLD,
        silence_ms: int = STREAM_SILENCE_MS,
        max_phrase_sec: float = STREAM_MAX_PHRASE_SEC,
        min_phrase_sec: float = STREAM_MIN_PHRASE_SEC,
    ) -> None:
        self._vad = silero_vad.stream()
        self._threshold = threshold
        self._silence_ms = silence_ms
        self._max_samples = int(max_phrase_sec * SAMPLE_RATE)
        self._min_samples = int(min_phrase_sec * SAMPLE_RATE)

        # Незаполненный хвост кадра: клиент не обязан слать ровно по 512
        # сэмплов, а VadStream.feed требует именно столько.
        self._partial = np.zeros(0, dtype=np.float32)

        self._buffer: List[np.ndarray] = []   # копится текущая фраза
        self._buffered = 0                    # сэмплов в _buffer
        self._silence_ms_acc = 0.0            # длительность текущей паузы
        self._saw_speech = False              # была ли речь в буфере
        self._consumed = 0                    # сэмплов от начала сессии

        self.is_speaking = False              # для событий speech.started/stopped

    # ------------------------------------------------------------------ вход

    def feed(self, pcm: np.ndarray) -> List[Phrase]:
        """
        Скормить очередной кусок аудио. Возвращает фразы, закрывшиеся на нём
        (обычно ноль или одна; больше — если клиент прислал сразу много).
        """
        out: List[Phrase] = []
        data = np.concatenate([self._partial, pcm]) if self._partial.size else pcm

        n_frames = len(data) // FRAME_SAMPLES
        self._partial = data[n_frames * FRAME_SAMPLES :].copy()

        for i in range(n_frames):
            frame = data[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            phrase = self._feed_frame(frame)
            if phrase is not None:
                out.append(phrase)
        return out

    def _feed_frame(self, frame: np.ndarray) -> Optional[Phrase]:
        prob = self._vad.feed(frame)
        is_speech = prob >= self._threshold

        self._buffer.append(frame)
        self._buffered += FRAME_SAMPLES
        self._consumed += FRAME_SAMPLES

        if is_speech:
            self._silence_ms_acc = 0.0
            self._saw_speech = True
            self.is_speaking = True
        else:
            self._silence_ms_acc += _FRAME_MS
            if self._silence_ms_acc >= self._silence_ms:
                self.is_speaking = False

            # Речь ещё не начиналась — держим только короткий pre-roll, а не
            # всю накопленную тишину. Иначе минута молчания перед первым
            # словом уехала бы в модель вместе с фразой.
            if not self._saw_speech and self._buffered > _PREROLL_SAMPLES:
                drop = 0
                while self._buffer and self._buffered - drop > _PREROLL_SAMPLES:
                    drop += len(self._buffer.pop(0))
                self._buffered -= drop

        # Фраза закрыта паузой достаточной длины после реальной речи
        if self._saw_speech and self._silence_ms_acc >= self._silence_ms:
            return self._cut(forced=False)

        # Предохранитель: непрерывная речь не должна копиться бесконечно
        if self._buffered >= self._max_samples:
            return self._cut(forced=True)

        return None

    # ----------------------------------------------------------------- выход

    def flush(self) -> Optional[Phrase]:
        """Хвост, оставшийся при закрытии соединения (если в нём есть речь)."""
        if self._partial.size:
            # Добить неполный кадр нулями, чтобы не потерять последние ~30 мс
            pad = FRAME_SAMPLES - self._partial.size
            self._buffer.append(
                np.concatenate([self._partial, np.zeros(pad, dtype=np.float32)])
            )
            self._buffered += FRAME_SAMPLES
            self._partial = np.zeros(0, dtype=np.float32)
        if not self._saw_speech:
            return None
        return self._cut(forced=True)

    def _cut(self, forced: bool) -> Optional[Phrase]:
        audio = (
            np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.float32)
        )
        start = (self._consumed - len(audio)) / SAMPLE_RATE

        # Срезать хвостовую тишину, оставив короткий pad: паузу в 600 мс
        # незачем гонять через модель, а совсем впритык резать нельзя —
        # обрубается последний звук слова.
        if not forced and self._silence_ms_acc > _TRAIL_PAD_MS:
            trim = int((self._silence_ms_acc - _TRAIL_PAD_MS) / 1000 * SAMPLE_RATE)
            if 0 < trim < len(audio):
                audio = audio[: len(audio) - trim]

        n = len(audio)
        self._reset_buffer()

        if n < self._min_samples:
            return None  # щелчок, кашель, стук по столу
        return Phrase(
            audio=audio,
            start_sec=max(0.0, start),
            duration_sec=n / SAMPLE_RATE,
            forced=forced,
        )

    def _reset_buffer(self) -> None:
        self._buffer = []
        self._buffered = 0
        self._silence_ms_acc = 0.0
        self._saw_speech = False


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """
    Кадры от клиента -> float32 в [-1, 1].

    Формат входа зафиксирован как PCM signed 16-bit little-endian, 16 кГц,
    моно — сознательно. Сжатый поток (opus/webm) потребовал бы декодера на
    КАЖДОЕ соединение, то есть отдельного процесса ffmpeg с его памятью;
    сырой PCM стоит 256 кбит/с и ноль CPU.
    """
    if len(raw) % 2:
        raw = raw[:-1]  # половина сэмпла: обрезаем, придёт со следующим кадром
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
