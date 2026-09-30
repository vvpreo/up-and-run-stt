"""
Живая сессия распознавания — логика потокового эндпоинта `WS /v1/realtime`
(src/routes/realtime.py), отделённая от транспорта.

Класс не знает ни про WebSocket, ни про формат сообщений: на вход — float32
PCM 16 кГц, на выход — внутренние события (dict с ключом "kind"), которые
роут переводит в протокол OpenAI Realtime:

    speech_started / speech_stopped   {at_sec, phrase_id}
    phrase_closed {phrase_id, start, duration}   — фраза отрезана и ушла на финал
    empty      {phrase_id}                       — финал фразы оказался пустым
    partial    {text, committed, tentative, new_words, rewrite, changed_from,
                phrase, start, duration, inference_sec}
    final      {text, seq, start, duration, inference_sec, forced,
                revised_from, committed_before}
    emotion    {seq, dominant, emotions, inference_sec}
    overflow   {queued}
    error      {message}

Внутри: сегментация по паузам (PhraseSegmenter), очередь финалов, таймер
черновиков со стабилизацией слов (Stabilizer), мягкий срез длинной фразы по
границе зафиксированного слова, эмоции по фразам.

Память за сессию не растёт: аудио хранится только для открытой фразы, очередь
финалов ограничена, история текста не копится — сессия может длиться часами.
"""

import asyncio
import logging
import re
import time
from typing import Awaitable, Callable, List, Optional

import numpy as np

from src.asr.registry import list_models
from src.config import (
    SAMPLE_RATE,
    STREAM_MAX_QUEUED_PHRASES,
    STREAM_MAX_SESSIONS,
    STREAM_PARTIALS,
    STREAM_PARTIAL_GUARD_MS,
    STREAM_PARTIAL_INTERVAL_MS,
    STREAM_PARTIAL_MIN_SEC,
    STREAM_SOFT_CUT_SEC,
)
from src.services.meter import meter
from src.services.stabilizer import Stabilizer, revised_from
from src.services.stream_session import Phrase, PhraseSegmenter

logger = logging.getLogger(__name__)

Emit = Callable[[dict], Awaitable[None]]

# Число живых сессий. Открытая сессия стоит ~0.5% ядра (VAD) и ~2 МБ, так что
# лимит здесь на порядок выше, чем MAX_PENDING_REQUESTS для обычных запросов.
_sessions = 0
_sessions_lock = asyncio.Lock()


async def acquire_session() -> bool:
    global _sessions
    async with _sessions_lock:
        if STREAM_MAX_SESSIONS > 0 and _sessions >= STREAM_MAX_SESSIONS:
            return False
        _sessions += 1
        return True


async def release_session() -> None:
    global _sessions
    async with _sessions_lock:
        _sessions = max(0, _sessions - 1)


def active_sessions() -> int:
    """Для /health и /stats."""
    return _sessions


def partial_model(selected):
    """
    Модель для черновиков. Всегда CTC, если она есть на инстансе: черновик
    считается каждые полсекунды, а RNNT-декодер (покадровый цикл на CPU) в
    разы дороже при том же энкодере. Финал фразы считает выбранная модель.
    """
    models = list_models()
    for name in ("v3_e2e_ctc", "v3_ctc"):
        if name in models:
            return models[name]
    return selected


def partials_enabled(requested: Optional[bool], model) -> bool:
    """Явный параметр клиента > STREAM_PARTIALS > auto (только на CUDA)."""
    if requested is not None:
        return requested
    if STREAM_PARTIALS in ("true", "1", "yes"):
        return True
    if STREAM_PARTIALS in ("false", "0", "no"):
        return False
    try:
        return model.get_info().get("device") == "cuda"
    except Exception:
        return False


class LiveSession:
    def __init__(
        self,
        *,
        model,
        draft_model,
        language: str,
        partials: bool,
        emotions: bool,
        emit: Emit,
        segmenter: Optional[PhraseSegmenter] = None,
    ) -> None:
        self.model = model
        self.draft_model = draft_model
        self.language = language
        self.partials_on = partials
        self.emotions_on = emotions
        self._emit = emit

        self.segmenter = segmenter or PhraseSegmenter()
        self._queue: asyncio.Queue = asyncio.Queue()
        self.seq = 0
        self._emotion_tasks: set = set()
        # Фраза, начавшаяся мягким срезом посреди предложения: её первое слово
        # модель напишет с заглавной (для неё это начало аудио) — возвращаем
        # строчную, как было в гипотезе с контекстом.
        self._continuation_phrase = -1
        # Финал фразы в работе: черновики в это время не считаются, чтобы не
        # занимать модель и не обгонять финал предыдущей фразы.
        self._final_busy = False
        self._speaking = False
        self._speech_phrase = 0
        self._stab = Stabilizer(guard_sec=STREAM_PARTIAL_GUARD_MS / 1000)
        self._stab_phrase = 0
        self._worker: Optional[asyncio.Task] = None
        self._partial_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------ жизненный цикл

    def start(self) -> None:
        self._worker = asyncio.create_task(self._final_worker())
        if self.partials_on:
            self._partial_task = asyncio.create_task(self._partial_worker())

    async def feed(self, pcm: np.ndarray) -> None:
        """Очередной кусок аудио (float32, 16 кГц моно)."""
        for phrase in self.segmenter.feed(pcm):
            await self._enqueue(phrase)
        if self.segmenter.is_speaking != self._speaking:
            self._speaking = self.segmenter.is_speaking
            if self._speaking:
                self._speech_phrase = self.segmenter.phrase_id
            await self._emit(
                {
                    "kind": "speech_started" if self._speaking else "speech_stopped",
                    "at_sec": round(self.segmenter.consumed / SAMPLE_RATE, 3),
                    # фраза, к которой относится событие: «stopped» приходит уже
                    # после закрытия фразы, когда счётчик сегментатора ушёл вперёд
                    "phrase_id": self._speech_phrase,
                }
            )

    async def commit(self) -> None:
        """Принудительно закрыть текущую фразу, не дожидаясь паузы."""
        if (tail := self.segmenter.flush()) is not None:
            await self._enqueue(tail)

    async def finish(self) -> None:
        """Добить хвост, дождаться финалов и эмоций. После этого событий не будет."""
        if self._partial_task is not None:
            self._partial_task.cancel()
        try:
            if (tail := self.segmenter.flush()) is not None:
                await self._enqueue(tail)
        except Exception:
            pass
        self._queue.put_nowait(None)
        if self._worker is not None:
            try:
                await asyncio.wait_for(self._worker, timeout=60)
            except (asyncio.TimeoutError, Exception):
                self._worker.cancel()
        if self._emotion_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._emotion_tasks, return_exceptions=True), timeout=30
                )
            except (asyncio.TimeoutError, Exception):
                pass

    # ------------------------------------------------------------------ внутреннее

    async def _enqueue(self, phrase: Phrase) -> None:
        # Что было зафиксировано в черновиках этой фразы — чтобы финал мог
        # сообщить, переписал ли он уже показанные слова.
        phrase.committed_before = (
            list(self._stab.committed) if self._stab_phrase == phrase.phrase_id else []
        )
        self._stab = Stabilizer(guard_sec=STREAM_PARTIAL_GUARD_MS / 1000)
        self._stab_phrase = self.segmenter.phrase_id
        await self._emit(
            {
                "kind": "phrase_closed",
                "phrase_id": phrase.phrase_id,
                "start": round(phrase.start_sec, 3),
                "duration": round(phrase.duration_sec, 3),
            }
        )

        # Бэкпрешер: клиент может лить быстрее реального времени (например,
        # проигрывая файл в сокет). Копить фразы без предела нельзя — растёт
        # память, поэтому самые старые выбрасываются, о чём клиент узнаёт явно.
        if self._queue.qsize() >= STREAM_MAX_QUEUED_PHRASES:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                pass
            await self._emit({"kind": "overflow", "queued": self._queue.qsize()})
        self._queue.put_nowait(phrase)

    async def _final_worker(self) -> None:
        while True:
            phrase = await self._queue.get()
            if phrase is None:
                self._queue.task_done()
                return
            self._final_busy = True
            t0 = time.perf_counter()
            try:
                result = await asyncio.to_thread(
                    self.model.transcribe, phrase.audio, "transcribe", self.language, False, "text"
                )
                text = _tidy((result if isinstance(result, str) else result.text).strip())
                if phrase.phrase_id == self._continuation_phrase:
                    text = _decapitalize(text)
                meter.record("live:final", time.perf_counter() - t0, phrase.duration_sec)
            except Exception as e:
                logger.exception("[live] phrase transcription failed")
                await self._emit({"kind": "error", "message": str(e)})
                self._queue.task_done()
                self._final_busy = False
                continue

            if text:
                await self._emit_final(
                    phrase, text, getattr(phrase, "committed_before", []), time.perf_counter() - t0
                )
            else:
                await self._emit({"kind": "empty", "phrase_id": phrase.phrase_id})
            self._queue.task_done()
            self._final_busy = False

    async def _emit_final(self, phrase: Phrase, text: str, committed: List[str], took: float) -> None:
        self.seq += 1
        await self._emit(
            {
                "kind": "final",
                "phrase_id": phrase.phrase_id,
                "text": text,
                "seq": self.seq,
                "start": round(phrase.start_sec, 2),
                "duration": round(phrase.duration_sec, 2),
                "inference_sec": round(took, 3),
                "forced": phrase.forced,
                "committed_before": committed,
                "revised_from": revised_from(committed, text),
            }
        )
        if self.emotions_on:
            task = asyncio.create_task(self._emotion(phrase, self.seq))
            self._emotion_tasks.add(task)
            task.add_done_callback(self._emotion_tasks.discard)

    async def _soft_cut(self, words, n_keep: int, snapshot_start: float, audio: np.ndarray) -> bool:
        """
        Закрыть начало слишком длинной фразы по границе слова. `words` — слова
        последней гипотезы (WordTimestamp, время от начала аудио фразы),
        первые n_keep из них зафиксированы. Текст закрываемой части берётся
        прямо из этой гипотезы: она видела контекст с обеих сторон, поэтому
        пунктуация и регистр в ней правильные, а второй проход не нужен.
        """
        seg = self.segmenter
        last = words[n_keep - 1]
        nxt = words[n_keep] if n_keep < len(words) else None
        # Резать ближе к концу зафиксированного слова, а не посередине
        # промежутка: CTC отмечает начало слова с запозданием, и «промежуток»
        # по таймстемпам захватывает начало следующего слова. Хвост последнего
        # слова при этом можно и задеть — текст закрываемой части берётся из
        # гипотезы, а не распознаётся заново по обрезанному аудио.
        gap = (nxt.start - last.end) if nxt is not None else 0.2
        cut_sec = _quietest_point(audio, last.end, last.end + max(0.06, gap / 2))
        phrase = seg.split_at(int(cut_sec * SAMPLE_RATE))
        if phrase is None:
            return False
        text = _tidy(" ".join(w.word for w in words[:n_keep]))
        if phrase.phrase_id == self._continuation_phrase:
            text = _decapitalize(text)
        committed = list(self._stab.committed)
        # Следующая фраза — продолжение предложения, если слово после среза
        # в гипотезе с контекстом начиналось со строчной
        self._continuation_phrase = (
            seg.phrase_id if nxt is not None and nxt.word[:1].islower() else -1
        )
        self._stab = Stabilizer(guard_sec=STREAM_PARTIAL_GUARD_MS / 1000)
        self._stab_phrase = seg.phrase_id
        await self._emit(
            {
                "kind": "phrase_closed",
                "phrase_id": phrase.phrase_id,
                "start": round(phrase.start_sec, 3),
                "duration": round(phrase.duration_sec, 3),
            }
        )
        await self._emit_final(phrase, text, committed, 0.0)
        # Речь не прерывалась, но относится уже к новой фразе
        self._speech_phrase = seg.phrase_id
        if self._speaking:
            await self._emit(
                {
                    "kind": "speech_started",
                    "at_sec": round(snapshot_start + cut_sec, 3),
                    "phrase_id": seg.phrase_id,
                }
            )
        # Хвост гипотезы после среза — уже черновик новой фразы: показать его
        # сразу, иначе нестабильные слова исчезли бы до следующего тика.
        rest = [w.word for w in words[n_keep:]]
        if rest:
            if seg.phrase_id == self._continuation_phrase:
                rest[0] = _decapitalize(rest[0])
            tail = " ".join(rest)
            await self._emit(
                {
                    "kind": "partial",
                    "phrase_id": seg.phrase_id,
                    "text": tail,
                    "committed": "",
                    "tentative": tail,
                    "new_words": [],
                    "rewrite": False,
                    "changed_from": None,
                    "phrase": self.seq + 1,
                    "start": round(snapshot_start + cut_sec, 2),
                    "duration": round(seg.consumed / SAMPLE_RATE - (snapshot_start + cut_sec), 2),
                    "inference_sec": 0.0,
                }
            )
        return True

    async def _emotion(self, phrase: Phrase, seq: int) -> None:
        from src.asr.onnx_emo import emo_model

        t0 = time.perf_counter()
        try:
            probs = await asyncio.to_thread(emo_model.classify, phrase.audio)
            meter.record("live:emotion", time.perf_counter() - t0, phrase.duration_sec)
            await self._emit(
                {
                    "kind": "emotion",
                    "phrase_id": phrase.phrase_id,
                    "seq": seq,
                    "dominant": max(probs, key=probs.get),
                    "emotions": {k: round(float(v), 4) for k, v in probs.items()},
                    "inference_sec": round(time.perf_counter() - t0, 3),
                }
            )
        except ValueError:
            pass  # слишком короткий фрагмент — эмоцию не определить
        except Exception:
            logger.exception("[live] phrase emotion failed")

    async def _partial_worker(self) -> None:
        """
        Черновики текущей фразы. Раз в интервал берёт снимок незакрытой фразы,
        распознаёт его целиком (с таймстемпами слов) и прогоняет через
        стабилизатор. Финалы важнее: если фраза в очереди или в работе, тик
        пропускается. Черновик, досчитанный уже после закрытия своей фразы,
        выбрасывается (иначе он появился бы поверх финального текста).
        """
        interval = STREAM_PARTIAL_INTERVAL_MS / 1000
        min_samples = int(STREAM_PARTIAL_MIN_SEC * SAMPLE_RATE)
        last_consumed = -1
        seg = self.segmenter
        while True:
            await asyncio.sleep(interval)
            if self._final_busy or not self._queue.empty():
                continue
            snap = seg.snapshot()
            if snap is None:
                continue
            audio, start = snap
            if len(audio) < min_samples or seg.consumed == last_consumed:
                continue
            phrase_id = seg.phrase_id
            last_consumed = seg.consumed
            t0 = time.perf_counter()
            try:
                result = await asyncio.to_thread(
                    self.draft_model.transcribe, audio, "transcribe", self.language, True, "json"
                )
            except Exception:
                logger.exception("[live] partial transcription failed")
                continue
            duration = len(audio) / SAMPLE_RATE
            meter.record("live:partial", time.perf_counter() - t0, duration)
            if seg.phrase_id != phrase_id or self._final_busy:
                continue  # фраза уже закрылась — её покажет финал

            words = [
                w
                for s in (getattr(result, "segments", None) or [])
                for w in (s.words or [])
            ]
            if not words:
                continue
            if phrase_id == self._continuation_phrase:
                words[0].word = _decapitalize(words[0].word)
            if self._stab_phrase != phrase_id:
                self._stab = Stabilizer(guard_sec=STREAM_PARTIAL_GUARD_MS / 1000)
                self._stab_phrase = phrase_id
            upd = self._stab.update([(w.word, w.end) for w in words], duration)

            # Фраза переросла порог — закрыть её зафиксированную часть по границе
            # слова. Режем, только если срез уносит заметную долю буфера:
            # иначе он ничего не даст.
            n_keep = len(upd.committed)
            if (
                STREAM_SOFT_CUT_SEC > 0
                and duration >= STREAM_SOFT_CUT_SEC
                and 0 < n_keep <= len(words)
                and words[n_keep - 1].end >= duration * 0.5
            ):
                # Резать лучше там, где после слова есть микропауза: тогда срез
                # не заденет начало следующего слова. Ждём такого момента до
                # трёх секунд сверх порога, дальше режем в любом случае.
                gap = (words[n_keep].start if n_keep < len(words) else duration) - words[n_keep - 1].end
                if gap < _SOFT_CUT_MIN_GAP and duration < STREAM_SOFT_CUT_SEC + 3:
                    pass
                else:
                  try:
                    if await self._soft_cut(words, n_keep, start, audio):
                        continue
                  except Exception:
                    logger.exception("[live] soft cut failed")
            try:
                await self._emit(
                    {
                        "kind": "partial",
                        "phrase_id": phrase_id,
                        "text": upd.text,
                        "committed": upd.committed_text,
                        "tentative": upd.tentative_text,
                        "new_words": upd.new_words,
                        "rewrite": upd.rewrite,
                        "changed_from": upd.changed_from,
                        "phrase": self.seq + 1,
                        "start": round(start, 2),
                        "duration": round(duration, 2),
                        "inference_sec": round(time.perf_counter() - t0, 3),
                    }
                )
            except Exception:
                return  # транспорт закрыт


_SOFT_CUT_MIN_GAP = 0.12  # сек тишины после слова, при которой срез предпочтителен
_DOUBLE_PUNCT = re.compile(r"([?!…])\.(?=\s|$)")


def _tidy(text: str) -> str:
    """Убрать артефакт модели — точку сразу после ?, ! или многоточия («индикатор?.»)."""
    return _DOUBLE_PUNCT.sub(r"\1", text)


def _quietest_point(audio: np.ndarray, left_sec: float, right_sec: float) -> float:
    """
    Самое тихое место между двумя словами — там и резать. Таймстемпы CTC
    приблизительны (кадр 40 мс, токен запаздывает), поэтому середина
    промежутка может попасть на начало следующего слова; минимум энергии в
    окрестности промежутка надёжнее.
    """
    frame = SAMPLE_RATE // 50  # 20 мс
    lo = max(0, int(left_sec * SAMPLE_RATE))
    hi = min(len(audio), int(right_sec * SAMPLE_RATE))
    if hi - lo < 2 * frame:
        return (left_sec + right_sec) / 2
    n = (hi - lo) // frame
    chunk = audio[lo : lo + n * frame].reshape(n, frame)
    energy = (chunk.astype(np.float64) ** 2).mean(axis=1)
    k = int(energy.argmin())
    return (lo + k * frame + frame // 2) / SAMPLE_RATE


def _decapitalize(text: str) -> str:
    """
    Первая буква — строчная. Для фразы, начавшейся мягким срезом посреди
    предложения. Слова целиком из заглавных (аббревиатуры) и «Я» не трогаются.
    """
    head = text.split(" ", 1)[0]
    letters = [c for c in head if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return text
    if len(letters) == 1 and letters[0] == "Я":
        return text
    for i, c in enumerate(text):
        if c.isalpha():
            return text[:i] + c.lower() + text[i + 1:]
    return text
