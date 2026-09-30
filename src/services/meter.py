"""
Учёт времени по видам работы — для панели «Загрузка» (/stats).

Каждый замер — (ключ, длительность, секунды аудио). По ключу считается доля
занятости за последнее окно (сколько секунд работы пришлось на секунду
времени), среднее время вызова и счётчики. Ключи:

    asr:<модель>    — инференс модели распознавания (фичи + энкодер + декод)
    emo             — модель эмоций
    vad             — silero-VAD (покадрово в живых сессиях, пакетно в файлах)
    live:final      — финалы фраз живой диктовки (с ожиданием очереди к модели)
    live:partial    — черновики незакрытых фраз
    live:emotion    — эмоции фраз

`asr:*`, `emo`, `vad` — чистое время модели; `live:*` — то же время с точки
зрения живой сессии, то есть на что именно оно ушло.

Стоимость. Запись — доли микросекунды на вызов. Пооконная история ведётся
только пока панель открыта (см. Meter.observe): свёрнутая панель не
запрашивает /stats, и тогда остаются лишь накопительные счётчики.
"""

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

_KEEP_SEC = 30.0       # сколько истории держать, пока панель смотрят
_OBSERVE_SEC = 10.0    # сколько после последнего запроса /stats продолжать запись


class Meter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict = defaultdict(deque)   # key -> deque[(t_end, seconds, audio_sec)]
        self._totals: dict = defaultdict(lambda: [0, 0.0, 0.0])  # calls, seconds, audio
        # Пооконная история пишется только пока панель «Загрузка» кто-то
        # смотрит (есть свежие запросы /stats). Без наблюдателя остаются одни
        # счётчики — три сложения на вызов, никакой истории и памяти под неё.
        self._observed_until = 0.0

    def observe(self) -> None:
        """Вызывается из /stats: включает запись истории на ближайшие секунды."""
        self._observed_until = time.monotonic() + _OBSERVE_SEC

    def record(self, key: str, seconds: float, audio_sec: float = 0.0) -> None:
        now = time.monotonic()
        with self._lock:
            t = self._totals[key]
            t[0] += 1
            t[1] += seconds
            t[2] += audio_sec
            if now > self._observed_until:
                if self._events:
                    self._events.clear()  # наблюдатель ушёл — историю не держим
                return
            q = self._events[key]
            q.append((now, seconds, audio_sec))
            while q and q[0][0] < now - _KEEP_SEC:
                q.popleft()

    @contextmanager
    def timed(self, key: str, audio_sec: float = 0.0):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(key, time.perf_counter() - t0, audio_sec)

    def snapshot(self, window: float = 5.0) -> dict:
        now = time.monotonic()
        out = {}
        with self._lock:
            for key in list(self._totals):
                q = self._events.get(key, ())
                recent = [(s, a) for (t, s, a) in q if t >= now - window]
                busy = sum(s for s, _ in recent)
                audio = sum(a for _, a in recent)
                calls, seconds, audio_total = self._totals[key]
                out[key] = {
                    # 100 % = вид работы занимал модель всё окно целиком
                    "busy_pct": round(100.0 * busy / window, 1),
                    "calls": len(recent),
                    "avg_ms": round(1000.0 * busy / len(recent), 1) if recent else None,
                    "audio_sec": round(audio, 2),
                    "calls_total": calls,
                    "seconds_total": round(seconds, 2),
                    "audio_sec_total": round(audio_total, 1),
                }
        return out


meter = Meter()
