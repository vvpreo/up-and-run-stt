"""
Стабилизация слов в черновиках живой диктовки (LocalAgreement-2).

Каждый тик открытая фраза распознаётся заново целиком, и соседние гипотезы
отличаются в хвосте. Стабилизатор решает, какие слова уже можно считать
окончательными («зафиксированными»):

  1. слово стоит на той же позиции в двух гипотезах подряд (сравнение без
     регистра и пунктуации), и
  2. оно кончилось не позже, чем за `guard_sec` до правого края аудио — то
     есть модель уже слышала, что было после него.

Зафиксированная часть только дописывается. Исключение — ПЕРЕЗАПИСЬ: модель с
нормализацией текста может задним числом изменить уже зафиксированное
(«двадцать пять» -> «25», имя собственное после уточняющего контекста). Если
две гипотезы подряд согласны между собой, но расходятся с зафиксированным,
фиксация заменяется целиком и помечается `rewrite` — клиент перерисовывает
фразу (и может подсветить изменившееся). Одиночное расхождение игнорируется:
это шум одного тика.

Модуль чистый (без numpy и без сети) — тестируется отдельно.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

_PUNCT = re.compile(r"[^\w]+", re.UNICODE)


def norm_word(word: str) -> str:
    """Форма для сравнения: без регистра, пунктуации и различия е/ё."""
    return _PUNCT.sub("", word.lower().replace("ё", "е"))


def norm_words(words: Sequence[str]) -> List[str]:
    return [norm_word(w) for w in words]


@dataclass
class Update:
    committed: List[str]                 # все зафиксированные слова фразы
    tentative: List[str]                 # нестабильный хвост текущей гипотезы
    new_words: List[str] = field(default_factory=list)   # дописано в этом тике
    rewrite: bool = False                # зафиксированное было заменено
    changed_from: Optional[int] = None   # с какого слова (индекс) началась замена

    @property
    def committed_text(self) -> str:
        return " ".join(self.committed)

    @property
    def tentative_text(self) -> str:
        return " ".join(self.tentative)

    @property
    def text(self) -> str:
        return " ".join(self.committed + self.tentative)


class Stabilizer:
    """Состояние одной открытой фразы. На новую фразу — новый экземпляр."""

    def __init__(self, guard_sec: float = 0.5) -> None:
        self.guard_sec = guard_sec
        self.committed: List[str] = []
        self._prev: List[str] = []

    def update(self, words: Sequence[Tuple[str, float]], audio_end: float) -> Update:
        """
        Args:
            words: слова текущей гипотезы как (слово, конец_сек) от начала аудио фразы.
            audio_end: длительность аудио, по которому построена гипотеза.
        """
        cur = [w for w, _ in words]
        cur_n = norm_words(cur)
        prev_n = norm_words(self._prev)

        # Сколько слов подряд с начала совпало с прошлой гипотезой
        agree = 0
        for a, b in zip(cur_n, prev_n):
            if a != b:
                break
            agree += 1

        # Сколько слов с начала уже вне опасной зоны у правого края
        eligible = 0
        for _, end in words:
            if end > audio_end - self.guard_sec:
                break
            eligible += 1

        stable = min(agree, eligible)
        cand, cand_n = cur[:stable], cur_n[:stable]
        old, old_n = self.committed, norm_words(self.committed)

        new_words: List[str] = []
        rewrite = False
        changed_from: Optional[int] = None

        # Первое расхождение кандидата с уже зафиксированным
        diverge = next(
            (i for i, (a, b) in enumerate(zip(cand_n, old_n)) if a != b), None
        )
        if diverge is not None:
            # Две гипотезы подряд согласны на другом слове -> перезапись
            self.committed = list(cand)
            rewrite, changed_from = True, diverge
        elif len(cand) > len(old):
            new_words = list(cand[len(old):])
            self.committed = old + new_words
        # иначе: кандидат короче зафиксированного и не противоречит ему —
        # оснований что-либо менять нет

        self._prev = cur
        tentative = cur[len(self.committed):] if len(cur) > len(self.committed) else []
        return Update(
            committed=list(self.committed),
            tentative=tentative,
            new_words=new_words,
            rewrite=rewrite,
            changed_from=changed_from,
        )


def revised_from(committed: Sequence[str], final_text: str) -> Optional[int]:
    """
    Индекс первого слова, которым финальный текст фразы отличается от
    зафиксированного в черновиках, либо None, если финал лишь продолжил его.
    """
    final_n = norm_words(final_text.split())
    for i, c in enumerate(norm_words(committed)):
        if i >= len(final_n) or final_n[i] != c:
            return i
    return None
