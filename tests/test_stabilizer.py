"""
Стабилизатор слов для черновиков живой диктовки — чистая логика, без сервиса.
Модуль грузится по пути: в CI тесты идут в образе без зависимостей сервиса.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "stabilizer", Path(__file__).resolve().parent.parent / "src" / "services" / "stabilizer.py"
)
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)


def W(text, step=0.4):
    """Гипотеза -> [(слово, конец_сек)], слова идут каждые step секунд."""
    return [(w, (i + 1) * step) for i, w in enumerate(text.split())]


def feed(s, text, tail=0.1):
    words = W(text)
    return s.update(words, words[-1][1] + tail)


def test_word_is_committed_after_two_agreeing_drafts_outside_guard_zone():
    s = st.Stabilizer(guard_sec=0.5)
    u = feed(s, "проверяем транскри")
    assert u.committed == [] and u.tentative == ["проверяем", "транскри"]

    u = feed(s, "проверяем транскрипцию через")
    assert u.committed == ["проверяем"], "совпало дважды и вне опасной зоны"
    assert u.new_words == ["проверяем"] and not u.rewrite
    assert u.tentative == ["транскрипцию", "через"]

    u = feed(s, "Проверяем транскрипцию через гигачат")
    assert u.committed[:2] == ["проверяем", "транскрипцию"]
    assert "гигачат" in u.tentative, "последнее слово у края не фиксируется"


def test_comparison_ignores_case_and_punctuation():
    s = st.Stabilizer(guard_sec=0.5)
    feed(s, "привет мир как")
    u = feed(s, "Привет, мир! Как дела")
    assert [st.norm_word(w) for w in u.committed] == ["привет", "мир", "как"]
    assert u.committed[0] == "Привет,", "слово берётся в написании текущей гипотезы"
    assert not u.rewrite


def test_committed_words_are_append_only_without_rewrite():
    s = st.Stabilizer(guard_sec=0.5)
    seen = []
    for text in ["раз два", "раз два три", "раз два три четыре", "раз два три четыре пять"]:
        u = feed(s, text)
        assert u.committed[: len(seen)] == seen, "зафиксированное не меняется"
        seen = u.committed
    assert seen[:3] == ["раз", "два", "три"]


def test_single_disagreeing_draft_does_not_rewrite():
    s = st.Stabilizer(guard_sec=0.5)
    feed(s, "мне двадцать пять лет")
    u = feed(s, "мне двадцать пять лет и")
    committed = list(u.committed)
    assert committed[:2] == ["мне", "двадцать"]

    u = feed(s, "мне 25 лет и я")  # один тик с другим началом — шум
    assert not u.rewrite and u.committed == committed


def test_two_agreeing_drafts_rewrite_committed_prefix():
    """Нормализация чисел задним числом: «двадцать пять» -> «25»."""
    s = st.Stabilizer(guard_sec=0.5)
    feed(s, "мне двадцать пять лет")
    feed(s, "мне двадцать пять лет и")
    feed(s, "мне 25 лет и я живу")
    u = feed(s, "мне 25 лет и я живу в")
    assert u.rewrite and u.changed_from == 1
    assert u.committed[:3] == ["мне", "25", "лет"]


def test_revised_from_reports_first_changed_word_of_the_final():
    assert st.revised_from(["мне", "двадцать"], "Мне 25 лет.") == 1
    assert st.revised_from(["мне"], "Мне 25 лет.") is None
    assert st.revised_from([], "что угодно") is None
    assert st.revised_from(["раз", "два", "три"], "Раз, два.") == 2
