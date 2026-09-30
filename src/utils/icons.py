"""
Иконка приложения и её варианты для стендов.

Прод-исходник один: src/static/icons/favicon.svg. Для непродовых стендов поверх
него рисуется сплошная рамка по внешнему краю — чтобы вкладку dev/test/uat
нельзя было спутать с продом:

    prod — без маркера (ровно исходник)
    dev  — красная  #E53935
    test — синяя    #1E88E5
    uat  — зелёная  #43A047

Рамка рисуется ВНУТРИ холста и повторяет форму иконки (скруглённый квадрат),
толщина ~9 % стороны, но не тоньше 2 px на мелких размерах (16×16). Кроме
рамки в иконке ничего не меняется.

Модуль без зависимостей: его использует и сервис (SVG отдаётся на лету), и
scripts/gen_icons.py при сборке образа (PNG/ICO всех стендов).
"""

from pathlib import Path

SOURCE_SVG = Path(__file__).resolve().parent.parent / "static" / "icons" / "favicon.svg"

CANVAS = 512          # сторона viewBox исходника
CORNER = 112          # радиус скругления фона в исходнике
FRAME_SHARE = 0.09    # толщина рамки в долях стороны

STAND_COLORS = {
    "dev": "#E53935",
    "test": "#1E88E5",
    "uat": "#43A047",
}
STANDS = ("prod", *STAND_COLORS)


def normalize_stand(value: str | None) -> str:
    """APP_ENV -> prod|dev|test|uat. Неизвестное значение — ошибка, а не «прод»:
    немаркированная иконка — это утверждение, что сборка продовая."""
    stand = (value or "prod").strip().lower()
    aliases = {"production": "prod", "development": "dev", "staging": "uat"}
    stand = aliases.get(stand, stand)
    if stand not in STANDS:
        raise ValueError(f"APP_ENV must be one of {STANDS}, got {value!r}")
    return stand


def badged_svg(stand: str, size_px: int | None = None, source: str | None = None) -> str:
    """
    SVG иконки для стенда. Для prod — исходник без изменений.

    size_px — целевой размер растра: на мелких размерах рамка утолщается,
    чтобы остаться не тоньше 2 px.
    """
    svg = source if source is not None else SOURCE_SVG.read_text(encoding="utf-8")
    stand = normalize_stand(stand)
    if stand == "prod":
        return svg

    share = FRAME_SHARE
    if size_px:
        share = max(share, 2.0 / size_px)
    width = CANVAS * share
    inset = width / 2
    frame = (
        f'  <rect x="{inset:.1f}" y="{inset:.1f}" '
        f'width="{CANVAS - width:.1f}" height="{CANVAS - width:.1f}" '
        f'rx="{max(CORNER - inset, 0):.1f}" fill="none" '
        f'stroke="{STAND_COLORS[stand]}" stroke-width="{width:.1f}"/>\n'
    )
    head, sep, tail = svg.rpartition("</svg>")
    return head + frame + sep + tail
