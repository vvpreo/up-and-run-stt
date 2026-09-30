#!/usr/bin/env python3
"""
Генерация иконок всех стендов из прод-исходника (src/static/icons/favicon.svg).

Запускается при сборке образа (стадия builder), результат — артефакт сборки,
в git не хранится:

    <out>/<stand>/favicon.svg
    <out>/<stand>/favicon.ico            (16, 32, 48)
    <out>/<stand>/apple-touch-icon.png   (180)
    <out>/<stand>/icon-192.png, icon-512.png

Стенды: prod (без рамки), dev / test / uat (рамка, см. src/utils/icons.py).
Какой набор отдавать, сервис решает в рантайме по APP_ENV — образ один на все
стенды, и в нём лежат все варианты.

    uv run --no-project --with resvg-py python scripts/gen_icons.py --out src/static/icons/generated
"""

import argparse
import importlib.util
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Модуль грузится по пути, без импорта пакета src: на стадии сборки
# зависимостей сервиса в окружении нет.
_spec = importlib.util.spec_from_file_location("icons", ROOT / "src" / "utils" / "icons.py")
icons = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(icons)


def render_png(svg: str, size: int) -> bytes:
    import resvg_py

    return bytes(resvg_py.svg_to_bytes(svg_string=svg, width=size, height=size))


def make_ico(pngs: dict[int, bytes]) -> bytes:
    """ICO-контейнер с PNG внутри (поддерживается всеми браузерами и ОС с Vista)."""
    header = struct.pack("<HHH", 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    entries, blobs = b"", b""
    for size, data in sorted(pngs.items()):
        entries += struct.pack(
            "<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset + len(blobs)
        )
        blobs += data
    return header + entries + blobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "src" / "static" / "icons" / "generated"))
    args = ap.parse_args()
    out = Path(args.out)

    for stand in icons.STANDS:
        d = out / stand
        d.mkdir(parents=True, exist_ok=True)
        (d / "favicon.svg").write_text(icons.badged_svg(stand), encoding="utf-8")
        ico = {s: render_png(icons.badged_svg(stand, size_px=s), s) for s in (16, 32, 48)}
        (d / "favicon.ico").write_bytes(make_ico(ico))
        for name, size in (("apple-touch-icon.png", 180), ("icon-192.png", 192), ("icon-512.png", 512)):
            (d / name).write_bytes(render_png(icons.badged_svg(stand, size_px=size), size))
        print(f"{stand}: {sorted(p.name for p in d.iterdir())}")


if __name__ == "__main__":
    main()
