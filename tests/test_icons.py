"""
Иконка приложения и маркировка стендов.

Иконки открыты без токена (браузер запрашивает их до авторизации). На проде
иконка без маркера; на dev/test/uat поверх неё рамка цвета стенда — вкладку
непродового стенда нельзя спутать с продом. Стенд берётся из /health (app_env),
поэтому тест верен против любого экземпляра.
"""

import requests

STAND_COLORS = {"dev": "#E53935", "test": "#1E88E5", "uat": "#43A047"}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _stand(base_url):
    return requests.get(f"{base_url}/health", timeout=10).json()["app_env"]


def test_favicon_svg_is_public_and_marked_for_the_stand(base_url):
    r = requests.get(f"{base_url}/favicon.svg", timeout=10)  # без токена
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    svg = r.text
    assert svg.lstrip().startswith("<svg") and "</svg>" in svg

    stand = _stand(base_url)
    if stand == "prod":
        # Маркер на проде — баг: немаркированная иконка и есть признак прода
        assert not any(c in svg for c in STAND_COLORS.values())
    else:
        assert STAND_COLORS[stand] in svg
        assert not any(c in svg for s, c in STAND_COLORS.items() if s != stand)


def test_raster_icons_are_served(base_url):
    ico = requests.get(f"{base_url}/favicon.ico", timeout=10)
    assert ico.status_code == 200 and ico.headers["content-type"] == "image/x-icon"
    assert ico.content[:4] == b"\x00\x00\x01\x00", "не ICO-контейнер"

    for name in ("apple-touch-icon.png", "icon-192.png", "icon-512.png"):
        r = requests.get(f"{base_url}/{name}", timeout=10)
        assert r.status_code == 200, name
        assert r.content[:8] == PNG_MAGIC, name


def test_page_links_the_icons(base_url):
    html = requests.get(f"{base_url}/", timeout=10).text
    assert 'href="/favicon.svg"' in html
    assert 'href="/favicon.ico"' in html
    assert 'href="/apple-touch-icon.png"' in html
