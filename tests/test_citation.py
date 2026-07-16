from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import probixi
from probixi import probixi as driver

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "docs" / "assets" / "citation.bib"


def _version_from(text: str) -> str:
    m = re.search(r"version\s*[=:]\s*[\"{]?\s*([0-9][^\"}\s]*)", text)
    assert m is not None, f"no version found in:\n{text}"
    return m.group(1)


def test_exports():
    assert callable(probixi.citation)
    assert isinstance(probixi.__citation__, str)
    assert "@software" in probixi.__citation__


def test_offline_falls_back_to_default_silently(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    out = driver.citation(timeout=0.01)

    assert out == driver._CITATION.strip() + "\n"
    assert "@software{odea_probixi" in out
    assert capsys.readouterr().out == out  # printed, no warning


def test_online_fetch_is_used_when_available(monkeypatch, capsys):
    online = "@software{odea_probixi,\n  version = {9.9.9}\n}\n"

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return online.encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    out = driver.citation()

    assert out == online.strip() + "\n"
    assert capsys.readouterr().out == out


def test_asset_default_and_pyproject_versions_agree():
    pyproject = (ROOT / "pyproject.toml").read_text()
    py_version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject).group(1)

    assert _version_from(BIB.read_text()) == py_version
    assert _version_from(driver._CITATION) == py_version
