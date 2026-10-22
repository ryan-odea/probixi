from __future__ import annotations

from pathlib import Path

import pytest

from probixi.io import CellParams, Geometry, read_crystfel_cell, read_geometry

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _offline_citation(monkeypatch):
    # the CLI pulls the citation from GitHub at start; keep tests off the network
    import urllib.request

    def offline(*args, **kwargs):
        raise OSError("tests run offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)


@pytest.fixture(scope="session")
def cell_file() -> Path:
    return FIXTURES / "bR.cell"


@pytest.fixture(scope="session")
def geom_file() -> Path:
    return FIXTURES / "Eiger4M.geom"


@pytest.fixture(scope="session")
def multipanel_geom_file() -> Path:
    # two 64x32 panels of a 4-D (N, panel, ss, fs) array tile a 64x64 image
    return FIXTURES / "MultiPanel.geom"


@pytest.fixture
def cell(cell_file: Path) -> CellParams:
    # bacteriorhodopsin, hexagonal P, a = b = 62.23 A, c = 110.77 A
    return read_crystfel_cell(cell_file)


@pytest.fixture
def geometry(geom_file: Path) -> Geometry:
    # Eiger 4M, one 2070 x 2167 panel, 12398 eV (1.0 A), 75 um pixels
    return read_geometry(geom_file)


@pytest.fixture
def geometry_dict(geometry: Geometry) -> dict:
    return geometry.to_dict()


# Some Metal builds cannot compile torch's MPS matmul2d shader; skip only that error
_METAL_COMPILE_MARKERS = (
    "Failed to created pipeline state object",
    "Domain=CompilerError",
)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except RuntimeError as exc:
        text = str(exc)
        if "mps" in item.keywords and any(m in text for m in _METAL_COMPILE_MARKERS):
            pytest.skip(
                f"MPS shader compilation unsupported by this Metal build: {text[:120]}"
            )
        raise
