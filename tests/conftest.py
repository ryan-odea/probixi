from __future__ import annotations

import os
from pathlib import Path

import pytest

from probixi.io import CellParams, Geometry, read_crystfel_cell, read_geometry

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def cell_file() -> Path:
    """Path to the committed real bacteriorhodopsin ``.cell`` fixture."""
    return FIXTURES / "bR.cell"


@pytest.fixture(scope="session")
def geom_file() -> Path:
    """Path to the committed real Eiger 4M ``.geom`` fixture."""
    return FIXTURES / "Eiger4M.geom"


@pytest.fixture
def cell(cell_file: Path) -> CellParams:
    """Parsed bacteriorhodopsin cell (hexagonal P, a=b=62.23, c=110.77 A)."""
    return read_crystfel_cell(cell_file)


@pytest.fixture
def geometry(geom_file: Path) -> Geometry:
    """Parsed Eiger 4M geometry (single panel, ~1 A, 75 um pixels)."""
    return read_geometry(geom_file)


@pytest.fixture
def geometry_dict(geometry: Geometry) -> dict:
    """Indexer/writer-style geometry dict (beam_center, clen, pixel_size, ...)."""
    return geometry.to_dict()


def _resolve_real_frames() -> Path | None:
    env = os.environ.get("PROBIXI_TEST_DATA")
    candidates = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).parent
    candidates.append(here / "test_data" / "br_frames.h5")
    candidates.extend(sorted(here.glob("*.h5")))
    candidates.extend(sorted((here / "test_data").glob("*.h5")))
    for path in candidates:
        if path.is_file():
            return path
    return None


@pytest.fixture(scope="session")
def real_frames() -> Path:
    """A local real frame stack, or skip; never committed."""
    path = _resolve_real_frames()
    if path is None:
        pytest.skip(
            "no real frame stack found; set PROBIXI_TEST_DATA or drop an .h5 "
            "into tests/test_data/ to run real-data tests"
        )
    return path
