from __future__ import annotations

import math

import pytest
import torch

from probixi.indexer.lattice import B_to_cell, cell_to_B
from probixi.io import CellParams

DT = torch.float32

CUBIC = CellParams(10.0, 10.0, 10.0, math.pi / 2, math.pi / 2, math.pi / 2)
ORTHO = CellParams(10.0, 20.0, 30.0, math.pi / 2, math.pi / 2, math.pi / 2)
MONO = CellParams(10.0, 15.0, 20.0, math.pi / 2, math.radians(110.0), math.pi / 2)
HEX = CellParams(62.23, 62.23, 110.77, math.pi / 2, math.pi / 2, math.radians(120.0))


def test_cell_to_B_orthorhombic_is_diagonal():
    B = cell_to_B(ORTHO, dtype=DT)
    expected = torch.diag(torch.tensor([0.1, 0.05, 1.0 / 30.0], dtype=DT))
    assert torch.allclose(B, expected, atol=1e-6)


def test_q_of_h00_has_magnitude_one_over_a():
    B = cell_to_B(ORTHO, dtype=DT)
    q = B @ torch.tensor([1.0, 0.0, 0.0], dtype=DT)
    assert float(torch.linalg.vector_norm(q)) == pytest.approx(1.0 / ORTHO.a)


@pytest.mark.parametrize(
    "cell", [CUBIC, ORTHO, MONO, HEX], ids=["cubic", "ortho", "mono", "hex"]
)
def test_B_to_cell_recovers_parameters(cell):
    recovered = B_to_cell(cell_to_B(cell, dtype=DT))
    assert recovered.a == pytest.approx(cell.a, rel=1e-6)
    assert recovered.b == pytest.approx(cell.b, rel=1e-6)
    assert recovered.c == pytest.approx(cell.c, rel=1e-6)
    assert recovered.alpha == pytest.approx(cell.alpha, abs=1e-6)
    assert recovered.beta == pytest.approx(cell.beta, abs=1e-6)
    assert recovered.gamma == pytest.approx(cell.gamma, abs=1e-6)


def test_cell_to_B_rejects_zero_gamma():
    with pytest.raises(ValueError, match="gamma"):
        cell_to_B(CellParams(10.0, 10.0, 10.0, math.pi / 2, math.pi / 2, 0.0), dtype=DT)


def test_cell_to_B_rejects_degenerate_cell():
    # alpha=beta=gamma=150 deg has no realisable cell (cz^2 < 0)
    bad = CellParams(10.0, 10.0, 10.0, *(math.radians(150.0),) * 3)
    with pytest.raises(ValueError, match="cz"):
        cell_to_B(bad, dtype=DT)


def test_B_to_cell_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        B_to_cell(torch.eye(2, dtype=DT))
