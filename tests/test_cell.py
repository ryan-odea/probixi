from __future__ import annotations

import math

import pytest

from probixi.io import CellParams, read_crystfel_cell


def test_reads_real_bR_cell(cell):
    assert cell.a == pytest.approx(62.23)
    assert cell.b == pytest.approx(62.23)
    assert cell.c == pytest.approx(110.77)
    assert cell.lattice_type == "hexagonal"
    assert cell.unique_axis == "c"
    assert cell.centering == "P"


def test_angles_parsed_in_radians(cell):
    assert cell.alpha == pytest.approx(math.radians(90.0))
    assert cell.beta == pytest.approx(math.radians(90.0))
    assert cell.gamma == pytest.approx(math.radians(120.0))


def test_short_angle_aliases(tmp_path):
    path = tmp_path / "c.cell"
    path.write_text(
        "a = 10 A\nb = 20 A\nc = 30 A\nal = 90 deg\nbe = 90 deg\nga = 90 deg\n"
    )
    cell = read_crystfel_cell(path)
    assert (cell.a, cell.b, cell.c) == pytest.approx((10.0, 20.0, 30.0))
    assert cell.alpha == pytest.approx(math.pi / 2)


def test_inline_comments_are_stripped(tmp_path):
    path = tmp_path / "c.cell"
    path.write_text(
        "a = 50.0 A ; the long axis\nb = 50.0 A\nc = 50.0 A\n"
        "alpha = 90 deg\nbeta = 90 deg\ngamma = 90 deg\n"
    )
    assert read_crystfel_cell(path).a == pytest.approx(50.0)


def test_missing_keys_raises(tmp_path):
    path = tmp_path / "c.cell"
    path.write_text("a = 10 A\nb = 20 A\nc = 30 A\n")  # no angles
    with pytest.raises(ValueError, match="missing keys"):
        read_crystfel_cell(path)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_crystfel_cell("does-not-exist.cell")


def test_volume_cubic():
    cell = CellParams(10.0, 10.0, 10.0, math.pi / 2, math.pi / 2, math.pi / 2)
    assert cell.volume == pytest.approx(1000.0)


def test_volume_hexagonal_matches_closed_form(cell):
    # gamma=120: V = a^2 c sqrt(3) / 2
    expected = cell.a * cell.a * cell.c * math.sqrt(3.0) / 2.0
    assert cell.volume == pytest.approx(expected, rel=1e-9)


def test_volume_degenerate_clamped_nonnegative():
    # negative discriminant clamps volume to 0
    cell = CellParams(
        10.0, 10.0, 10.0, math.radians(20), math.radians(20), math.radians(170)
    )
    assert cell.volume == 0.0


def test_as_dict_degrees(cell):
    d = cell.as_dict_degrees()
    assert d["alpha_deg"] == pytest.approx(90.0)
    assert d["gamma_deg"] == pytest.approx(120.0)
    assert d["volume_A3"] == pytest.approx(cell.volume)
    assert d["lattice_type"] == "hexagonal"


def test_as_dict_omits_absent_metadata():
    cell = CellParams(10.0, 10.0, 10.0, math.pi / 2, math.pi / 2, math.pi / 2)
    d = cell.as_dict_degrees()
    assert "lattice_type" not in d
    assert "centering" not in d
