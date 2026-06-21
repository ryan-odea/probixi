from __future__ import annotations

import pytest

from probixi.io import Geometry, read_geometry
from probixi.io.geometry import EV_ANGSTROM


def test_reads_real_eiger_geometry(geometry):
    assert geometry.distance == pytest.approx(0.2007)
    assert geometry.parameters["max_adu"] == pytest.approx(12287.0)
    assert len(geometry.panels) == 1
    assert "0" in geometry.panels


def test_pixel_size_from_res(geometry):
    # pixel size = 1 / res metres
    assert geometry.pixel_size == pytest.approx(1.0 / 13333.3)


def test_wavelength_from_photon_energy(geometry):
    assert geometry.wavelength == pytest.approx(EV_ANGSTROM / 12398.0)


def test_beam_center_from_single_panel(geometry):
    # beam_center = (-corner_y, -corner_x) for a single panel
    row, col = geometry.beam_center
    assert row == pytest.approx(1080.459892)
    assert col == pytest.approx(1020.401102)


def test_bad_regions_parsed(geometry):
    by_name = {br.name: br for br in geometry.bad_regions}
    assert set(by_name) == {"bad_shadow", "bad_shadow_2"}
    shadow = by_name["bad_shadow"]
    assert (shadow.min_ss, shadow.max_ss) == (1032, 1128)
    assert (shadow.min_fs, shadow.max_fs) == (0, 1075)


def test_scalar_coercion_keeps_strings(geometry):
    # numerics coerce to int/float; the dataset path stays a string
    assert isinstance(geometry.parameters["clen"], float)
    assert geometry.parameters["data"] == "/entry/data/data"


def test_to_dict_exposes_required_fields(geometry):
    d = geometry.to_dict()
    assert set(d) == {"beam_center", "clen", "pixel_size", "wavelength", "panels"}
    assert d["clen"] == pytest.approx(0.2007)


def test_to_dict_reports_missing_fields():
    geom = Geometry(parameters={}, panels={})
    with pytest.raises(ValueError, match="missing required fields"):
        geom.to_dict()


def test_clen_given_as_hdf5_path_is_not_a_distance(tmp_path):
    path = tmp_path / "p.geom"
    path.write_text(
        "clen = /entry/instrument/detector/distance\n"
        "res = 13333.3\nwavelength = 1.0\n"
        "0/min_fs = 0\n0/max_fs = 9\n0/min_ss = 0\n0/max_ss = 9\n"
        "0/corner_x = -5\n0/corner_y = -5\n"
    )
    geom = read_geometry(path)
    assert geom.distance is None


def test_multi_panel_has_no_single_beam_center(tmp_path):
    path = tmp_path / "p.geom"
    panel = (
        "{n}/min_fs = 0\n{n}/max_fs = 9\n{n}/min_ss = 0\n{n}/max_ss = 9\n"
        "{n}/corner_x = -5\n{n}/corner_y = -5\n"
    )
    path.write_text(panel.format(n="p0") + panel.format(n="p1"))
    geom = read_geometry(path)
    assert len(geom.panels) == 2
    assert geom.beam_center is None


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_geometry("does-not-exist.geom")
