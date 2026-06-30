from __future__ import annotations

import pytest

from probixi.io import Geometry, read_geometry
from probixi.io.geometry import EV_ANGSTROM, _parse_mask_bits


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


def test_multi_panel_beam_center_is_shared_data_space_center(tmp_path):
    # Two side-by-side panels sharing one beam at data-space (row=5, col=15):
    # beam = (min_ss - corner_y, min_fs - corner_x) for each panel.
    path = tmp_path / "p.geom"
    path.write_text(
        "res = 13333.3\nwavelength = 1.0\nclen = 0.1\n"
        # p0 spans fs 0..9
        "p0/min_fs = 0\np0/max_fs = 9\np0/min_ss = 0\np0/max_ss = 9\n"
        "p0/corner_x = -15\np0/corner_y = -5\n"
        # p1 spans fs 10..19, same beam
        "p1/min_fs = 10\np1/max_fs = 19\np1/min_ss = 0\np1/max_ss = 9\n"
        "p1/corner_x = -5\np1/corner_y = -5\n"
    )
    geom = read_geometry(path)
    assert len(geom.panels) == 2
    row, col = geom.beam_center
    assert row == pytest.approx(5.0)
    assert col == pytest.approx(15.0)
    # the whole point: a multi-panel geom can now build the indexer dict
    assert geom.to_dict()["beam_center"] == (pytest.approx(5.0), pytest.approx(15.0))


def test_data_layout_parsed_from_eiger(geometry):
    layout = geometry.data_layout
    assert layout is not None
    assert layout.data_path == "/entry/data/data"
    assert layout.dims == ["%", "ss", "fs"]
    assert (layout.event_axis, layout.ss_axis, layout.fs_axis) == (0, 1, 2)
    assert layout.fixed == {}


def test_mask_spec_parsed_from_eiger(geometry):
    spec = geometry.mask_spec
    assert spec is not None
    assert spec.mask_path == "/pixel_mask"
    assert spec.mask_file.endswith("mask.h5")
    assert spec.mask_good == 0x0
    assert spec.mask_bad == 0xFFFFFFFF


def test_parse_mask_bits_handles_hex_and_int():
    assert _parse_mask_bits("0xFFFFFFFF") == 0xFFFFFFFF
    assert _parse_mask_bits("0x0") == 0
    assert _parse_mask_bits(65535) == 65535
    assert _parse_mask_bits(None, default=7) == 7


def test_multipanel_layout_has_fixed_panel_selectors(multipanel_geom_file):
    geom = read_geometry(multipanel_geom_file)
    assert set(geom.panels) == {"0", "1"}
    # top-level layout omits the panel axis; each panel pins it via dim1
    assert geom.panel_layouts["0"].fixed == {1: 0}
    assert geom.panel_layouts["1"].fixed == {1: 1}
    assert geom.panel_layouts["0"].dims == ["%", 0, "ss", "fs"]
    assert geom.panel_layouts["1"].event_axis == 0
    # a multi-panel geometry now builds the indexer dict
    assert geom.beam_center is not None
    assert geom.to_dict()["beam_center"] is not None


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_geometry("does-not-exist.geom")
