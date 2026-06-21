from __future__ import annotations

import pytest

from probixi.io.metadata import H5Info
from probixi.io.writer import (
    DataOffloader,
    _build_frame_ranges,
    _panel_bounds,
    _resolution_line,
)


@pytest.mark.parametrize(
    "recip, expected",
    [
        (0.0, "0.000000 nm^-1 or inf A"),
        (-1.0, "0.000000 nm^-1 or inf A"),
        (2.0, "2.000000 nm^-1 or 5.000000 A"),  # d = 10 / recip
    ],
)
def test_resolution_line(recip, expected):
    assert _resolution_line(recip) == expected


def test_panel_bounds_from_real_geometry(geometry):
    bounds = _panel_bounds(geometry.panels)
    assert bounds == [("0", 0, 2069, 0, 2166)]


def test_panel_bounds_skips_malformed_entries():
    panels = {
        "good": {"min_fs": 0, "max_fs": 9, "min_ss": 0, "max_ss": 9},
        "bad": {"min_fs": 0},  # missing keys -> skipped
    }
    assert _panel_bounds(panels) == [("good", 0, 9, 0, 9)]


def test_panel_bounds_empty():
    assert _panel_bounds(None) == []


def test_build_frame_ranges_is_cumulative():
    files = {
        "a.h5": H5Info("a.h5", "/d", n_frames=3, frame_shape=(8, 8)),
        "b.h5": H5Info("b.h5", "/d", n_frames=5, frame_shape=(8, 8)),
    }
    assert _build_frame_ranges(files) == [(0, 3, "a.h5"), (3, 8, "b.h5")]


def test_build_frame_ranges_empty():
    assert _build_frame_ranges(None) == []


def _writer(geometry_dict, tmp_path, **kw):
    return DataOffloader(tmp_path / "out.stream", geometry=geometry_dict, **kw)


def test_resolution_nm_inv_zero_at_beam_center(geometry_dict, tmp_path):
    off = _writer(geometry_dict, tmp_path)
    row, col = geometry_dict["beam_center"]
    assert off._resolution_nm_inv(row, col) == pytest.approx(0.0)


def test_resolution_nm_inv_increases_with_radius(geometry_dict, tmp_path):
    off = _writer(geometry_dict, tmp_path)
    row, col = geometry_dict["beam_center"]
    near = off._resolution_nm_inv(row + 50, col)
    far = off._resolution_nm_inv(row + 500, col)
    assert 0.0 < near < far


def test_panel_for_inside_and_fallback(geometry_dict, tmp_path):
    off = _writer(geometry_dict, tmp_path, panel="FALLBACK")
    assert off._panel_for(fs=100.0, ss=100.0) == "0"  # inside panel "0"
    assert off._panel_for(fs=9999.0, ss=100.0) == "FALLBACK"  # outside all panels


def test_locate_maps_global_index_to_file_and_event(geometry_dict, tmp_path):
    files = {
        "a.h5": H5Info("a.h5", "/d", n_frames=3, frame_shape=(8, 8)),
        "b.h5": H5Info("b.h5", "/d", n_frames=5, frame_shape=(8, 8)),
    }
    off = _writer(geometry_dict, tmp_path, files=files)
    assert off._locate(0) == ("a.h5", 0)
    assert off._locate(2) == ("a.h5", 2)
    assert off._locate(3) == ("b.h5", 0)
    assert off._locate(7) == ("b.h5", 4)


def test_locate_handles_none_and_out_of_range(geometry_dict, tmp_path):
    off = _writer(geometry_dict, tmp_path, files={})
    assert off._locate(None) == ("unknown", 0)
    assert off._locate(100) == ("unknown", 100)


def test_write_before_entering_context_raises(geometry_dict, tmp_path):
    off = _writer(geometry_dict, tmp_path)
    with pytest.raises(RuntimeError, match="context manager"):
        off.write(object())
