from __future__ import annotations

import numpy as np
import torch

from probixi.io import render_frame
from probixi.io.visualize import _xy


def test_render_frame_writes_file_with_overlays(tmp_path):
    img = np.random.default_rng(0).normal(100, 5, (64, 80)).astype("float32")
    out = tmp_path / "f.png"
    p = render_frame(
        img,
        path=out,
        peaks=[[10, 20], [30, 40]],
        reflections=[[11, 21]],
        title="t",
    )
    assert p == out
    assert out.is_file() and out.stat().st_size > 0


def test_render_frame_accepts_tensor_and_no_overlays(tmp_path):
    img = torch.randn(32, 48) + 100.0
    out = tmp_path / "g.png"
    render_frame(img, path=out)
    assert out.is_file()


def test_render_frame_mask_excludes_hot_pixels(tmp_path):
    img = np.full((40, 40), 50.0, dtype="float32")
    img[0, 0] = 1e9
    mask = np.ones_like(img, dtype=bool)
    mask[0, 0] = False
    out = tmp_path / "m.png"
    render_frame(img, path=out, mask=mask)
    assert out.is_file()


def test_xy_swaps_row_col_and_handles_empty():
    xy = _xy([[5, 9]])
    assert xy.tolist() == [[9, 5]]
    assert _xy(None) is None
    assert _xy(np.empty((0, 2))) is None


def test_physical_place_matches_assemble(tmp_path):
    from probixi.io import build_physical_assembler, read_geometry

    geom = tmp_path / "tiled.geom"
    geom.write_text(
        "clen = 0.05\nres = 20000\nphoton_energy = 10000\n"
        "a/min_fs = 0\na/max_fs = 7\na/min_ss = 0\na/max_ss = 15\n"
        "a/fs = -1.0y\na/ss = +1.0x\na/corner_x = 2\na/corner_y = 10\n"
        "b/min_fs = 0\nb/max_fs = 7\nb/min_ss = 16\nb/max_ss = 31\n"
        "b/fs = +1.0y\nb/ss = -1.0x\nb/corner_x = -3\nb/corner_y = -9\n"
    )
    assemble, _, _, place = build_physical_assembler(read_geometry(geom))
    raw = np.zeros((32, 8))
    points = np.array([[1, 2], [5, 7], [20, 0], [31, 6]])
    raw[points[:, 0], points[:, 1]] = np.arange(1, 5)
    y, x = np.rint(place(points)).astype(int).T
    np.testing.assert_array_equal(assemble(raw)[y, x], np.arange(1, 5))
    assert np.isnan(place([[40.0, 0.0]])).all()
