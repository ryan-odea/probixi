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
