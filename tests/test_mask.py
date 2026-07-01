from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sim

from probixi import Probixi
from probixi.io import read_geometry, read_mask
from probixi.io.assemble import _apply_mask_bits


def _single_panel_geom(
    tmp_path: Path,
    *,
    h: int = 8,
    w: int = 6,
    max_adu: int = 65535,
    mask_path: str = "/entry/data/mask",
    mask_file: str | None = None,
    mask_good: str = "0x0",
    mask_bad: str = "0xFFFFFFFF",
    bad_region: tuple[int, int, int, int] | None = None,
) -> Path:
    lines = [
        "clen = 0.1",
        "photon_energy = 12398.0",
        "res = 13333.3",
        f"max_adu = {max_adu}",
        "data = /entry/data/data",
        "dim0 = %",
        "dim1 = ss",
        "dim2 = fs",
        f"mask = {mask_path}",
        f"mask_good = {mask_good}",
        f"mask_bad = {mask_bad}",
    ]
    if mask_file is not None:
        lines.append(f"mask_file = {mask_file}")
    if bad_region is not None:
        min_ss, max_ss, min_fs, max_fs = bad_region
        lines += [
            f"bad/min_ss = {min_ss}",
            f"bad/max_ss = {max_ss}",
            f"bad/min_fs = {min_fs}",
            f"bad/max_fs = {max_fs}",
        ]
    lines += [
        "0/min_fs = 0",
        f"0/max_fs = {w - 1}",
        "0/min_ss = 0",
        f"0/max_ss = {h - 1}",
        "0/corner_x = -2.5",
        "0/corner_y = -3.5",
        "0/fs = +1.0x +0.0y",
        "0/ss = +0.0x +1.0y",
    ]
    path = tmp_path / "single.geom"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- the CrystFEL two-clause bit rule ----------------------------------------


def test_mask_bits_all_ones_bad_only_zero_is_good():
    raw = np.array([[0, 1, 0xFF]], dtype=np.uint32)
    good = _apply_mask_bits(raw, mask_good=0x0, mask_bad=0xFFFFFFFF)
    assert good.tolist() == [[True, False, False]]


def test_mask_bits_two_clause_rule():
    # GOOD iff bit0 set (mask_good) AND bit1 clear (mask_bad)
    raw = np.array([0b00, 0b01, 0b10, 0b11], dtype=np.uint32)
    good = _apply_mask_bits(raw, mask_good=0x1, mask_bad=0x2)
    assert good.tolist() == [False, True, False, False]


def test_mask_bits_handles_signed_input():
    raw = np.array([0, 1, 2], dtype=np.int32)
    good = _apply_mask_bits(raw, mask_good=0x0, mask_bad=0xFFFFFFFF)
    assert good.tolist() == [True, False, False]


# --- read_mask: external file, missing file, multi-panel assembly ------------


def test_read_mask_external_file_2d(tmp_path):
    mask = np.zeros((8, 6), dtype=np.int32)
    mask[2, 3] = 1
    mfile = sim.write_external_mask(tmp_path / "m.h5", mask, mask_path="/pixel_mask")
    geom = read_geometry(
        _single_panel_geom(tmp_path, mask_path="/pixel_mask", mask_file=str(mfile))
    )
    good = read_mask(geom, "unused.h5", (8, 6))
    assert good.shape == (8, 6)
    assert not bool(good[2, 3])
    assert bool(good[0, 0])


def test_read_mask_missing_file_warns_and_returns_none(tmp_path):
    geom = read_geometry(_single_panel_geom(tmp_path, mask_file="/no/such/file.h5"))
    with pytest.warns(UserWarning):
        good = read_mask(geom, "also_missing.h5", (8, 6))
    assert good is None


def test_read_mask_slices_one_event_from_per_event_mask(tmp_path):
    # a per-event (N, ss, fs) mask: read_mask must slice event 0, not the whole stack
    h, w = 8, 6
    mask = np.zeros((3, h, w), dtype=np.int32)
    mask[0, 2, 3] = 1  # bad in event 0 -> must show up
    mask[1, 5, 5] = 1  # bad only in event 1 -> must NOT show up
    cxi, _ = sim.write_cxi_run(
        tmp_path, np.zeros((3, h, w), dtype=np.float32), mask=mask
    )
    geom = read_geometry(_single_panel_geom(tmp_path, h=h, w=w))
    good = read_mask(geom, str(cxi), (h, w))
    assert good.shape == (h, w)
    assert not bool(good[2, 3])  # event-0 bad pixel
    assert bool(good[5, 5])  # bad in event 1 only -> good at event 0


def test_read_mask_assembles_multipanel(tmp_path, multipanel_geom_file):
    mask = np.zeros((2, 64, 32), dtype=np.int32)
    mask[0, 1, 1] = 1  # panel 0 -> data-space (1, 1)
    mask[1, 2, 2] = 1  # panel 1 -> data-space (2, 34)
    rng = np.random.Generator(np.random.PCG64(0))
    data4d = rng.normal(100, 10, size=(1, 2, 64, 32)).astype(np.float32)
    cxi, _ = sim.write_multipanel_run(tmp_path, data4d, mask=mask)
    geom = read_geometry(multipanel_geom_file)
    good = read_mask(geom, str(cxi), (64, 64))
    assert good.shape == (64, 64)
    assert not bool(good[1, 1])
    assert not bool(good[2, 32 + 2])
    assert bool(good[0, 0])


# --- folding into the pipeline valid mask ------------------------------------


def test_static_mask_combines_hdf5_mask_maxadu_and_bad_region(tmp_path, cell_file):
    h, w = 8, 6
    geom_path = _single_panel_geom(
        tmp_path, h=h, w=w, max_adu=1000, bad_region=(6, 7, 0, 1)
    )
    frames = np.full((4, h, w), 100.0, dtype=np.float32)
    frames[:, 0, 0] = 2000.0  # >= max_adu -> masked
    mask = np.zeros((h, w), dtype=np.int32)
    mask[h - 1, w - 1] = 1  # hdf5-masked corner
    _, lst = sim.write_cxi_run(
        tmp_path, frames, mask=mask, mask_path="/entry/data/mask"
    )

    px = Probixi(list_file=lst, geometry_file=geom_path, cell_file=cell_file)
    px.fit_noise(list(px.frames()))
    vm = px.noise.valid_mask

    assert not bool(vm[0, 0])  # max_adu
    assert not bool(vm[6, 0])  # bad region
    assert not bool(vm[h - 1, w - 1])  # hdf5 mask
    assert bool(vm[3, 3])  # untouched


def test_static_mask_without_mask_spec_is_unaffected(tmp_path, cell_file):
    # a geom with no mask block -> valid mask is just max_adu + bad regions
    geom_path = tmp_path / "nomask.geom"
    geom_path.write_text(
        "clen = 0.1\nphoton_energy = 12398.0\nres = 13333.3\nmax_adu = 65535\n"
        "data = /entry/data/data\ndim0 = %\ndim1 = ss\ndim2 = fs\n"
        "0/min_fs = 0\n0/max_fs = 5\n0/min_ss = 0\n0/max_ss = 7\n"
        "0/corner_x = -2.5\n0/corner_y = -3.5\n",
        encoding="utf-8",
    )
    frames = np.full((3, 8, 6), 100.0, dtype=np.float32)
    _, lst = sim.write_cxi_run(tmp_path, frames)
    px = Probixi(list_file=lst, geometry_file=geom_path, cell_file=cell_file)
    px.fit_noise(list(px.frames()))
    assert bool(px.noise.valid_mask.all())
