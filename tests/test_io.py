from __future__ import annotations

import numpy as np
import pytest
import sim
import torch

from probixi.io.cell import CellParams
from probixi.io.frames import DataLoader, iter_frames
from probixi.io.geometry import Geometry
from probixi.io.metadata import H5Info, Metadata, scan_h5


def _frames(n: int, h: int, w: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.normal(100.0, 10.0, size=(n, h, w)).astype(np.float32)


def test_scan_h5_finds_3d_dataset_and_shape(tmp_path):
    h5_path, _ = sim.write_run(tmp_path, _frames(5, 8, 6))
    info = scan_h5(h5_path)
    assert isinstance(info, H5Info)
    assert info.filename == str(h5_path)
    # write_run defaults to this internal path
    assert info.dataset == "entry/data/data"
    assert info.n_frames == 5
    assert info.frame_shape == (8, 6)


def test_scan_h5_raises_when_no_3d_dataset(tmp_path):
    import h5py

    path = tmp_path / "flat.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("entry/data/data", data=np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        scan_h5(path)


def test_loader_builds_metadata(tmp_path, geom_file, cell_file):
    h5_path, lst_path = sim.write_run(tmp_path, _frames(7, 8, 6))
    loader = DataLoader(lst_path, geom_file, cell_file)
    md = loader.metadata
    assert isinstance(md, Metadata)
    assert md.n_files == 1
    assert md.n_frames == 7
    assert md.frame_size == (8, 6)
    assert set(md.files) == {str(h5_path)}
    assert isinstance(md.files[str(h5_path)], H5Info)
    assert isinstance(md.geometry, Geometry)
    assert isinstance(md.cell, CellParams)
    assert len(loader) == 7
    assert loader.files is md.files


def test_loader_metadata_without_geom_and_cell(tmp_path):
    _, lst_path = sim.write_run(tmp_path, _frames(3, 8, 6))
    loader = DataLoader(lst_path)
    assert loader.metadata.geometry is None
    assert loader.metadata.cell is None
    assert loader.metadata.n_frames == 3


def test_loader_sums_frames_across_two_files(tmp_path):
    a_h5, _ = sim.write_run(tmp_path / "a", _frames(4, 8, 6, seed=1))
    b_h5, _ = sim.write_run(tmp_path / "b", _frames(6, 8, 6, seed=2))
    lst = tmp_path / "both.lst"
    lst.write_text(f"{a_h5}\n{b_h5}\n", encoding="utf-8")
    md = DataLoader(lst).metadata
    assert md.n_files == 2
    assert md.n_frames == 10
    assert md.frame_size == (8, 6)
    assert set(md.files) == {str(a_h5), str(b_h5)}


def test_inconsistent_frame_sizes_raise(tmp_path):
    a_h5, _ = sim.write_run(tmp_path / "a", _frames(3, 8, 6))
    b_h5, _ = sim.write_run(tmp_path / "b", _frames(3, 10, 6))
    lst = tmp_path / "mixed.lst"
    lst.write_text(f"{a_h5}\n{b_h5}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        DataLoader(lst)


def test_missing_list_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataLoader(tmp_path / "does_not_exist.lst")


def test_missing_geometry_file_raises(tmp_path):
    _, lst_path = sim.write_run(tmp_path, _frames(2, 8, 6))
    with pytest.raises(FileNotFoundError):
        DataLoader(lst_path, geometry_file=tmp_path / "nope.geom")


def test_missing_cell_file_raises(tmp_path):
    _, lst_path = sim.write_run(tmp_path, _frames(2, 8, 6))
    with pytest.raises(FileNotFoundError):
        DataLoader(lst_path, cell_file=tmp_path / "nope.cell")


def test_iter_frames_total_count_and_single_frame_shape(tmp_path):
    _, lst_path = sim.write_run(tmp_path, _frames(5, 8, 6))
    loader = DataLoader(lst_path)
    out = list(iter_frames(loader, batch_size=1))
    assert len(out) == 5
    for t in out:
        assert t.shape == (8, 6)


def test_iter_frames_batched_shape(tmp_path):
    _, lst_path = sim.write_run(tmp_path, _frames(7, 8, 6))
    loader = DataLoader(lst_path)
    out = list(iter_frames(loader, batch_size=3))
    # 7 frames in batches of 3 -> 3, 3, 1
    assert [t.shape[0] for t in out] == [3, 3, 1]
    for t in out:
        assert t.ndim == 3
        assert t.shape[1:] == (8, 6)
    assert sum(t.shape[0] for t in out) == 7


def test_iter_frames_recovers_exact_frame_values(tmp_path):
    frames = _frames(4, 8, 6, seed=11)
    _, lst_path = sim.write_run(tmp_path, frames)
    loader = DataLoader(lst_path)
    out = list(iter_frames(loader, batch_size=1, dtype=torch.float64))
    got = torch.stack(out)
    expected = torch.as_tensor(frames, dtype=torch.float64)
    assert torch.equal(got, expected)


def test_iter_frames_start_stop_slices_subset(tmp_path):
    frames = _frames(8, 8, 6, seed=5)
    _, lst_path = sim.write_run(tmp_path, frames)
    loader = DataLoader(lst_path)
    out = list(iter_frames(loader, start=2, stop=6, batch_size=1, dtype=torch.float64))
    assert len(out) == 4
    got = torch.stack(out)
    expected = torch.as_tensor(frames[2:6], dtype=torch.float64)
    assert torch.equal(got, expected)


def test_iter_frames_start_stop_across_two_files(tmp_path):
    fa = _frames(4, 8, 6, seed=1)
    fb = _frames(4, 8, 6, seed=2)
    a_h5, _ = sim.write_run(tmp_path / "a", fa)
    b_h5, _ = sim.write_run(tmp_path / "b", fb)
    lst = tmp_path / "both.lst"
    lst.write_text(f"{a_h5}\n{b_h5}\n", encoding="utf-8")
    loader = DataLoader(lst)
    # global [3, 6) spans last frame of a and first two of b
    out = list(iter_frames(loader, start=3, stop=6, batch_size=1, dtype=torch.float64))
    got = torch.stack(out)
    expected = torch.as_tensor(
        np.concatenate([fa[3:4], fb[0:2]], axis=0), dtype=torch.float64
    )
    assert torch.equal(got, expected)


def test_iter_frames_dtype_is_honored(tmp_path):
    _, lst_path = sim.write_run(tmp_path, _frames(3, 8, 6))
    loader = DataLoader(lst_path)
    for t in iter_frames(loader, dtype=torch.float64):
        assert t.dtype == torch.float64
    for t in iter_frames(loader, dtype=torch.float32):
        assert t.dtype == torch.float32


def test_iter_frames_files_subset(tmp_path):
    fa = _frames(3, 8, 6, seed=1)
    fb = _frames(3, 8, 6, seed=2)
    a_h5, _ = sim.write_run(tmp_path / "a", fa)
    b_h5, _ = sim.write_run(tmp_path / "b", fb)
    lst = tmp_path / "both.lst"
    lst.write_text(f"{a_h5}\n{b_h5}\n", encoding="utf-8")
    loader = DataLoader(lst)
    out = list(
        iter_frames(loader, files=[str(b_h5)], batch_size=1, dtype=torch.float64)
    )
    got = torch.stack(out)
    expected = torch.as_tensor(fb, dtype=torch.float64)
    assert torch.equal(got, expected)


def test_iter_frames_terminates_without_full_consumption(tmp_path):
    # abandon the iterator early; the prefetch thread must not hang the gc/close
    _, lst_path = sim.write_run(tmp_path, _frames(20, 8, 6))
    loader = DataLoader(lst_path)
    it = iter_frames(loader, batch_size=1, prefetch=2)
    first = next(it)
    assert first.shape == (8, 6)
    it.close()  # triggers the finally: stop_event + worker.join


def test_iter_frames_full_consumption_joins_worker(tmp_path):
    import threading

    before = threading.active_count()
    _, lst_path = sim.write_run(tmp_path, _frames(6, 8, 6))
    loader = DataLoader(lst_path)
    out = list(iter_frames(loader, batch_size=2))
    assert sum(t.shape[0] for t in out) == 6
    # the daemon prefetch thread is joined on clean termination
    assert threading.active_count() == before
