from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sim
import torch

from probixi import Probixi
from probixi.io import DataLoader, iter_frames, read_geometry, read_mask, scan_h5
from probixi.io.geometry import EV_ANGSTROM


def _frames(n: int, h: int, w: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.normal(100.0, 10.0, size=(n, h, w)).astype(np.float32)


CLEN_PATH = "/LCLS/detector_1/EncoderValue"
ENERGY_PATH = "/LCLS/photon_energy_eV"


def _lcls_geom(
    tmp_path: Path,
    *,
    h: int = 8,
    w: int = 6,
    clen: str = CLEN_PATH,
    photon_energy: str = ENERGY_PATH,
    coffset: float | None = 0.583,
) -> Path:
    # geom with clen/photon_energy as HDF5 paths (LCLS CSPAD case)
    lines = [
        f"clen = {clen}",
        f"photon_energy = {photon_energy}",
        "res = 13333.3",
        "data = /entry/data/data",
        "dim0 = %",
        "dim1 = ss",
        "dim2 = fs",
        "0/min_fs = 0",
        f"0/max_fs = {w - 1}",
        "0/min_ss = 0",
        f"0/max_ss = {h - 1}",
        "0/corner_x = -2.5",
        "0/corner_y = -3.5",
    ]
    if coffset is not None:
        lines.append(f"0/coffset = {coffset}")
    path = tmp_path / "lcls.geom"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- resolving HDF5-path clen / photon_energy against the data file ----------


def test_resolves_hdf5_clen_and_wavelength_from_data_file(tmp_path):
    geom = _lcls_geom(tmp_path, coffset=0.583)
    # encoder in mm (|value| > 2 m -> read as mm); energy chosen so lambda == 1 A
    _, lst = sim.write_cxi_run(
        tmp_path,
        _frames(3, 8, 6),
        extra_datasets={
            CLEN_PATH: np.array([-500.0, -500.0, -500.0], dtype=np.float64),
            ENERGY_PATH: np.full(3, EV_ANGSTROM, dtype=np.float64),
        },
    )
    g = DataLoader(lst, geom).metadata.geometry
    # clen = mean(encoder) mm -> m + coffset = -0.5 + 0.583
    assert g.distance == pytest.approx(0.083)
    assert g.wavelength == pytest.approx(1.0)
    # the whole point: the geometry dict is now complete for the indexer
    d = g.to_dict()
    assert d["clen"] == pytest.approx(0.083)
    assert d["wavelength"] == pytest.approx(1.0)


def test_clen_already_in_metres_is_not_rescaled(tmp_path):
    # a clen path holding a plausible metre value (|value| <= 2 m) is used as-is
    geom = _lcls_geom(tmp_path, coffset=0.0)
    _, lst = sim.write_cxi_run(
        tmp_path,
        _frames(2, 8, 6),
        extra_datasets={
            CLEN_PATH: np.array([0.092, 0.092], dtype=np.float64),
            ENERGY_PATH: np.full(2, 12398.0, dtype=np.float64),
        },
    )
    g = DataLoader(lst, geom).metadata.geometry
    assert g.distance == pytest.approx(0.092)


def test_missing_clen_dataset_warns_and_leaves_unresolved(tmp_path):
    geom = _lcls_geom(tmp_path)  # references CLEN_PATH / ENERGY_PATH
    _, lst = sim.write_cxi_run(tmp_path, _frames(2, 8, 6))  # but writes neither
    with pytest.warns(UserWarning):
        g = DataLoader(lst, geom).metadata.geometry
    assert g.distance is None and g.wavelength is None
    with pytest.raises(ValueError, match="missing required fields"):
        g.to_dict()


def test_clen_path_naming_a_group_warns_not_crashes(tmp_path):
    # a clen path pointing at a group (not a dataset) must warn, not crash
    geom = _lcls_geom(tmp_path, clen="/entry/data", photon_energy=ENERGY_PATH)
    _, lst = sim.write_cxi_run(
        tmp_path,
        _frames(2, 8, 6),
        extra_datasets={ENERGY_PATH: np.full(2, 12398.0, dtype=np.float64)},
    )
    with pytest.warns(UserWarning, match="not a dataset"):
        g = DataLoader(lst, geom).metadata.geometry
    assert g.distance is None  # group path -> unresolved, no crash
    assert g.wavelength == pytest.approx(EV_ANGSTROM / 12398.0)  # the valid one still resolves


# --- honoring the geom data= path over the first-3-D fallback ----------------


def test_scan_h5_honors_geom_data_path_over_decoy(tmp_path, geom_file):
    cxi, _ = sim.write_cxi_run(tmp_path, _frames(4, 8, 6), decoy=True)
    # no geometry: the first-3-D fallback grabs the decoy (its path sorts first)
    assert scan_h5(cxi).dataset == "aaa_decoy/data"
    # with geometry: the declared data path wins
    info = scan_h5(cxi, read_geometry(geom_file))
    assert info.dataset == "/entry/data/data"
    assert info.n_frames == 4
    assert info.frame_shape == (8, 6)
    assert info.placements is None  # plain 3-D stack -> fast path


def test_iter_frames_reads_real_data_not_decoy(tmp_path, geom_file):
    frames = _frames(4, 8, 6, seed=3)
    _, lst = sim.write_cxi_run(tmp_path, frames, decoy=True)
    loader = DataLoader(lst, geom_file)
    out = torch.stack(list(iter_frames(loader, dtype=torch.float64)))
    assert torch.equal(out, torch.as_tensor(frames, dtype=torch.float64))


# --- fast-path invariance for plain stacks -----------------------------------


def test_plain_stack_keeps_fast_path_with_and_without_geom(tmp_path, geom_file):
    h5, _ = sim.write_run(tmp_path, _frames(5, 8, 6, seed=7))
    info_no = scan_h5(h5)
    info_geo = scan_h5(h5, read_geometry(geom_file))
    assert info_no.placements is None and info_geo.placements is None
    assert info_no.frame_shape == info_geo.frame_shape == (8, 6)
    assert info_geo.dataset == "/entry/data/data"


# --- multi-panel / 4-D assembly ----------------------------------------------


def test_multipanel_assembly_recovers_dataspace_image(tmp_path, multipanel_geom_file):
    rng = np.random.Generator(np.random.PCG64(0))
    data4d = rng.normal(100, 10, size=(3, 2, 64, 32)).astype(np.float32)
    _, lst = sim.write_multipanel_run(tmp_path, data4d)
    loader = DataLoader(lst, multipanel_geom_file)
    assert loader.metadata.frame_size == (64, 64)
    info = next(iter(loader.metadata.files.values()))
    assert info.placements is not None and len(info.placements) == 2

    out = torch.stack(list(iter_frames(loader, dtype=torch.float64)))
    # panel 0 -> cols 0:32, panel 1 -> cols 32:64
    ref = np.concatenate([data4d[:, 0], data4d[:, 1]], axis=2)
    assert torch.equal(out, torch.as_tensor(ref, dtype=torch.float64))


def test_multipanel_assembly_batched(tmp_path, multipanel_geom_file):
    rng = np.random.Generator(np.random.PCG64(5))
    data4d = rng.normal(100, 10, size=(3, 2, 64, 32)).astype(np.float32)
    _, lst = sim.write_multipanel_run(tmp_path, data4d)
    loader = DataLoader(lst, multipanel_geom_file)
    out = list(iter_frames(loader, batch_size=2, dtype=torch.float64))
    assert [t.shape[0] for t in out] == [2, 1]
    got = torch.cat(out, dim=0)
    ref = np.concatenate([data4d[:, 0], data4d[:, 1]], axis=2)
    assert torch.equal(got, torch.as_tensor(ref, dtype=torch.float64))


def test_multipanel_layout_mismatch_fails_fast(tmp_path, multipanel_geom_file):
    # a wrong array shape must be rejected at load, not in the prefetch worker
    bad = np.zeros((2, 2, 64, 30), dtype=np.float32)  # fs 30 != 32
    _, lst = sim.write_multipanel_run(tmp_path, bad)
    with pytest.raises(ValueError, match="panel"):
        DataLoader(lst, multipanel_geom_file)


# --- end-to-end smoke over a multi-panel run ---------------------------------


def test_probixi_builds_and_runs_on_multipanel(
    tmp_path, multipanel_geom_file, cell_file
):
    rng = np.random.Generator(np.random.PCG64(1))
    data4d = rng.normal(100, 10, size=(8, 2, 64, 32)).astype(np.float32)
    mask = np.zeros((2, 64, 32), dtype=np.int32)
    mask[0, 5, 5] = 1  # nonzero -> bad under mask_bad=0xFFFFFFFF
    _, lst = sim.write_multipanel_run(tmp_path, data4d, mask=mask)

    px = Probixi(list_file=lst, geometry_file=multipanel_geom_file, cell_file=cell_file)
    assert px.geometry["beam_center"] is not None

    frames = list(px.frames())
    assert len(frames) == 8
    assert frames[0].shape == (64, 64)

    results = list(px.peak_stream(frames))
    assert len(results) == 8
    # the HDF5-masked pixel is excluded from the valid mask
    assert not bool(px.noise.valid_mask[5, 5])


# --- real LCLS CXI + optimised geom intake (skips without local data) --------


def test_real_cxi_and_optimised_geom_ingest(tmp_path, real_cxi, real_optimised_geom):
    """Real LCLS CXI + optimised CSPAD geom: clen/wavelength resolve, frames stream, mask slices."""
    lst = tmp_path / "real.lst"
    lst.write_text(f"{real_cxi}\n", encoding="utf-8")

    loader = DataLoader(lst, real_optimised_geom)
    g = loader.metadata.geometry
    frame_size = loader.metadata.frame_size
    assert g is not None and frame_size is not None
    frame_size = tuple(frame_size)

    # clen/wavelength were HDF5 paths; now resolved to physical values
    assert g.distance is not None and 0.0 < g.distance < 2.0
    assert g.wavelength is not None and 0.1 < g.wavelength < 10.0
    d = g.to_dict()  # complete -> the indexer/forward model can be built
    assert set(d) >= {"beam_center", "clen", "pixel_size", "wavelength", "panels"}

    # frames stream at the resolved data-space shape
    f0 = next(iter(iter_frames(loader, start=0, stop=1)))
    assert tuple(f0.shape) == frame_size

    # the (possibly per-event) mask reads as one slab matching the frame
    if g.mask_spec is not None and g.mask_spec.mask_path is not None:
        good = read_mask(g, str(real_cxi), frame_size)
        assert good is None or good.shape == frame_size

    # peak-only Probixi builds and exposes the resolved geometry
    px = Probixi(list_file=lst, geometry_file=real_optimised_geom)
    assert px.geometry["clen"] == pytest.approx(g.distance)
