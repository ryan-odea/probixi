from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
from torch import Tensor

from probixi.indexer.forward import q_to_detector
from probixi.indexer.lattice import cell_to_B
from probixi.io.cell import CellParams

PathLike = Union[str, Path]

SATURATED_VALUE = 65535.0
DEAD_VALUE = 0.0


@dataclass
class SimTruth:
    # positions (N, 2) as (row, col); intensities (N,) total counts; A = U @ B
    positions: Tensor
    intensities: Tensor
    hkl: Tensor
    U: Tensor
    cell: CellParams
    background: float
    noise_sigma: float
    frame_shape: tuple[int, int]


def synthetic_geometry(
    h: int = 256,
    w: int = 256,
    clen: float = 0.2,
    pixel_size: float = 75e-6,
    wavelength: float = 1.0,
) -> dict:
    # beam at ((h - 1) / 2, (w - 1) / 2); clen and pixel_size in m, wavelength in A
    bc_row = (h - 1) / 2.0
    bc_col = (w - 1) / 2.0
    panels = {
        "0": {
            "min_fs": 0,
            "max_fs": w - 1,
            "min_ss": 0,
            "max_ss": h - 1,
            # read_geometry: beam_center = (-corner_y, -corner_x)
            "corner_x": -bc_col,
            "corner_y": -bc_row,
            "fs": "+1.0x +0.0y",
            "ss": "+0.0x +1.0y",
        }
    }
    return {
        "beam_center": (bc_row, bc_col),
        "clen": float(clen),
        "pixel_size": float(pixel_size),
        "wavelength": float(wavelength),
        "panels": panels,
    }


def render_frame(
    shape: tuple[int, int],
    positions,
    intensities,
    *,
    psf_sigma: float = 1.3,
    background: float = 100.0,
    noise_sigma: float = 10.0,
    seed: int = 0,
    saturated_pixels=None,
    dead_pixels=None,
) -> np.ndarray:
    # Each spot adds its total count as a unit-sum Gaussian (psf_sigma px) at (row, col)
    H, W = int(shape[0]), int(shape[1])
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    frame = rng.normal(background, noise_sigma, size=(H, W)).astype(np.float32)

    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
    inten = np.asarray(intensities, dtype=np.float32).reshape(-1)
    if pos.shape[0] != inten.shape[0]:
        raise ValueError("positions and intensities must have equal length")

    radius = max(1, int(math.ceil(4.0 * psf_sigma)))
    two_s2 = 2.0 * psf_sigma * psf_sigma
    for (r0, c0), total in zip(pos, inten):
        ri = int(round(r0))
        ci = int(round(c0))
        rlo, rhi = max(0, ri - radius), min(H, ri + radius + 1)
        clo, chi = max(0, ci - radius), min(W, ci + radius + 1)
        if rlo >= rhi or clo >= chi:
            continue
        rr = np.arange(rlo, rhi, dtype=np.float32)[:, None]
        cc = np.arange(clo, chi, dtype=np.float32)[None, :]
        g = np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / two_s2)
        s = g.sum()
        if s <= 0:
            continue
        frame[rlo:rhi, clo:chi] += (total / s) * g

    if dead_pixels is not None:
        d = np.asarray(dead_pixels, dtype=np.int64).reshape(-1, 2)
        frame[d[:, 0], d[:, 1]] = DEAD_VALUE
    if saturated_pixels is not None:
        s = np.asarray(saturated_pixels, dtype=np.int64).reshape(-1, 2)
        frame[s[:, 0], s[:, 1]] = SATURATED_VALUE

    return frame.astype(np.float32)


def simulate_noise_frames(
    shape: tuple[int, int],
    n: int,
    *,
    background: float = 100.0,
    noise_sigma: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    # (n, H, W) signal-free frames; frame i draws from PCG64(seed + i)
    H, W = int(shape[0]), int(shape[1])
    out = np.empty((int(n), H, W), dtype=np.float32)
    for i in range(int(n)):
        rng = np.random.Generator(np.random.PCG64(int(seed) + i))
        out[i] = rng.normal(background, noise_sigma, size=(H, W)).astype(np.float32)
    return out


def lattice_peaks(
    geometry_dict: dict,
    cell: CellParams,
    U: Tensor,
    *,
    max_index: int = 6,
) -> tuple[Tensor, Tensor]:
    # (row, col) and hkl of the on-panel reflections with max |h|, |k|, |l| <= max_index
    dtype = torch.float32
    B = cell_to_B(cell, dtype=dtype)
    Um = U.to(dtype=dtype)
    A = Um @ B  # q = A @ hkl ; columns of A map hkl -> q

    rng = range(-int(max_index), int(max_index) + 1)
    hkl = torch.tensor(
        [list(t) for t in product(rng, rng, rng) if any(t)],
        dtype=dtype,
    )
    q = hkl @ A.transpose(-1, -2)  # (M, 3)

    # Ewald excitation error |S| - 1 with S = lambda q + zhat
    wl = float(geometry_dict["wavelength"])
    S = q * wl
    S[:, 2] += 1.0
    Snorm = torch.linalg.vector_norm(S, dim=-1)
    near_ewald = (Snorm - 1.0).abs() < 0.0025
    forward = S[:, 2] > 0.0  # scattered ray heads toward the detector plane
    sel = near_ewald & forward
    q = q[sel]
    hkl = hkl[sel]
    if q.shape[0] == 0:
        return q.new_empty(0, 2), hkl.new_empty(0, 3)

    pos = q_to_detector(q, geometry_dict, dtype=dtype)  # (N, 2) (row, col)

    panel = next(iter(geometry_dict["panels"].values()))
    min_ss, max_ss = float(panel["min_ss"]), float(panel["max_ss"])
    min_fs, max_fs = float(panel["min_fs"]), float(panel["max_fs"])
    rows, cols = pos[:, 0], pos[:, 1]
    on = (rows >= min_ss) & (rows <= max_ss) & (cols >= min_fs) & (cols <= max_fs)
    return pos[on], hkl[on]


def simulate_indexable_frame(
    geometry_dict: dict,
    cell: CellParams,
    U: Tensor,
    *,
    peak_intensity: float = 5000.0,
    psf_sigma: float = 1.3,
    background: float = 100.0,
    noise_sigma: float = 10.0,
    seed: int = 0,
) -> tuple[np.ndarray, SimTruth]:
    panel = next(iter(geometry_dict["panels"].values()))
    H = int(panel["max_ss"]) + 1
    W = int(panel["max_fs"]) + 1
    positions, hkl = lattice_peaks(geometry_dict, cell, U)
    intensities = torch.full(
        (positions.shape[0],), float(peak_intensity), dtype=torch.float32
    )
    frame = render_frame(
        (H, W),
        positions.numpy(),
        intensities.numpy(),
        psf_sigma=psf_sigma,
        background=background,
        noise_sigma=noise_sigma,
        seed=seed,
    )
    truth = SimTruth(
        positions=positions,
        intensities=intensities,
        hkl=hkl,
        U=U.to(dtype=torch.float32),
        cell=cell,
        background=float(background),
        noise_sigma=float(noise_sigma),
        frame_shape=(H, W),
    )
    return frame, truth


def write_run(
    dirpath: PathLike,
    frames: np.ndarray,
    *,
    dataset: str = "entry/data/data",
) -> tuple[Path, Path]:
    # scan_h5 takes the first 3-D dataset, which is the only one written here
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("frames must be (N, H, W)")
    h5_path = dirpath / "sim_run.h5"
    lst_path = dirpath / "sim_run.lst"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(dataset, data=arr)
    lst_path.write_text(f"{h5_path}\n", encoding="utf-8")
    return h5_path, lst_path


def write_cxi_run(
    dirpath: PathLike,
    frames: np.ndarray,
    *,
    data_path: str = "/entry/data/data",
    decoy: bool = False,
    decoy_path: str = "/aaa_decoy/data",
    mask: Optional[np.ndarray] = None,
    mask_path: str = "/entry/data/mask",
    extra_datasets: Optional[dict] = None,
) -> tuple[Path, Path]:
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("frames must be (N, H, W)")
    cxi_path = dirpath / "cxi_run.cxi"
    lst_path = dirpath / "cxi_run.lst"
    with h5py.File(cxi_path, "w") as f:
        # The decoy sorts first, so only a reader that honours geometry data= skips it
        if decoy:
            decoy_arr = np.zeros((1, arr.shape[1], arr.shape[2]), dtype=np.float32)
            f.create_dataset(decoy_path, data=decoy_arr)
        f.create_dataset(data_path, data=arr)
        # mask is 2-D (static) or (N, H, W) per event, the LCLS/Cheetah layout
        if mask is not None:
            f.create_dataset(mask_path, data=np.asarray(mask))
        # e.g. per-event /LCLS/photon_energy_eV that a geometry reads photon_energy from
        for path, values in (extra_datasets or {}).items():
            f.create_dataset(path, data=np.asarray(values))
    lst_path.write_text(f"{cxi_path}\n", encoding="utf-8")
    return cxi_path, lst_path


def write_multipanel_run(
    dirpath: PathLike,
    data4d: np.ndarray,
    *,
    data_path: str = "/entry/data/data",
    mask: Optional[np.ndarray] = None,
    mask_path: str = "/entry/data/mask",
) -> tuple[Path, Path]:
    # Axis 1 is the panel a geometry's dimN = <int> selects; mask is (panel, ss, fs)
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data4d, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError("data4d must be (N, panel, ss, fs)")
    cxi_path = dirpath / "mp_run.cxi"
    lst_path = dirpath / "mp_run.lst"
    with h5py.File(cxi_path, "w") as f:
        f.create_dataset(data_path, data=arr)
        if mask is not None:
            f.create_dataset(mask_path, data=np.asarray(mask))
    lst_path.write_text(f"{cxi_path}\n", encoding="utf-8")
    return cxi_path, lst_path


def write_external_mask(
    path: PathLike,
    mask_array: np.ndarray,
    *,
    mask_path: str = "/pixel_mask",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(mask_path, data=np.asarray(mask_array))
    return path


def proper_rotation(seed: int, max_angle_deg: Optional[float] = None) -> Tensor:
    # Random axis; angle in [0.25, 1) x max_angle_deg if given, else in [0, 2 pi)
    g = torch.Generator()
    g.manual_seed(int(seed))
    axis = torch.randn(3, generator=g, dtype=torch.float32)
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
    if max_angle_deg is not None:
        frac = torch.rand(1, generator=g, dtype=torch.float32).item()
        angle = math.radians(float(max_angle_deg)) * (0.25 + 0.75 * frac)
    else:
        angle = float(torch.rand(1, generator=g, dtype=torch.float32).item()) * (
            2.0 * math.pi
        )
    omega = axis * angle
    theta = angle
    eye = torch.eye(3, dtype=torch.float32)
    if theta < 1e-12:
        return eye
    axis_hat = omega / theta
    xh, yh, zh = axis_hat.tolist()
    Kh = torch.tensor(
        [[0.0, -zh, yh], [zh, 0.0, -xh], [-yh, xh, 0.0]], dtype=torch.float32
    )
    R = eye + math.sin(theta) * Kh + (1.0 - math.cos(theta)) * (Kh @ Kh)
    return R


if __name__ == "__main__":
    # Standalone check of indexing, noise calibration, and peakfinding
    import time

    from probixi.indexer.indexer import Indexer, RefineConfig, SeedConfig
    from probixi.io.cell import read_crystfel_cell
    from probixi.io.geometry import read_geometry
    from probixi.peakfinding.noise.calibrate import calibrate_noise
    from probixi.peakfinding.noise.model import NoiseModel
    from probixi.peakfinding.peaks.peakfinder import PeakFinder

    here = Path(__file__).parent
    cell = read_crystfel_cell(here / "fixtures" / "bR.cell")
    geom = read_geometry(here / "fixtures" / "Eiger4M.geom").to_dict()

    U = proper_rotation(0, max_angle_deg=8)
    positions, hkl = lattice_peaks(geom, cell, U)
    print(f"[index] on-frame reflections: {positions.shape[0]}")
    seedcfg = SeedConfig(
        n_directions=1500, n_spin=60, top_directions=12, max_candidates=32
    )
    idxr = Indexer(
        geom, cell, seed=seedcfg, refine=RefineConfig(max_iters=150, reassign_every=10)
    )
    t0 = time.perf_counter()
    res = idxr.index_frames({0: positions})
    dt = time.perf_counter() - t0
    r = res.get(0)
    assert r is not None, "indexing returned no result"
    edges = sorted([r.cell.a, r.cell.b, r.cell.c])
    print(
        f"[index] n_indexed={r.n_indexed}/{positions.shape[0]} "
        f"rmsd={r.rmsd:.4f} edges={[round(e, 2) for e in edges]} "
        f"time={dt:.2f}s"
    )
    tgt = sorted([cell.a, cell.b, cell.c])
    assert all(abs(o - t) / t < 0.02 for o, t in zip(edges, tgt)), "cell mismatch"
    assert r.n_indexed >= int(0.8 * positions.shape[0]), "too few indexed"
    assert r.rmsd < 0.01, "rmsd too large"

    nframes = simulate_noise_frames((128, 128), 40)
    nm = NoiseModel((128, 128), mode="online")
    cal = calibrate_noise(nm, [torch.from_numpy(f) for f in nframes])
    cal.apply(nm)
    z = nm.z_score(torch.from_numpy(nframes[0]), combination="calibrated")
    zstd = float(z.std())
    print(f"[noise] z.std={zstd:.3f} kappa={cal.kappa:.2f}")
    assert abs(zstd - 1.0) < 0.15, "z std off"
    assert cal.kappa > 1.0, "kappa <= 1"

    pshape = (160, 160)
    grng = np.random.Generator(np.random.PCG64(7))
    gridr = np.array([30, 30, 30, 80, 80, 80, 130, 130, 130, 105], dtype=float)
    gridc = np.array([30, 80, 130, 30, 80, 130, 30, 80, 130, 105], dtype=float)
    ppos = np.stack(
        [gridr + grng.uniform(-5, 5, 10), gridc + grng.uniform(-5, 5, 10)], axis=1
    )
    pframe = render_frame(pshape, ppos, [3000.0] * 10, seed=3)
    calib_frames = simulate_noise_frames(pshape, 30, seed=100)
    pnm = NoiseModel(pshape, mode="online")
    pcal = calibrate_noise(pnm, [torch.from_numpy(f) for f in calib_frames])
    finder = PeakFinder(pnm, size_max=80)
    pcal.apply(pnm, finder)
    pks = finder.peak_stream([torch.from_numpy(pframe)]).collect_peaks()
    det = torch.tensor([[p.row, p.col] for p in pks], dtype=torch.float32)
    matched = 0
    for tp in ppos:
        if det.shape[0]:
            d = torch.linalg.vector_norm(det - torch.tensor(tp), dim=1).min()
            if float(d) < 1.5:
                matched += 1
    print(f"[peaks] detected={len(pks)} matched={matched}/10")
    assert matched >= 8, "too few peaks matched"
    print("ALL CHECKS PASSED")
