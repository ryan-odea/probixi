from __future__ import annotations

import math

import sim
import torch

from probixi.indexer.indexer import Indexer, RefineConfig, SeedConfig

# seeding/refinement that sim.py verified as fast and reliable on CPU
SEED = SeedConfig(n_directions=1500, n_spin=60, top_directions=12, max_candidates=32)
REFINE = RefineConfig(max_iters=150, reassign_every=10)
ROT_SEED = 0
MAX_ANGLE_DEG = 8.0


def _make_indexer(geometry_dict, cell):
    return Indexer(geometry_dict, cell, seed=SEED, refine=REFINE)


def test_index_frames_recovers_known_cell_and_orientation(geometry_dict, cell):
    U = sim.proper_rotation(ROT_SEED, max_angle_deg=MAX_ANGLE_DEG)
    positions, hkl = sim.lattice_peaks(geometry_dict, cell, U)
    idxr = _make_indexer(geometry_dict, cell)
    results = idxr.index_frames({0: positions})

    r = results[0]
    n_peaks = positions.shape[0]

    edges_obs = sorted([r.cell.a, r.cell.b, r.cell.c])
    edges_tgt = sorted([cell.a, cell.b, cell.c])
    for o, t in zip(edges_obs, edges_tgt):
        assert abs(o - t) / t < 0.02

    angles_obs = sorted([r.cell.alpha, r.cell.beta, r.cell.gamma])
    angles_tgt = sorted([cell.alpha, cell.beta, cell.gamma])
    for o, t in zip(angles_obs, angles_tgt):
        assert abs(o - t) < math.radians(2.0)

    assert r.rmsd < 0.01
    assert r.n_peaks == n_peaks
    assert r.n_indexed >= int(0.8 * n_peaks)
    assert int(r.indexed_mask.sum()) == r.n_indexed

    # A @ hkl reproduces the observed q for indexed peaks
    q_obs = idxr.lift(positions)
    mask = r.indexed_mask
    q_pred = r.hkl[mask].to(r.A.dtype) @ r.A.transpose(-1, -2)
    resid = torch.linalg.vector_norm(q_pred - q_obs[mask], dim=-1)
    assert float(resid.max()) < idxr.q_tolerance


def test_index_frames_assigns_integer_hkl_to_indexed_peaks(geometry_dict, cell):
    U = sim.proper_rotation(ROT_SEED, max_angle_deg=MAX_ANGLE_DEG)
    positions, _ = sim.lattice_peaks(geometry_dict, cell, U)
    idxr = _make_indexer(geometry_dict, cell)
    r = idxr.index_frames({0: positions})[0]

    hkl_indexed = r.hkl[r.indexed_mask]
    # no indexed peak is assigned the origin
    assert int((hkl_indexed.abs().sum(dim=-1) == 0).sum()) == 0
    assert r.A.dtype == torch.float64


def test_index_frames_is_deterministic(geometry_dict, cell):
    U = sim.proper_rotation(ROT_SEED, max_angle_deg=MAX_ANGLE_DEG)
    positions, _ = sim.lattice_peaks(geometry_dict, cell, U)

    a = _make_indexer(geometry_dict, cell).index_frames({0: positions})[0]
    b = _make_indexer(geometry_dict, cell).index_frames({0: positions})[0]

    assert torch.allclose(a.A, b.A)
    assert a.n_indexed == b.n_indexed
    assert torch.equal(a.indexed_mask, b.indexed_mask)


def test_subpixel_jitter_still_indexes_majority_with_bounded_rmsd(geometry_dict, cell):
    U = sim.proper_rotation(ROT_SEED, max_angle_deg=MAX_ANGLE_DEG)
    positions, _ = sim.lattice_peaks(geometry_dict, cell, U)
    idxr = _make_indexer(geometry_dict, cell)

    clean = idxr.index_frames({0: positions})[0]

    g = torch.Generator().manual_seed(123)
    jitter = 0.3 * torch.randn(positions.shape, generator=g, dtype=positions.dtype)
    noisy_positions = positions + jitter

    noisy = idxr.index_frames({0: noisy_positions})[0]

    assert noisy.n_indexed >= int(0.7 * positions.shape[0])
    # jitter inflates the residual but it stays within tolerance
    assert noisy.rmsd >= clean.rmsd
    assert noisy.rmsd < idxr.q_tolerance

    # recovered cell still matches under the noise
    edges_obs = sorted([noisy.cell.a, noisy.cell.b, noisy.cell.c])
    edges_tgt = sorted([cell.a, cell.b, cell.c])
    for o, t in zip(edges_obs, edges_tgt):
        assert abs(o - t) / t < 0.03


def test_spurious_off_lattice_peaks_are_excluded(geometry_dict, cell):
    U = sim.proper_rotation(ROT_SEED, max_angle_deg=MAX_ANGLE_DEG)
    positions, _ = sim.lattice_peaks(geometry_dict, cell, U)
    n_real = positions.shape[0]

    panel = next(iter(geometry_dict["panels"].values()))
    max_ss = float(panel["max_ss"])
    max_fs = float(panel["max_fs"])
    g = torch.Generator().manual_seed(7)
    n_spurious = 8
    spurious = torch.stack(
        [
            torch.rand(n_spurious, generator=g, dtype=positions.dtype) * max_ss,
            torch.rand(n_spurious, generator=g, dtype=positions.dtype) * max_fs,
        ],
        dim=-1,
    )
    augmented = torch.cat([positions, spurious], dim=0)

    idxr = _make_indexer(geometry_dict, cell)
    r = idxr.index_frames({0: augmented})[0]

    assert r.n_peaks == n_real + n_spurious
    assert r.n_indexed < r.n_peaks
    assert int(r.indexed_mask.sum()) == r.n_indexed
    # the real lattice peaks are still recovered in bulk
    assert r.n_indexed >= int(0.8 * n_real)
    # most spurious peaks (trailing rows) are rejected
    spurious_indexed = int(r.indexed_mask[n_real:].sum())
    assert spurious_indexed <= n_spurious // 2
