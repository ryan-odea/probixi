from __future__ import annotations

import math
from itertools import product

import pytest
import sim
import torch

from probixi.indexer.lattice import cell_to_B
from probixi.indexer.refine import (
    RefineResult,
    _axis_angle_to_rotation,
    refine_multiframe_known_B,
)
from probixi.indexer.seed import sphere_seed_candidates

DT = torch.float64


def _synthetic_q(cell, U, max_index: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    # q = hkl @ (U @ cell_to_B(cell)).T over a small integer-hkl shell (origin dropped)
    B = cell_to_B(cell, dtype=DT)
    A = U.to(DT) @ B
    rng = range(-max_index, max_index + 1)
    hkl = torch.tensor(
        [list(t) for t in product(rng, rng, rng) if any(t)],
        dtype=DT,
    )
    q = hkl @ A.transpose(-1, -2)
    return q, hkl


def _best_candidate(
    A_cand: torch.Tensor, q: torch.Tensor, q_tolerance: float
) -> torch.Tensor:
    # the candidate explaining the most observed q via round(A^-1 q) -> q within tol
    A_inv = torch.linalg.inv(A_cand)
    hkl = torch.round(torch.einsum("cij,nj->cni", A_inv, q))
    q_pred = torch.einsum("cij,cnj->cni", A_cand, hkl)
    sq = ((q_pred - q.unsqueeze(0)) ** 2).sum(dim=-1)
    counts = (sq < q_tolerance * q_tolerance).sum(dim=-1)
    return A_cand[int(torch.argmax(counts))]


def test_sphere_seed_candidates_returns_candidates(cell):
    U = sim.proper_rotation(0, max_angle_deg=10)
    q, _ = _synthetic_q(cell, U)
    B = cell_to_B(cell, dtype=DT)
    cands = sphere_seed_candidates(
        q,
        B,
        q_tolerance=0.01,
        n_directions=1500,
        n_spin=60,
        top_directions=12,
        top_k=32,
    )
    assert cands.ndim == 3 and cands.shape[1:] == (3, 3)
    assert cands.shape[0] > 0
    assert cands.dtype == q.dtype


def test_best_candidate_maps_integer_hkl_within_tolerance(cell):
    U = sim.proper_rotation(1, max_angle_deg=10)
    q, _ = _synthetic_q(cell, U)
    B = cell_to_B(cell, dtype=DT)
    q_tol = 0.01
    cands = sphere_seed_candidates(
        q,
        B,
        q_tolerance=q_tol,
        n_directions=2000,
        n_spin=80,
        top_directions=16,
        top_k=32,
    )
    A = _best_candidate(cands, q, q_tol)
    hkl = torch.round(torch.linalg.solve(A, q.T).T)
    q_pred = (A @ hkl.T).T
    resid = torch.linalg.vector_norm(q_pred - q, dim=-1)
    inliers = resid < q_tol
    # the dominant lattice should explain most of the synthetic shell
    assert int(inliers.sum()) >= int(0.6 * q.shape[0])
    assert float(resid[inliers].max()) < q_tol


def test_best_candidate_reproduces_lattice_metric(cell):
    from probixi.indexer.lattice import decompose_A

    U = sim.proper_rotation(2, max_angle_deg=10)
    q, _ = _synthetic_q(cell, U)
    B = cell_to_B(cell, dtype=DT)
    q_tol = 0.01
    cands = sphere_seed_candidates(
        q,
        B,
        q_tolerance=q_tol,
        n_directions=2000,
        n_spin=80,
        top_directions=16,
        top_k=32,
    )
    A = _best_candidate(cands, q, q_tol).to(DT)
    _, _, rec = decompose_A(A)
    # orientation-invariant metric: sorted edges and volume match the target cell
    assert sorted((rec.a, rec.b, rec.c)) == pytest.approx(
        sorted((cell.a, cell.b, cell.c)), rel=1e-3
    )
    assert rec.volume == pytest.approx(cell.volume, rel=1e-3)


def test_identical_inputs_give_identical_candidates(cell):
    U = sim.proper_rotation(3, max_angle_deg=10)
    q, _ = _synthetic_q(cell, U)
    B = cell_to_B(cell, dtype=DT)
    kw = dict(
        q_tolerance=0.01, n_directions=1500, n_spin=60, top_directions=12, top_k=32
    )
    a = sphere_seed_candidates(q, B, **kw)
    b = sphere_seed_candidates(q, B, **kw)
    assert torch.equal(a, b)


def test_seed_rejects_bad_q_shape(cell):
    B = cell_to_B(cell, dtype=DT)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        sphere_seed_candidates(torch.zeros(5, 2, dtype=DT), B)


def test_seed_rejects_bad_B_shape(cell):
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        sphere_seed_candidates(torch.zeros(5, 3, dtype=DT), torch.eye(2, dtype=DT))


def test_seed_too_few_peaks_returns_empty(cell):
    B = cell_to_B(cell, dtype=DT)
    out = sphere_seed_candidates(torch.zeros(1, 3, dtype=DT), B)
    assert out.shape == (0, 3, 3)


def test_refine_from_near_truth_keeps_or_lowers_rmsd(cell):
    U = sim.proper_rotation(4, max_angle_deg=8)
    q, _ = _synthetic_q(cell, U)
    B = cell_to_B(cell, dtype=DT)
    A_true = U.to(DT) @ B
    # seed slightly off the truth so refinement has something to do
    A_seed = (sim.proper_rotation(99, max_angle_deg=0.5) @ A_true).unsqueeze(0)

    [res] = refine_multiframe_known_B(
        [A_seed], [q], q_tolerance=0.01, max_iters=120, reassign_every=20, min_indexed=6
    )
    assert isinstance(res, RefineResult)
    hist = res.history
    assert hist.numel() > 0
    # overall non-increasing trend: final loss does not exceed the initial loss
    assert float(hist[-1]) <= float(hist[0]) + 1e-9
    assert int(res.n_indexed[0]) >= 6
    assert float(res.rmsd[0]) < 0.01


def test_refine_recovers_orientation_from_perturbed_seed(cell):
    U = sim.proper_rotation(5, max_angle_deg=8)
    q, _ = _synthetic_q(cell, U)
    B = cell_to_B(cell, dtype=DT)
    A_true = U.to(DT) @ B
    A_seed = (sim.proper_rotation(123, max_angle_deg=0.4) @ A_true).unsqueeze(0)

    [res] = refine_multiframe_known_B(
        [A_seed], [q], q_tolerance=0.01, max_iters=150, reassign_every=20, min_indexed=6
    )
    # refined basis predicts the synthetic q within the tolerance for indexed peaks
    A_ref = res.A[0]
    indexed = res.indexed[0]
    assert int(indexed.sum()) >= 6
    q_pred = torch.einsum("ij,nj->ni", A_ref, res.hkl[0].to(DT))
    resid = torch.linalg.vector_norm(q_pred - q, dim=-1)
    assert float(resid[indexed].max()) < 0.01


def test_refine_empty_input_returns_empty_list():
    assert refine_multiframe_known_B([], []) == []


def test_refine_misaligned_frames_raises(cell):
    B = cell_to_B(cell, dtype=DT)
    A = (torch.eye(3, dtype=DT) @ B).unsqueeze(0)
    q, _ = _synthetic_q(cell, torch.eye(3, dtype=DT))
    with pytest.raises(ValueError, match="align"):
        refine_multiframe_known_B([A], [q, q])


def test_axis_angle_omega_zero_is_identity():
    R = _axis_angle_to_rotation(torch.zeros(3, dtype=DT))
    assert torch.allclose(R, torch.eye(3, dtype=DT), atol=1e-12)


def test_axis_angle_matches_rodrigues_for_known_axis_angle():
    angle = math.radians(37.0)
    axis = torch.tensor([1.0, 2.0, -2.0], dtype=DT)
    axis = axis / torch.linalg.vector_norm(axis)
    R = _axis_angle_to_rotation(axis * angle)
    x, y, z = axis.tolist()
    K = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=DT)
    eye = torch.eye(3, dtype=DT)
    expected = eye + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    assert torch.allclose(R, expected, atol=1e-12)


def test_axis_angle_rotates_vector_by_known_angle():
    # 90 deg about +z carries x_hat onto y_hat
    R = _axis_angle_to_rotation(torch.tensor([0.0, 0.0, math.pi / 2], dtype=DT))
    out = R @ torch.tensor([1.0, 0.0, 0.0], dtype=DT)
    assert torch.allclose(out, torch.tensor([0.0, 1.0, 0.0], dtype=DT), atol=1e-12)


def test_axis_angle_output_is_orthonormal_with_unit_determinant():
    g = torch.Generator().manual_seed(11)
    omega = torch.randn(3, generator=g, dtype=DT)
    R = _axis_angle_to_rotation(omega)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=DT), atol=1e-12)
    assert float(torch.linalg.det(R)) == pytest.approx(1.0, abs=1e-12)


def test_axis_angle_batched_shape_and_orthonormal():
    g = torch.Generator().manual_seed(12)
    omega = torch.randn(4, 5, 3, generator=g, dtype=DT)
    R = _axis_angle_to_rotation(omega)
    assert R.shape == (4, 5, 3, 3)
    eye = torch.eye(3, dtype=DT).expand(4, 5, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-12)
    dets = torch.linalg.det(R)
    assert torch.allclose(dets, torch.ones(4, 5, dtype=DT), atol=1e-12)


def test_axis_angle_rejects_bad_last_dim():
    with pytest.raises(ValueError, match="last dim 3"):
        _axis_angle_to_rotation(torch.zeros(2, dtype=DT))
