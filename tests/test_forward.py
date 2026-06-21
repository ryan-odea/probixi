from __future__ import annotations

import math

import pytest
import torch

from probixi.indexer.forward import detector_to_q, q_to_detector

DT = torch.float64


def proper_rotation(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=DT))
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def sample_positions(geometry_dict) -> torch.Tensor:
    # pixels offset from the beam center (and one on it)
    row, col = geometry_dict["beam_center"]
    offsets = [(0, 0), (200, 0), (0, 300), (-150, 250), (600, -400)]
    return torch.tensor([[row + dr, col + dc] for dr, dc in offsets], dtype=DT)


def test_beam_center_maps_to_zero_q(geometry_dict):
    row, col = geometry_dict["beam_center"]
    q = detector_to_q(torch.tensor([[row, col]], dtype=DT), geometry_dict)
    assert torch.allclose(q, torch.zeros_like(q), atol=1e-12)


def test_lift_then_project_is_identity(geometry_dict):
    pos = sample_positions(geometry_dict)
    q = detector_to_q(pos, geometry_dict)
    back = q_to_detector(q, geometry_dict)
    assert torch.allclose(back, pos, atol=1e-6)


def test_q_magnitude_matches_bragg_law(geometry_dict):
    pix_m = float(geometry_dict["pixel_size"])
    clen_m = float(geometry_dict["clen"])
    lam_A = float(geometry_dict["wavelength"])
    bc_row, bc_col = geometry_dict["beam_center"]

    pos = sample_positions(geometry_dict)
    q_norm = torch.linalg.vector_norm(detector_to_q(pos, geometry_dict), dim=-1)

    for (row, col), got in zip(pos.tolist(), q_norm.tolist()):
        radius_m = math.hypot((row - bc_row) * pix_m, (col - bc_col) * pix_m)
        two_theta = math.atan2(radius_m, clen_m)
        expected = 2.0 * math.sin(0.5 * two_theta) / lam_A  # |q| = 2 sin(theta)/lambda
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_frame_rotation_rotates_q(geometry_dict):
    R = proper_rotation(1)
    pos = sample_positions(geometry_dict)
    q_lab = detector_to_q(pos, geometry_dict)
    q_rot = detector_to_q(pos, geometry_dict, frame_rotation=R)
    assert torch.allclose(q_rot, q_lab @ R, atol=1e-12)


def test_frame_rotation_is_invertible_through_detector(geometry_dict):
    R = proper_rotation(2)
    pos = sample_positions(geometry_dict)
    q = detector_to_q(pos, geometry_dict, frame_rotation=R)
    back = q_to_detector(q, geometry_dict, frame_rotation=R)
    assert torch.allclose(back, pos, atol=1e-6)


def test_detector_to_q_rejects_wrong_shape(geometry_dict):
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        detector_to_q(torch.zeros((4, 3), dtype=DT), geometry_dict)


def test_q_to_detector_rejects_wrong_shape(geometry_dict):
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        q_to_detector(torch.zeros((4, 2), dtype=DT), geometry_dict)
