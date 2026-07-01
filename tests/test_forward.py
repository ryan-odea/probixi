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


# --- multi-panel (CSPAD-style) physical geometry -----------------------------


def _q_from_physical(x_pix, y_pix, g):
    # reference q for a pixel at physical lab (x, y) px, z = clen
    pix, clen, lam = (
        float(g["pixel_size"]),
        float(g["clen"]),
        float(g["wavelength"]),
    )
    xa, ya, za = x_pix * pix * 1e10, y_pix * pix * 1e10, clen * 1e10
    r = math.sqrt(xa * xa + ya * ya + za * za)
    return [xa / r / lam, ya / r / lam, (za / r - 1.0) / lam]


def _rotated_two_panel():
    # panel B sits right of A in data space but is physically rotated 90 deg (CSPAD-style)
    panels = {
        "A": {
            "min_ss": 0,
            "max_ss": 9,
            "min_fs": 0,
            "max_fs": 9,
            "corner_x": -20.0,
            "corner_y": -5.0,
            "fs": "+1.0x +0.0y",
            "ss": "+0.0x +1.0y",
        },
        "B": {
            "min_ss": 0,
            "max_ss": 9,
            "min_fs": 10,
            "max_fs": 19,
            "corner_x": 5.0,
            "corner_y": -5.0,
            "fs": "+0.0x +1.0y",
            "ss": "-1.0x +0.0y",
        },
    }
    # beam_center is a fallback only; panel pixels use corner + fs/ss
    return {
        "beam_center": (5.0, 12.5),
        "clen": 0.1,
        "pixel_size": 100e-6,
        "wavelength": 1.0,
        "panels": panels,
    }


def _flat_two_panel():
    # two identity panels tiling one plane -> panel model collapses to flat
    bc_row, bc_col = 5.0, 9.5
    panels = {
        "A": {
            "min_ss": 0,
            "max_ss": 9,
            "min_fs": 0,
            "max_fs": 9,
            "corner_x": 0 - bc_col,
            "corner_y": 0 - bc_row,
            "fs": "+1.0x +0.0y",
            "ss": "+0.0x +1.0y",
        },
        "B": {
            "min_ss": 0,
            "max_ss": 9,
            "min_fs": 10,
            "max_fs": 19,
            "corner_x": 10 - bc_col,
            "corner_y": 0 - bc_row,
            "fs": "+1.0x +0.0y",
            "ss": "+0.0x +1.0y",
        },
    }
    return {
        "beam_center": (bc_row, bc_col),
        "clen": 0.1,
        "pixel_size": 100e-6,
        "wavelength": 1.0,
        "panels": panels,
    }


def test_panel_model_uses_physical_position_on_rotated_panel():
    g = _rotated_two_panel()
    # (row=3, col=15) on panel B -> physical (2, 0)
    q = detector_to_q(torch.tensor([[3.0, 15.0]], dtype=DT), g)
    expected = torch.tensor(_q_from_physical(2.0, 0.0, g), dtype=DT)
    assert torch.allclose(q[0], expected, atol=1e-9)
    # the flat mapping would give a different (wrong) q
    flat = torch.tensor(_q_from_physical(15.0 - 12.5, 3.0 - 5.0, g), dtype=DT)
    assert not torch.allclose(q[0], flat, atol=1e-6)


def test_panel_model_round_trip_each_panel():
    g = _rotated_two_panel()
    pos = torch.tensor([[2.0, 3.0], [7.0, 5.0], [3.0, 15.0], [8.0, 18.0]], dtype=DT)
    back = q_to_detector(detector_to_q(pos, g), g)
    assert torch.allclose(back, pos, atol=1e-6)


def test_panel_model_off_panel_q_is_nan():
    # a q that misses every panel -> NaN (dropped as off-detector)
    g = _rotated_two_panel()
    far = torch.tensor([[0.5, 0.5, 0.0]], dtype=DT)  # large in-plane q -> misses
    back = q_to_detector(far, g)
    assert torch.isnan(back).any()


def test_axis_vector_parses_decimal_and_scientific():
    from probixi.indexer.forward import _parse_axis_vector

    assert _parse_axis_vector("+1.0x +0.0y") == (1.0, 0.0)
    assert _parse_axis_vector("-0.999992x +0.003976y") == (-0.999992, 0.003976)
    # scientific notation must not capture the exponent digits as the coefficient
    x, y = _parse_axis_vector("+1.2e-3x +0.99y")
    assert x == pytest.approx(0.0012) and y == pytest.approx(0.99)
    assert _parse_axis_vector("not a vector") is None


def test_panel_model_identity_panels_reduce_to_flat():
    g = _flat_two_panel()
    bc_row, bc_col = g["beam_center"]
    pos = torch.tensor([[4.0, 2.0], [6.0, 17.0]], dtype=DT)  # one per panel
    q = detector_to_q(pos, g)
    for i, (r0, c0) in enumerate(pos.tolist()):
        expected = torch.tensor(_q_from_physical(c0 - bc_col, r0 - bc_row, g), dtype=DT)
        assert torch.allclose(q[i], expected, atol=1e-9)
