from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .forward import detector_to_q, q_to_detector

# Given orientation A, a lattice point diffracts when its scattered
# vector S = lambda*q + zhat is ~unit (Ewald condition). Mosaicity/bandwidth
# thicken the sphere into a shell, modeled by a tolerance on |S| - 1


@dataclass
class PredictedReflections:
    """Reflections predicted to diffract on one frame.

    Attributes
    ----------
    hkl : Tensor
        (M, 3) integer Miller indices.
    positions : Tensor
        (M, 2) predicted detector ``(row, col)`` positions.
    q : Tensor
        (M, 3) predicted reciprocal-space vectors (A^-1), ``q = A @ hkl``.
    excitation_error : Tensor
        (M,) ``|S| - 1``, the dimensionless Ewald-sphere offset.
    resolution : Tensor
        (M,) ``|q|`` in A^-1 (1/d).
    """

    hkl: Tensor
    positions: Tensor
    q: Tensor
    excitation_error: Tensor
    resolution: Tensor

    def __len__(self) -> int:
        return int(self.hkl.shape[0])


def _enumerate_hkls_within(
    A: Tensor, q_max: float, dtype: torch.dtype, device
) -> Tensor:
    # All integer (h,k,l) != 0 that could fall inside |q| <= q_max; per-axis
    # bound h_i <= q_max / |A[:, i]| makes the box safe.
    col_norms = torch.linalg.vector_norm(A, dim=0).clamp_min(1e-12)
    maxima = torch.ceil(q_max / col_norms).to(torch.long).tolist()

    def _rng(m: int) -> Tensor:
        m = max(int(m), 1)
        return torch.arange(-m, m + 1, device=device, dtype=dtype)

    H, K, L = torch.meshgrid(
        _rng(maxima[0]), _rng(maxima[1]), _rng(maxima[2]), indexing="ij"
    )
    hkl = torch.stack([H, K, L], dim=-1).reshape(-1, 3)
    return hkl[~((hkl == 0).all(dim=-1))]


@torch.no_grad()
def predict_reflections(
    A: Tensor,
    geometry: dict,
    q_max: float,
    partiality_threshold: float = 0.0025,
    frame_shape: tuple[int, int] | None = None,
) -> PredictedReflections:
    """Predict the reflections that diffract on one frame for orientation ``A``.

    Parameters
    ----------
    A : Tensor
        (3, 3) reciprocal-to-lab matrix; ``q = A @ hkl`` in the lab frame.
    geometry : dict
        Detector geometry (``beam_center``, ``clen``, ``pixel_size``, ``wavelength``).
    q_max : float
        Resolution limit (A^-1); reflections with ``|q| > q_max`` are dropped.
    partiality_threshold : float, optional
        Max ``|S| - 1`` (excitation error) for a reflection to count as
        diffracting. Wider -> more (more partial) reflections.
    frame_shape : tuple of int, optional
        (rows, cols); predictions off the detector are dropped when given.

    Returns
    -------
    PredictedReflections
        Everything aligned row-for-row.
    """
    if A.shape != (3, 3):
        raise ValueError("A must be (3, 3)")
    device, dtype = A.device, A.dtype
    wavelength = float(geometry["wavelength"])

    hkl = _enumerate_hkls_within(A, q_max, dtype, device)
    q = hkl @ A.transpose(-1, -2)  # (M, 3) = (A @ hkl^T)^T
    qn = torch.linalg.vector_norm(q, dim=-1)

    # S = lambda*q + zhat; diffracts when |S| ~ 1. Keep the forward-scattering
    # hemisphere (Sz > 0) so the ray reaches the detector at z = clen.
    S = q * wavelength
    Sz = S[:, 2] + 1.0
    S_norm = torch.sqrt(S[:, 0] ** 2 + S[:, 1] ** 2 + Sz * Sz)
    eps = S_norm - 1.0

    keep = (qn <= q_max) & (Sz > 0) & (eps.abs() < partiality_threshold)
    hkl, q, qn, eps = hkl[keep], q[keep], qn[keep], eps[keep]
    if hkl.shape[0] == 0:
        empty = q.new_empty(0)
        return PredictedReflections(
            hkl.to(torch.long), q.new_empty(0, 2), q, empty, empty
        )

    positions = q_to_detector(q, geometry, dtype=dtype)
    if frame_shape is not None:
        rows, cols = positions[:, 0], positions[:, 1]
        on = (
            (rows >= 0)
            & (rows <= frame_shape[0] - 1)
            & (cols >= 0)
            & (cols <= frame_shape[1] - 1)
        )
        hkl, positions, q, qn, eps = (
            hkl[on],
            positions[on],
            q[on],
            qn[on],
            eps[on],
        )

    return PredictedReflections(
        hkl=hkl.to(torch.long),
        positions=positions,
        q=q,
        excitation_error=eps,
        resolution=qn,
    )


def detector_q_max(geometry: dict, frame_shape: tuple[int, int]) -> float:
    # Highest |q| (A^-1) reachable on the detector, from its four corners.
    rows, cols = frame_shape
    corners = torch.tensor(
        [[0.0, 0.0], [0.0, cols - 1], [rows - 1, 0.0], [rows - 1, cols - 1]],
        dtype=torch.float64,
    )
    q = detector_to_q(corners, geometry, dtype=torch.float64)
    return float(torch.linalg.vector_norm(q, dim=-1).max())
