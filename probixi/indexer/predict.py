from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .forward import detector_to_q, q_to_detector
from .rocking import rocking_radius

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


def _centering_mask(hkl: Tensor, centering: str | None) -> Tensor:
    n = hkl.shape[0]
    keep_all = torch.ones(n, dtype=torch.bool, device=hkl.device)
    if not centering:
        return keep_all
    c = centering.strip().upper()[:1]
    h = hkl[:, 0].round().to(torch.long)
    k = hkl[:, 1].round().to(torch.long)
    l = hkl[:, 2].round().to(torch.long)
    if c == "A":
        return (k + l) % 2 == 0
    if c == "B":
        return (h + l) % 2 == 0
    if c == "C":
        return (h + k) % 2 == 0
    if c == "I":
        return (h + k + l) % 2 == 0
    if c == "F":
        return ((h + k) % 2 == 0) & ((h + l) % 2 == 0) & ((k + l) % 2 == 0)
    if c == "R":  # rhombohedral on hexagonal axes, obverse setting
        return (-h + k + l) % 3 == 0
    if c == "H":  # hexagonal H-centered (triple) cell
        return (h - k) % 3 == 0
    return keep_all  # "P" or unknown symbol: no absences


def _enumerate_hkls_within(
    A: Tensor, q_max: float, dtype: torch.dtype, device
) -> Tensor:
    row_norms = torch.linalg.vector_norm(torch.linalg.inv(A), dim=1)
    maxima = torch.ceil(q_max * row_norms).to(torch.long).tolist()

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
    *,
    eta: float | None = None,
    r_size: float = 0.0,
    bandwidth: float = 0.0,
    predict_sigma: float = 1.5,
    partiality_threshold: float = 0.0025,
    centering: str | None = None,
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
    eta : float, optional
        Mosaic angular spread (rad). When given, the excitation-error tolerance
        is the physical rocking shell ``|eps| < wavelength * predict_sigma *
        R(|q|)`` with ``R`` from :func:`rocking_radius` (so the shell grows with
        resolution instead of being flat). When ``None`` the fixed
        ``partiality_threshold`` is used.
    r_size, bandwidth : float, optional
        Domain-size (constant, A^-1) and bandwidth (``dlambda/lambda``) terms of
        the rocking model; used only when ``eta`` is given.
    predict_sigma : float, optional
        How many rocking half-widths to integrate out to.
    partiality_threshold : float, optional
        Max ``|S| - 1`` (excitation error). With ``eta`` it is an absolute floor
        on the per-reflection tolerance
    centering : str, optional
        Bravais centering symbol (``P``/``A``/``B``/``C``/``I``/``F``/``R``).
        Reflections forbidden by the centering condition are dropped so absent
        positions aren't predicted. ``None`` or ``P`` applies no condition.
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
    if centering:
        hkl = hkl[_centering_mask(hkl, centering)]
    q = hkl @ A.transpose(-1, -2)  # (M, 3) = (A @ hkl^T)^T
    qn = torch.linalg.vector_norm(q, dim=-1)

    # S = lambda*q + zhat; diffracts when |S| ~ 1. Keep the forward-scattering
    # hemisphere (Sz > 0) so the ray reaches the detector at z = clen.
    S = q * wavelength
    Sz = S[:, 2] + 1.0
    S_norm = torch.sqrt(S[:, 0] ** 2 + S[:, 1] ** 2 + Sz * Sz)
    eps = S_norm - 1.0

    if eta is not None:
        tol = (
            wavelength
            * predict_sigma
            * rocking_radius(qn, eta, r_size, bandwidth, wavelength)
        )
        tol = tol.clamp_min(partiality_threshold)
        keep = (qn <= q_max) & (Sz > 0) & (eps.abs() < tol)
    else:
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
