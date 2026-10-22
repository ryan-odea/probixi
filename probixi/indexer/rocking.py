from __future__ import annotations

import math

import torch
from torch import Tensor


def rocking_radius(
    qn: Tensor, eta: float, r_size: float, bandwidth: float, wavelength: float
) -> Tensor:
    # Ewald-shell half-width rho(|q|) in A^-1; eta in rad, bandwidth as dlambda / lambda
    return r_size + 0.5 * eta * qn + 0.5 * wavelength * bandwidth * qn * qn


@torch.no_grad()
def estimate_mosaicity(
    q_pred: Tensor,
    wavelength: float,
    bandwidth: float,
    prior_eta: float,
    eta_min: float,
    eta_max: float,
    r_size_floor: float,
    min_peaks: int = 8,
) -> tuple[float, float]:
    # Fits eta (rad) to radial Ewald offsets of indexed peaks; r_size stays at its floor
    if int(q_pred.shape[0]) < min_peaks:
        return prior_eta, r_size_floor
    qn = torch.linalg.vector_norm(q_pred, dim=-1)
    # eps = |lambda q + z| - 1; r_s is its radial offset in A^-1 less the bandwidth term
    S = q_pred * wavelength
    Sz = S[:, 2] + 1.0
    eps = torch.sqrt(S[:, 0] ** 2 + S[:, 1] ** 2 + Sz * Sz) - 1.0
    r_s = (eps.abs() / wavelength) - 0.5 * wavelength * bandwidth * qn * qn
    r_s = r_s.clamp_min(0.0)
    denom = float((qn * qn).sum())
    if denom <= 1e-12:
        return prior_eta, r_size_floor
    # E|X| = sigma sqrt(2 / pi): the through-origin slope times sqrt(pi / 2) is eta / 2
    slope = float((qn * r_s).sum() / denom)
    eta = 2.0 * slope * math.sqrt(math.pi / 2.0)
    if not math.isfinite(eta):
        eta = prior_eta
    eta = min(max(eta, eta_min), eta_max)
    return eta, r_size_floor
