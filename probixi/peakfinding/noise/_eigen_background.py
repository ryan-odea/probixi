from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


@torch.no_grad()
def fit_eigen_background(
    noise_model,
    seed_frames: Iterable[Tensor],
    n_modes: int = 4,
    robust_k: float = 4.0,
    iters: int = 2,
) -> tuple[Tensor, Tensor]:
    """Fit ``n_modes`` low-rank background eigen-images on the seed frames and
    store them on ``noise_model.eigen_modes`` (shape ``(n_modes, H, W)``, zero
    outside the valid mask, orthonormal over valid pixels).

    Args:
        noise_model: a warmed ``NoiseModel`` (provides ``mu``, ``sigma``, mask).
        seed_frames: calibration frames (consumed once).
        n_modes: number of background modes to keep.
        robust_k: residuals beyond ``robust_k * sigma`` are clipped before the
            SVD so Bragg peaks do not leak into the modes.
        iters: trimmed-PCA passes; each re-clips against the current fit.

    Returns:
        ``(modes, explained)`` -- the ``(n_modes, H, W)`` modes and their
        eigenvalues (variance explained, descending).
    """
    frames = [f for f in seed_frames]
    if not frames:
        raise ValueError("seed_frames is empty")

    mu = noise_model.pixel.mean()
    dtype = mu.dtype
    device = mu.device
    sigma = noise_model.pixel.var().clamp_min(1e-12).sqrt()
    m = noise_model.valid_mask.to(dtype)
    cap = robust_k * sigma

    # raw per-frame residuals on valid pixels
    raw = torch.stack([(f.to(device=device, dtype=dtype) - mu) for f in frames])
    R = (torch.clamp(raw, min=-cap, max=cap) * m).flatten(1)  # (F, P)
    R = R - R.mean(dim=0, keepdim=True)  # center so modes capture variation

    modes = torch.zeros(n_modes, R.shape[1], dtype=dtype, device=device)
    explained = torch.zeros(n_modes, dtype=dtype, device=device)
    for it in range(max(1, iters)):
        # SVD via the small (F x F) Gram matrix: modes are R^T v / sqrt(lambda)
        G = R @ R.t()
        evals, evecs = torch.linalg.eigh(G)
        order = torch.argsort(evals, descending=True)[:n_modes]
        U = (R.t() @ evecs[:, order]) / evals[order].clamp_min(1e-12).sqrt()
        U = U.t()  # (n_modes, P)
        norms = U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        U = U / norms
        modes, explained = U, evals[order]
        if it + 1 >= iters:
            break
        coeffs = (raw.flatten(1) * m.flatten()) @ U.t()  # (F, n_modes)
        fit = coeffs @ U  # (F, P)
        resid = raw.flatten(1) - fit
        clipped = fit + torch.clamp(resid, min=-cap.flatten(), max=cap.flatten())
        R = clipped * m.flatten()
        R = R - R.mean(dim=0, keepdim=True)

    modes = (modes.reshape(n_modes, *mu.shape) * m).contiguous()
    noise_model.eigen_modes = modes
    return modes, explained
