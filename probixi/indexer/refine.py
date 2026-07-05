from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class RefineResult:
    """Refinement output over K candidate orientations and N peaks

    Attributes
    ----------
    A : Tensor
        (K, 3, 3) refined reciprocal-to-lab matrices.
    rmsd : Tensor
        (K,) RMS q-residual per candidate.
    n_indexed : Tensor
        (K,) hard-inlier count per candidate (for reporting).
    soft_score : Tensor
        (K,) confidence-weighted inlier mass; equals ``n_indexed`` when no
        weights are given. Used for ranking.
    indexed : Tensor
        (K, N) bool mask of indexed peaks.
    hkl : Tensor
        (K, N, 3) integer Miller indices per candidate/peak.
    history : Tensor
        (steps,) refinement loss history.
    """

    A: Tensor
    rmsd: Tensor
    n_indexed: Tensor
    soft_score: Tensor
    indexed: Tensor
    hkl: Tensor
    history: Tensor


def _axis_angle_to_rotation(omega: Tensor) -> Tensor:
    # Axis-angle (..., 3) -> rotation (..., 3, 3) via Rodrigues.
    if omega.shape[-1] != 3:
        raise ValueError("omega must have last dim 3")
    batch = omega.shape[:-1]
    dtype, device = omega.dtype, omega.device
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True).clamp_min(1e-24)
    theta = theta_sq.sqrt()
    axis = omega / theta
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    W = torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, dtype=dtype, device=device).expand(*batch, 3, 3)
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    return eye + sin_t * W + (1.0 - cos_t) * (W @ W)


def _assign_hkls_padded(
    A: Tensor,
    q_pad: Tensor,
    cand_mask: Tensor,
    obs_mask: Tensor,
    q_tolerance: float,
) -> tuple[Tensor, Tensor]:
    # Round A^{-1} q to integer hkl; flag peaks whose back-prediction is within tol.
    A_inv = torch.linalg.inv(A)
    hkl_cont = torch.einsum("fkij,fnj->fkni", A_inv, q_pad)
    hkl = torch.round(hkl_cont).long()
    q_pred = torch.einsum("fkij,fknj->fkni", A, hkl.to(A.dtype))
    diff = q_pred - q_pad.unsqueeze(1)
    sq = (diff * diff).sum(dim=-1)
    indexed = sq < (q_tolerance**2)
    indexed = indexed & obs_mask.unsqueeze(1) & cand_mask.unsqueeze(-1)
    return hkl, indexed


def _empty_refine_result(device, dtype) -> RefineResult:
    # RefineResult with zero candidates and zero peaks.
    return RefineResult(
        A=torch.zeros(0, 3, 3, dtype=dtype, device=device),
        rmsd=torch.zeros(0, dtype=dtype, device=device),
        n_indexed=torch.zeros(0, dtype=torch.long, device=device),
        soft_score=torch.zeros(0, dtype=dtype, device=device),
        indexed=torch.zeros(0, 0, dtype=torch.bool, device=device),
        hkl=torch.zeros(0, 0, 3, dtype=torch.long, device=device),
        history=torch.zeros(0, dtype=torch.float32),
    )


def refine_multiframe_known_B(
    A_init_per_frame: list[Tensor],
    q_obs_per_frame: list[Tensor],
    q_tolerance: float = 0.02,
    lr: float = 1e-3,
    max_iters: int = 200,
    reassign_every: int = 10,
    min_indexed: int = 6,
    patience: int = 40,
    rel_tol: float = 1e-3,
    weights_per_frame: Optional[list[Tensor]] = None,
) -> list[RefineResult]:
    # Pad (A_init, q_obs) to (F, K_max, ...) / (F, N_max, 3) and run one Adam over
    # (F, K_max, 3) axis-angle perturbations.
    F = len(A_init_per_frame)
    if F == 0:
        return []
    if len(q_obs_per_frame) != F:
        raise ValueError("A_init_per_frame and q_obs_per_frame must align")
    use_soft = weights_per_frame is not None

    K_per = [A.shape[0] for A in A_init_per_frame]
    N_per = [q.shape[0] for q in q_obs_per_frame]
    K_max, N_max = max(K_per) if K_per else 0, max(N_per) if N_per else 0
    device = A_init_per_frame[0].device
    dtype = A_init_per_frame[0].dtype
    if K_max == 0 or N_max == 0:
        return [_empty_refine_result(device, dtype) for _ in range(F)]

    A_anchor = torch.zeros(F, K_max, 3, 3, dtype=dtype, device=device)
    cand_mask = torch.zeros(F, K_max, dtype=torch.bool, device=device)
    q_pad = torch.zeros(F, N_max, 3, dtype=dtype, device=device)
    obs_mask = torch.zeros(F, N_max, dtype=torch.bool, device=device)
    w_pad = torch.zeros(F, N_max, dtype=dtype, device=device)
    eye = torch.eye(3, dtype=dtype, device=device)
    for f in range(F):
        K_f, N_f = K_per[f], N_per[f]
        A_anchor[f, :K_f] = A_init_per_frame[f]
        if K_f < K_max:
            A_anchor[f, K_f:] = eye
        cand_mask[f, :K_f] = True
        q_pad[f, :N_f] = q_obs_per_frame[f]
        obs_mask[f, :N_f] = True
        if use_soft and weights_per_frame[f] is not None:
            w_pad[f, :N_f] = weights_per_frame[f].to(dtype=dtype, device=device)
        else:
            w_pad[f, :N_f] = 1.0

    omega = torch.zeros(F, K_max, 3, dtype=dtype, device=device, requires_grad=True)
    optim = torch.optim.Adam([omega], lr=lr)

    history: list[float] = []
    with torch.no_grad():
        hkl, indexed = _assign_hkls_padded(
            A_anchor,
            q_pad,
            cand_mask,
            obs_mask,
            q_tolerance=q_tolerance,
        )
    # float view of the current hkl assignment
    hkl_dt = hkl.to(dtype)
    cand_mask_f = cand_mask.to(dtype)

    best_loss = float("inf")
    stale = 0

    for step in range(max_iters):
        reassigned = step % reassign_every == 0 and step > 0
        if reassigned:
            with torch.no_grad():
                R = _axis_angle_to_rotation(omega)
                hkl, indexed = _assign_hkls_padded(
                    R @ A_anchor,
                    q_pad,
                    cand_mask,
                    obs_mask,
                    q_tolerance=q_tolerance,
                )
            hkl_dt = hkl.to(dtype)

        R = _axis_angle_to_rotation(omega)
        A_eff = R @ A_anchor
        q_pred = torch.einsum("fkij,fknj->fkni", A_eff, hkl_dt)
        residual = q_pred - q_pad.unsqueeze(1)
        sq = (residual * residual).sum(dim=-1)
        mask = indexed.to(dtype)
        # weight each inlier's residual by detection confidence
        wmask = mask * w_pad.unsqueeze(1) if use_soft else mask
        per_w = wmask.sum(dim=-1).clamp_min(1.0)
        per_loss = (sq * wmask).sum(dim=-1) / per_w
        degen = (mask.sum(dim=-1) < min_indexed).to(dtype)
        per_loss = (per_loss + degen) * cand_mask_f
        loss = per_loss.sum()
        loss_val = float(loss.detach().item())
        history.append(loss_val)

        if reassigned:
            best_loss, stale = loss_val, 0
        elif patience > 0:
            if loss_val < best_loss * (1.0 - rel_tol):
                best_loss, stale = loss_val, 0
            else:
                stale += 1
                if stale >= patience:
                    break

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    with torch.no_grad():
        R = _axis_angle_to_rotation(omega)
        A_final = R @ A_anchor
        hkl, indexed = _assign_hkls_padded(
            A_final,
            q_pad,
            cand_mask,
            obs_mask,
            q_tolerance=q_tolerance,
        )
        q_pred = torch.einsum("fkij,fknj->fkni", A_final, hkl.to(dtype))
        sq = ((q_pred - q_pad.unsqueeze(1)) ** 2).sum(dim=-1)
        n_indexed = indexed.sum(dim=-1).long()
        mask = indexed.to(dtype)
        rmsd = torch.sqrt((sq * mask).sum(dim=-1) / n_indexed.clamp_min(1).to(dtype))
        # confidence-weighted inlier sum for ranking
        if use_soft:
            soft_score = (mask * w_pad.unsqueeze(1)).sum(dim=-1)
        else:
            soft_score = n_indexed.to(dtype)

    history_tensor = torch.tensor(history, dtype=torch.float32)
    results: list[RefineResult] = []
    for f in range(F):
        K_f, N_f = K_per[f], N_per[f]
        results.append(
            RefineResult(
                A=A_final[f, :K_f].detach(),
                rmsd=rmsd[f, :K_f],
                n_indexed=n_indexed[f, :K_f],
                soft_score=soft_score[f, :K_f],
                indexed=indexed[f, :K_f, :N_f],
                hkl=hkl[f, :K_f, :N_f],
                history=history_tensor,
            )
        )
    return results
