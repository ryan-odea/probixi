from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..io.cell import CellParams
from ..kernels import select
from .lattice import B_to_cell


@dataclass
class RefineResult:
    # Refinement output over K candidate orientations and N peaks: refined A,
    # per-candidate rmsd / inlier count / soft score, indexed mask, hkl, loss history.
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


_EDGE = {"a": 0, "b": 1, "c": 2}
_ANGLE_OF_AXIS = {"a": 3, "b": 4, "c": 5}
_TIED_EDGE_TYPES = ("tetragonal", "hexagonal", "trigonal")


def _cell_dofs(cell: CellParams, dtype, device) -> tuple[Tensor, Tensor]:
    # (6, m) map T and base p0 so p6 = p0 + T @ r
    lt = (cell.lattice_type or "triclinic").lower()
    u = (cell.unique_axis or ("b" if lt == "monoclinic" else "c")).lower()
    cols: list[list[int]] = []
    if lt in ("cubic", "rhombohedral"):
        cols.append([0, 1, 2])
    elif lt in _TIED_EDGE_TYPES:
        ua = _EDGE.get(u, 2)
        cols.append([i for i in range(3) if i != ua])
        cols.append([ua])
    else:
        cols += [[0], [1], [2]]
    if lt == "monoclinic":
        cols.append([_ANGLE_OF_AXIS.get(u, 4)])
    elif lt == "rhombohedral":
        cols.append([3, 4, 5])
    elif lt not in ("orthorhombic", "cubic", *_TIED_EDGE_TYPES):
        cols += [[3], [4], [5]]
    T = torch.zeros(6, len(cols), dtype=dtype, device=device)
    for j, rows in enumerate(cols):
        T[rows, j] = 1.0
    p0 = torch.tensor(
        [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma],
        dtype=dtype,
        device=device,
    )
    return T, p0


def _B_from_params(p: Tensor) -> Tensor:
    a, b, c, al, be, ga = p
    ca, cb, cg, sg = torch.cos(al), torch.cos(be), torch.cos(ga), torch.sin(ga)
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz = torch.sqrt((c * c - cx * cx - cy * cy).clamp_min(1e-12))
    zero = torch.zeros_like(a)
    M = torch.stack(
        [
            torch.stack([a, b * cg, cx]),
            torch.stack([zero, b * sg, cy]),
            torch.stack([zero, zero, cz]),
        ]
    )
    return torch.linalg.inv(M).transpose(-1, -2)


def _assign_hkls(
    A: Tensor, q: Tensor, q_tolerance: float, khat: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    hkl = torch.round(q @ torch.linalg.inv(A).transpose(-1, -2))
    d = hkl @ A.transpose(-1, -2) - q
    if khat is not None:
        d = d - (d * khat).sum(-1, keepdim=True) * khat
    indexed = d.norm(dim=-1) < q_tolerance
    return hkl, indexed


def refine_cell(
    A: Tensor,
    q: Tensor,
    hkl: Tensor,
    indexed: Tensor,
    template: CellParams,
    q_tolerance: float,
    iters: int = 8,
    rounds: int = 2,
    edge_tolerance: float = 0.05,
    angle_tolerance_rad: float = math.radians(3.0),
    wavelength: Optional[float] = None,
) -> Optional[tuple[Tensor, Tensor, Tensor, float, bool]]:
    dtype, out_device = A.dtype, A.device
    device = torch.device("cpu")
    wd = torch.float64
    try:
        c0 = B_to_cell(A)
    except Exception:
        return None
    A, q, hkl, indexed = (t.to(device) for t in (A, q, hkl, indexed))
    cell0 = CellParams(
        c0.a, c0.b, c0.c, c0.alpha, c0.beta, c0.gamma,
        lattice_type=template.lattice_type,
        unique_axis=template.unique_axis,
        centering=template.centering,
    )
    T, p0 = _cell_dofs(cell0, wd, device)
    U0 = A.to(wd) @ torch.linalg.inv(_B_from_params(p0))
    m = 3 + T.shape[1]
    theta = torch.zeros(m, dtype=wd, device=device)
    q_d = q.to(wd)
    hkl_d = hkl.to(wd)
    indexed = indexed.clone()
    eye = torch.eye(m, dtype=wd, device=device)

    def build(th: Tensor) -> Tensor:
        return _axis_angle_to_rotation(th[:3]) @ U0 @ _B_from_params(p0 + T @ th[3:])

    prior_width = torch.tensor(
        [edge_tolerance * c0.a, edge_tolerance * c0.b, edge_tolerance * c0.c]
        + [angle_tolerance_rad] * 3,
        dtype=wd,
        device=device,
    ) / 3.0

    khat = None
    if wavelength is not None and wavelength > 0:
        k_out = q_d + torch.tensor([0.0, 0.0, 1.0 / wavelength], dtype=wd, device=device)
        khat = k_out / k_out.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def split(d: Tensor, k: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        if k is None:
            return d, d.new_zeros(d.shape[0])
        par = (d * k).sum(-1)
        return d - par[:, None] * k, par

    d0 = (hkl_d @ A.to(wd).transpose(-1, -2) - q_d)[indexed]
    d0_perp, d0_par = split(d0, khat[indexed] if khat is not None else None)
    s_par = d0_par.square().mean().sqrt().clamp_min(1e-9)
    n_perp = 3.0 if khat is None else 2.0
    s_perp = (d0_perp.square().sum(-1).mean() / n_perp).sqrt().clamp_min(1e-9)
    if khat is not None:
        s_perp = torch.maximum(s_perp, s_par / 100.0)

    def whitened(A_: Tensor, hkl_: Tensor, sel: Tensor) -> Tensor:
        d = (hkl_ @ A_.transpose(-1, -2) - q_d)[sel]
        dp, da = split(d, khat[sel] if khat is not None else None)
        return torch.cat([(dp / s_perp).reshape(-1), da / s_par])

    before = float(whitened(A.to(wd), hkl_d, indexed).square().mean())
    A_new = A.to(wd)
    for _ in range(rounds):
        if int(indexed.sum()) * 3 <= m:
            return None
        sel = indexed.clone()

        def resid(th: Tensor) -> Tensor:
            prior = (T @ th[3:]) / prior_width
            return torch.cat([whitened(build(th), hkl_d, sel), prior])

        lam = 1e-3
        f = resid(theta)
        loss = float(f.square().sum())
        for _ in range(iters):
            J = torch.autograd.functional.jacobian(resid, theta, vectorize=True)
            g = J.transpose(-1, -2) @ f
            Hm = J.transpose(-1, -2) @ J
            damp = lam * torch.diag(torch.diagonal(Hm)) + 1e-18 * eye
            try:
                step = torch.linalg.solve(Hm + damp, -g)
            except Exception:
                return None
            cand = theta + step
            fc = resid(cand)
            lc = float(fc.square().sum())
            if lc < loss and torch.isfinite(cand).all():
                theta, f, loss, lam = cand, fc, lc, lam / 3.0
            else:
                lam *= 10.0
        A_new = build(theta)
        if not torch.isfinite(A_new).all():
            return None
        hkl_d, indexed = _assign_hkls(A_new, q_d, q_tolerance)
    n = int(indexed.sum())
    if n == 0:
        return None
    improved = float(whitened(A_new, hkl_d, indexed).square().mean()) <= before
    sq = (hkl_d @ A_new.transpose(-1, -2) - q_d).square().sum(-1)
    rmsd = float(torch.sqrt(sq[indexed].mean()))
    return (
        A_new.to(dtype=dtype, device=out_device),
        hkl_d.long().to(out_device),
        indexed.to(out_device),
        rmsd,
        improved,
    )


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
    weights_per_frame: Optional[list[Optional[Tensor]]] = None,
) -> list[RefineResult]:
    # Pad (A_init, q_obs) to (F, K_max, ...) / (F, N_max, 3) and run one Adam over
    # (F, K_max, 3) axis-angle perturbations.
    F = len(A_init_per_frame)
    if F == 0:
        return []
    if len(q_obs_per_frame) != F:
        raise ValueError("A_init_per_frame and q_obs_per_frame must align")
    supported = (
        max_iters > 0
        and reassign_every > 0
        and all(0 < len(a) <= 128 and a.shape[1:] == (3, 3) for a in A_init_per_frame)
        and all(0 < len(q) <= 128 and q.shape[1:] == (3,) for q in q_obs_per_frame)
        and all(
            t.is_cuda
            and t.dtype == torch.float32
            and not t.requires_grad
            and t.device == A_init_per_frame[0].device
            for t in [*A_init_per_frame, *q_obs_per_frame]
        )
        and (weights_per_frame is None or len(weights_per_frame) == F)
    )
    kernel = select("refine", supported)
    if kernel is not None:
        return kernel.refine_triton(
            A_init_per_frame,
            q_obs_per_frame,
            q_tolerance=q_tolerance,
            lr=lr,
            max_iters=max_iters,
            reassign_every=reassign_every,
            min_indexed=min_indexed,
            weights_per_frame=weights_per_frame,
        )
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
        w_f = weights_per_frame[f] if weights_per_frame is not None else None
        if use_soft and w_f is not None:
            w_pad[f, :N_f] = w_f.to(dtype=dtype, device=device)
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
