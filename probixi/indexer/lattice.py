from __future__ import annotations

import math

import torch
from torch import Tensor

from ..io.cell import CellParams


def cell_to_B(
    cell: CellParams,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Reciprocal basis ``B`` (columns map hkl -> q) from unit-cell parameters.

    Parameters
    ----------
    cell : CellParams
        Unit cell (edges and angles in radians).
    device : optional
        Torch device for the result.
    dtype : torch.dtype, optional
        Result dtype (default ``torch.float64``).

    Returns
    -------
    Tensor
        (3, 3) matrix ``B = M^{-T}`` so that ``q = B @ hkl`` (crystallographer's
        convention, no 2*pi factor).
    """
    a, b, c = cell.a, cell.b, cell.c
    ca, cb, cg = math.cos(cell.alpha), math.cos(cell.beta), math.cos(cell.gamma)
    sg = math.sin(cell.gamma)
    if abs(sg) < 1e-12:
        raise ValueError("gamma cannot be 0 or pi")

    a_vec = [a, 0.0, 0.0]
    b_vec = [b * cg, b * sg, 0.0]
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq <= 0.0:
        raise ValueError("invalid cell parameters (cz^2 <= 0)")
    c_vec = [cx, cy, math.sqrt(cz_sq)]

    M = torch.tensor([a_vec, b_vec, c_vec], dtype=dtype, device=device).T
    return torch.linalg.inv(M).transpose(-1, -2)


def B_to_cell(B: Tensor) -> CellParams:
    """Recover cell edges and angles from a reciprocal basis ``B`` (inverse of cell_to_B)."""
    if B.shape[-2:] != (3, 3):
        raise ValueError("B must be (3, 3)")
    M = torch.linalg.inv(B.transpose(-1, -2))
    av, bv, cv = M[:, 0], M[:, 1], M[:, 2]
    a = float(torch.linalg.vector_norm(av))
    b = float(torch.linalg.vector_norm(bv))
    c = float(torch.linalg.vector_norm(cv))
    alpha = math.acos(float(torch.dot(bv, cv)) / (b * c))
    beta = math.acos(float(torch.dot(av, cv)) / (a * c))
    gamma = math.acos(float(torch.dot(av, bv)) / (a * b))
    return CellParams(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)


def reduce_cell(B: Tensor, max_coef: int = 3) -> Tensor:
    if B.shape[-2:] != (3, 3):
        raise ValueError("B must be (3, 3)")
    device, dtype = B.device, B.dtype
    M = torch.linalg.inv(B.transpose(-1, -2))

    coefs = torch.tensor(
        [
            [i, j, k]
            for i in range(-max_coef, max_coef + 1)
            for j in range(-max_coef, max_coef + 1)
            for k in range(-max_coef, max_coef + 1)
            if not (i == 0 and j == 0 and k == 0)
        ],
        dtype=dtype,
        device=device,
    )
    vectors = coefs @ M.transpose(-1, -2)
    lengths = torch.linalg.vector_norm(vectors, dim=-1)
    order = torch.argsort(lengths)

    chosen: list[Tensor] = []
    for idx in order.tolist():
        v = vectors[idx]
        if not chosen:
            chosen.append(v)
            continue
        if len(chosen) == 1:
            denom = (
                torch.linalg.vector_norm(v) * torch.linalg.vector_norm(chosen[0])
            ).clamp_min(1e-12)
            if float((v @ chosen[0] / denom).abs()) < 0.99:
                chosen.append(v)
            continue
        if len(chosen) == 2:
            cross = torch.linalg.cross(chosen[0], chosen[1])
            denom = (
                torch.linalg.vector_norm(cross) * torch.linalg.vector_norm(v)
            ).clamp_min(1e-12)
            if float(((cross @ v).abs() / denom)) > 0.05:
                chosen.append(v)
                break
    if len(chosen) < 3:
        return B

    M_red = torch.stack(chosen, dim=1)
    if float(torch.linalg.det(M_red)) * float(torch.linalg.det(M)) < 0:
        M_red[:, 2] = -M_red[:, 2]
    return torch.linalg.inv(M_red.transpose(-1, -2))


def decompose_A(A: Tensor) -> tuple[Tensor, Tensor, CellParams]:
    """Factor ``A = U @ B`` via QR, with ``B`` near-Niggli-reduced.

    Parameters
    ----------
    A : Tensor
        (3, 3) reciprocal-to-lab matrix, ``q = A @ hkl``.

    Returns
    -------
    tuple of (Tensor, Tensor, CellParams)
        ``(U, B, cell)``: orientation rotation, cell-only reciprocal basis, and
        the unit cell recovered from ``B``.
    """
    if A.shape[-2:] != (3, 3):
        raise ValueError("A must be (3, 3)")
    B_reduced = reduce_cell(A)
    U_candidate = A @ torch.linalg.inv(B_reduced)
    Q, _ = torch.linalg.qr(U_candidate)
    if float(torch.linalg.det(Q)) < 0:
        Q = -Q
        B_reduced = -B_reduced
    return Q, B_reduced, B_to_cell(B_reduced)
