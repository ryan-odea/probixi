from __future__ import annotations

import math

import torch
from torch import Tensor

from ..io.cell import CellParams


def cell_to_B(
    cell: CellParams,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    # B = M^-T with the direct axes a, b, c as columns of M; q = B @ hkl, no 2 pi
    a, b, c = cell.a, cell.b, cell.c
    ca, cb, cg = math.cos(cell.alpha), math.cos(cell.beta), math.cos(cell.gamma)
    sg = math.sin(cell.gamma)
    if abs(sg) < 1e-12:
        raise ValueError("gamma cannot be 0 or pi")

    # Standard setting: a along x, b in the xy plane
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
    # Inverse of cell_to_B: edges in A and angles in rad from B
    if B.shape[-2:] != (3, 3):
        raise ValueError("B must be (3, 3)")
    M = torch.linalg.inv(B.transpose(-1, -2))
    av, bv, cv = M[:, 0], M[:, 1], M[:, 2]
    a, b, c, bc, ac, ab = torch.stack(
        [
            torch.linalg.vector_norm(av),
            torch.linalg.vector_norm(bv),
            torch.linalg.vector_norm(cv),
            torch.dot(bv, cv),
            torch.dot(av, cv),
            torch.dot(av, bv),
        ]
    ).tolist()
    alpha = math.acos(bc / (b * c))
    beta = math.acos(ac / (a * c))
    gamma = math.acos(ab / (a * b))
    return CellParams(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
