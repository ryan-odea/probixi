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
    # Reciprocal basis B = M^-T from cell params: q = B @ hkl (crystallographer
    # convention, no 2*pi). M columns are the direct lattice vectors a, b, c.
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
    # Recover cell edges/angles from reciprocal basis B (inverse of cell_to_B).
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
