from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _geometry_constants(
    geometry: dict,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> dict:
    bc = geometry["beam_center"]
    return {
        "bc_row": torch.tensor(float(bc[0]), dtype=dtype, device=device),
        "bc_col": torch.tensor(float(bc[1]), dtype=dtype, device=device),
        "clen_A": torch.tensor(
            float(geometry["clen"]) * 1e10, dtype=dtype, device=device
        ),
        "pix_A": torch.tensor(
            float(geometry["pixel_size"]) * 1e10, dtype=dtype, device=device
        ),
        "wavelength_A": torch.tensor(
            float(geometry["wavelength"]), dtype=dtype, device=device
        ),
    }


def detector_to_q(
    positions: Tensor,
    geometry: dict,
    frame_rotation: Optional[Tensor] = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Map detector pixel positions to reciprocal-space q-vectors.

    Elastic (Ewald) construction with the beam along +z: a pixel at lab position
    ``(x, y, clen)`` defines a scattered unit direction ``s_hat`` and
    ``q = (s_hat - s0_hat) / lambda`` with ``s0_hat = (0, 0, 1)``.

    Parameters
    ----------
    positions : Tensor
        (N, 2) detector ``(row, col)`` pixel positions.
    geometry : dict
        Geometry dict (``beam_center``, ``clen``, ``pixel_size``, ``wavelength``).
    frame_rotation : Tensor, optional
        (3, 3) rotation taking q from lab into the crystal frame.
    dtype : torch.dtype, optional
        Working dtype (default ``torch.float64``).

    Returns
    -------
    Tensor
        (N, 3) reciprocal-space q-vectors (A^-1).
    """
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError("positions must be (N, 2)")
    device = positions.device
    g = _geometry_constants(geometry, device=device, dtype=dtype)
    pos = positions.to(device=device, dtype=dtype)
    rows, cols = pos[:, 0], pos[:, 1]
    # pixel offset from beam center translation to lab-frame coordinates \AA
    x_lab = (cols - g["bc_col"]) * g["pix_A"]
    y_lab = (rows - g["bc_row"]) * g["pix_A"]
    z_lab = g["clen_A"].expand_as(x_lab)
    r = torch.sqrt(x_lab * x_lab + y_lab * y_lab + z_lab * z_lab)
    inv_lambda = 1.0 / g["wavelength_A"]
    qx = (x_lab / r) * inv_lambda
    qy = (y_lab / r) * inv_lambda
    qz = (z_lab / r - 1.0) * inv_lambda
    q_lab = torch.stack([qx, qy, qz], dim=-1)
    if frame_rotation is None:
        return q_lab
    R = frame_rotation.to(device=device, dtype=dtype)
    if R.shape != (3, 3):
        raise ValueError("frame_rotation must be (3, 3)")
    return q_lab @ R


def q_to_detector(
    q: Tensor,
    geometry: dict,
    frame_rotation: Optional[Tensor] = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    # S = q*lambda + s0
    # Intersect the ray with z = clen.
    if q.ndim != 2 or q.shape[-1] != 3:
        raise ValueError("q must be (N, 3)")
    device = q.device
    g = _geometry_constants(geometry, device=device, dtype=dtype)
    qv = q.to(device=device, dtype=dtype)
    if frame_rotation is not None:
        R = frame_rotation.to(device=device, dtype=dtype)
        if R.shape != (3, 3):
            raise ValueError("frame_rotation must be (3, 3)")
        qv = qv @ R.transpose(-1, -2)
    inv_lambda = 1.0 / g["wavelength_A"]
    Sx = qv[:, 0] / inv_lambda
    Sy = qv[:, 1] / inv_lambda
    Sz = qv[:, 2] / inv_lambda + 1.0
    scale = g["clen_A"] / Sz.clamp_min(1e-30)
    col = (Sx * scale) / g["pix_A"] + g["bc_col"]
    row = (Sy * scale) / g["pix_A"] + g["bc_row"]
    return torch.stack([row, col], dim=-1)
