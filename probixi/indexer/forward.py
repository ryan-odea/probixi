from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..io.geometry import parse_axis_vector as _parse_axis_vector


def _panel_bases(
    geometry: dict, device: Optional[torch.device], dtype: torch.dtype
) -> Optional[Tensor]:
    # (P, 10) tensor [min_ss, max_ss, min_fs, max_fs, corner_x, corner_y,
    # fs_x, fs_y, ss_x, ss_y] per panel
    panels = geometry.get("panels") or {}
    rows: list[list[float]] = []
    for p in panels.values():
        fs = _parse_axis_vector(p.get("fs"))
        ss = _parse_axis_vector(p.get("ss"))
        if fs is None or ss is None:
            return None
        try:
            row = [
                float(p["min_ss"]),
                float(p["max_ss"]),
                float(p["min_fs"]),
                float(p["max_fs"]),
                float(p["corner_x"]),
                float(p["corner_y"]),
                fs[0],
                fs[1],
                ss[0],
                ss[1],
            ]
        except (KeyError, TypeError, ValueError):
            return None
        if abs(fs[0] * ss[1] - ss[0] * fs[1]) < 1e-9:  # degenerate basis
            return None
        rows.append(row)
    if not rows:
        return None
    return torch.tensor(rows, device=device, dtype=dtype)


def _panel_model_bases(
    geometry: dict, device: Optional[torch.device], dtype: torch.dtype
) -> Optional[Tensor]:
    if len(geometry.get("panels") or {}) < 2:
        return None
    return _panel_bases(geometry, device, dtype)


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


def _lab_xy_pixels(
    positions: Tensor, geometry: dict, bases: Optional[Tensor]
) -> Tensor:
    bc_row = float(geometry["beam_center"][0])
    bc_col = float(geometry["beam_center"][1])
    rows, cols = positions[:, 0], positions[:, 1]
    x = cols - bc_col
    y = rows - bc_row
    if bases is None:
        return torch.stack([x, y], dim=-1)
    x = x.clone()
    y = y.clone()
    for b in bases:
        min_ss, max_ss, min_fs, max_fs = b[0], b[1], b[2], b[3]
        cx, cy, fsx, fsy, ssx, ssy = b[4], b[5], b[6], b[7], b[8], b[9]
        m = (rows >= min_ss) & (rows <= max_ss) & (cols >= min_fs) & (cols <= max_fs)
        if not torch.any(m):
            continue
        fs_j = cols[m] - min_fs
        ss_i = rows[m] - min_ss
        x[m] = cx + fs_j * fsx + ss_i * ssx
        y[m] = cy + fs_j * fsy + ss_i * ssy
    return torch.stack([x, y], dim=-1)


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
        Geometry dict (``beam_center``, ``clen``, ``pixel_size``, ``wavelength``,
        and``panels`` with ``corner_x/y`` and ``fs``/``ss`` for multipanel).
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
    bases = _panel_model_bases(geometry, device, dtype)

    xy_pix = _lab_xy_pixels(pos, geometry, bases)
    x_lab = xy_pix[:, 0] * g["pix_A"]
    y_lab = xy_pix[:, 1] * g["pix_A"]
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


def _project_to_panels(
    x_pix: Tensor, y_pix: Tensor, valid: Tensor, bases: Tensor
) -> Tensor:
    n = x_pix.shape[0]
    row = torch.full((n,), float("nan"), device=x_pix.device, dtype=x_pix.dtype)
    col = torch.full((n,), float("nan"), device=x_pix.device, dtype=x_pix.dtype)
    for b in bases:
        min_ss, _, min_fs, _ = b[0], b[1], b[2], b[3]
        cx, cy, fsx, fsy, ssx, ssy = b[4], b[5], b[6], b[7], b[8], b[9]
        fs_len = b[3] - b[2]
        ss_len = b[1] - b[0]
        det = fsx * ssy - ssx * fsy
        rx = x_pix - cx
        ry = y_pix - cy
        fs_j = (ssy * rx - ssx * ry) / det
        ss_i = (-fsy * rx + fsx * ry) / det
        on = (
            valid
            & (fs_j >= 0)
            & (fs_j <= fs_len)
            & (ss_i >= 0)
            & (ss_i <= ss_len)
            & torch.isnan(row)
        )
        row[on] = min_ss + ss_i[on]
        col[on] = min_fs + fs_j[on]
    return torch.stack([row, col], dim=-1)


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
    x_pix = (Sx * scale) / g["pix_A"]
    y_pix = (Sy * scale) / g["pix_A"]

    bases = _panel_model_bases(geometry, device, dtype)
    if bases is None:
        col = x_pix + g["bc_col"]
        row = y_pix + g["bc_row"]
        return torch.stack([row, col], dim=-1)
    return _project_to_panels(x_pix, y_pix, Sz > 0, bases)
