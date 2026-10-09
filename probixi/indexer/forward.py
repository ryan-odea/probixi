from __future__ import annotations

from functools import lru_cache
from operator import itemgetter
from typing import Optional

import torch
from torch import Tensor

from ..io.geometry import parse_axis_vector as _parse_axis_vector
from ..kernels import select

_PANEL_FIELDS = itemgetter(
    "min_ss", "max_ss", "min_fs", "max_fs", "corner_x", "corner_y", "fs", "ss"
)


def _panel_bases(
    geometry: dict, device: Optional[torch.device], dtype: torch.dtype
) -> Optional[Tensor]:
    # (P, 10) tensor [min_ss, max_ss, min_fs, max_fs, corner_x, corner_y,
    # fs_x, fs_y, ss_x, ss_y] per panel
    # Panels are coplanar at z = clen: fs/ss keep only their in-plane (x, y)
    # components.
    #
    # NEED TO TEST [TODO] (3D): a tilted-panel detector (nonzero fs/ss z) needs per-panel
    # 3-D placement and ray-plane intersection in q_to_detector.
    panels = geometry.get("panels") or {}
    try:
        return _panel_tensor(tuple(map(_PANEL_FIELDS, panels.values())), device, dtype)
    except KeyError:
        return None


@lru_cache(maxsize=8)
def _panel_tensor(
    specs: tuple, device: Optional[torch.device], dtype: torch.dtype
) -> Optional[Tensor]:
    rows: list[list[float]] = []
    for min_ss, max_ss, min_fs, max_fs, corner_x, corner_y, fs_spec, ss_spec in specs:
        fs = _parse_axis_vector(fs_spec)
        ss = _parse_axis_vector(ss_spec)
        if fs is None or ss is None:
            return None
        try:
            row = [
                float(min_ss),
                float(max_ss),
                float(min_fs),
                float(max_fs),
                float(corner_x),
                float(corner_y),
                fs[0],
                fs[1],
                ss[0],
                ss[1],
            ]
        except (TypeError, ValueError):
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
    # General per-panel affine map when panels carry an fs/ss basis (any count);
    # None if any panel lacks fs/ss, so the caller falls back to the beam-centre map.
    if not (geometry.get("panels") or {}):
        return None
    return _panel_bases(geometry, device, dtype)


def _geometry_constants(
    geometry: dict,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    bc = geometry["beam_center"]
    return _constants(
        float(bc[0]),
        float(bc[1]),
        float(geometry["clen"]),
        float(geometry["pixel_size"]),
        float(geometry["wavelength"]),
        device,
        dtype,
    )


@lru_cache(maxsize=32)
def _constants(
    bc_row: float,
    bc_col: float,
    clen: float,
    pixel_size: float,
    wavelength: float,
    device: Optional[torch.device],
    dtype: torch.dtype,
) -> dict:
    consts = {
        "bc_row": torch.tensor(bc_row, dtype=dtype, device=device),
        "bc_col": torch.tensor(bc_col, dtype=dtype, device=device),
        "clen_A": torch.tensor(clen * 1e10, dtype=dtype, device=device),
        "pix_A": torch.tensor(pixel_size * 1e10, dtype=dtype, device=device),
        "wavelength_A": torch.tensor(wavelength, dtype=dtype, device=device),
    }
    consts["clen_sq"] = consts["clen_A"] * consts["clen_A"]
    consts["inv_lambda"] = 1.0 / consts["wavelength_A"]
    consts["packed"] = torch.stack(
        [consts[k] for k in ("bc_row", "bc_col", "pix_A", "clen_A", "wavelength_A")]
    )
    return consts


def _lab_xy_pixels(
    positions: Tensor, geometry: dict, bases: Optional[Tensor]
) -> Tensor:
    bc_row = float(geometry["beam_center"][0])
    bc_col = float(geometry["beam_center"][1])
    rows, cols = positions[:, 0], positions[:, 1]
    xy = torch.stack([cols - bc_col, rows - bc_row], dim=-1)
    if bases is None:
        return xy
    min_ss, max_ss, min_fs, max_fs = bases[:, 0], bases[:, 1], bases[:, 2], bases[:, 3]
    # each pixel takes the last panel whose ss/fs bounds contain it
    inside = (
        (rows[:, None] >= min_ss)
        & (rows[:, None] <= max_ss)
        & (cols[:, None] >= min_fs)
        & (cols[:, None] <= max_fs)
    )
    pid = torch.arange(bases.shape[0], device=positions.device)
    chosen = torch.where(inside, pid, -1).amax(1)
    on = chosen >= 0
    panel = bases[chosen.clamp_min(0)]
    fs_j = cols - panel[:, 2]
    ss_i = rows - panel[:, 0]
    placed = (
        panel[:, 4:6] + fs_j[:, None] * panel[:, 6:8] + ss_i[:, None] * panel[:, 8:10]
    )
    return torch.where(on[:, None], placed, xy)


def _kernel(t: Tensor, dtype: torch.dtype):
    if (
        t.is_cuda
        and t.device.index == torch.cuda.current_device()
        and dtype == torch.float32
        and t.shape[0] > 0
        and not t.requires_grad
    ):
        return select("forward", True)
    return None


def detector_to_q(
    positions: Tensor,
    geometry: dict,
    frame_rotation: Optional[Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    # Detector pixels -> reciprocal-space q (A^-1) via the Ewald construction
    # (beam +z): pixel lab direction s_hat gives q = (s_hat - s0)/lambda, s0 = (0,0,1).
    # frame_rotation (3,3) optionally maps q from lab into the crystal frame.
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError("positions must be (N, 2)")
    device = positions.device
    g = _geometry_constants(geometry, device=device, dtype=dtype)
    pos = positions.to(device=device, dtype=dtype)
    bases = _panel_model_bases(geometry, device, dtype)

    kernel = _kernel(pos, dtype)
    if kernel is not None:
        q_lab = kernel.lift(pos, bases, g["packed"])
    else:
        xy_lab = _lab_xy_pixels(pos, geometry, bases) * g["pix_A"]
        sq = xy_lab * xy_lab
        r = torch.sqrt(sq[:, 0] + sq[:, 1] + g["clen_sq"])
        qz = (g["clen_A"] / r - 1.0) * g["inv_lambda"]
        q_lab = torch.cat(
            [(xy_lab / r[:, None]) * g["inv_lambda"], qz[:, None]], dim=-1
        )
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
    P = bases.shape[0]
    min_ss, max_ss, min_fs, max_fs = bases[:, 0], bases[:, 1], bases[:, 2], bases[:, 3]
    cx, cy = bases[:, 4], bases[:, 5]
    fsx, fsy, ssx, ssy = bases[:, 6], bases[:, 7], bases[:, 8], bases[:, 9]
    fs_len = (max_fs - min_fs)[:, None]
    ss_len = (max_ss - min_ss)[:, None]
    det = (fsx * ssy - ssx * fsy)[:, None]
    rx = x_pix[None, :] - cx[:, None]
    ry = y_pix[None, :] - cy[:, None]
    fs_j = (ssy[:, None] * rx - ssx[:, None] * ry) / det
    ss_i = (-fsy[:, None] * rx + fsx[:, None] * ry) / det
    on = (
        valid[None, :] & (fs_j >= 0) & (fs_j <= fs_len) & (ss_i >= 0) & (ss_i <= ss_len)
    )
    # each q takes the first panel it lands on
    pid = torch.arange(P, device=x_pix.device)[:, None]
    chosen = torch.where(on, pid, P).amin(0)
    has = chosen < P
    sel = chosen.clamp_max(P - 1)
    ar = torch.arange(n, device=x_pix.device)
    row = torch.where(has, min_ss[sel] + ss_i[sel, ar], float("nan"))
    col = torch.where(has, min_fs[sel] + fs_j[sel, ar], float("nan"))
    return torch.stack([row, col], dim=-1)


def q_to_detector(
    q: Tensor,
    geometry: dict,
    frame_rotation: Optional[Tensor] = None,
    dtype: torch.dtype = torch.float32,
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
    bases = _panel_model_bases(geometry, device, dtype)
    kernel = _kernel(qv, dtype)
    if kernel is not None:
        return kernel.project(qv, bases, g["packed"])
    inv_lambda = 1.0 / g["wavelength_A"]
    Sx = qv[:, 0] / inv_lambda
    Sy = qv[:, 1] / inv_lambda
    Sz = qv[:, 2] / inv_lambda + 1.0
    scale = g["clen_A"] / Sz.clamp_min(1e-30)
    x_pix = (Sx * scale) / g["pix_A"]
    y_pix = (Sy * scale) / g["pix_A"]

    if bases is None:
        col = x_pix + g["bc_col"]
        row = y_pix + g["bc_row"]
        return torch.stack([row, col], dim=-1)
    return _project_to_panels(x_pix, y_pix, Sz > 0, bases)
