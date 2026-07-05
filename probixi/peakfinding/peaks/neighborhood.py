from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def gaussian_kernel_2d(
    size: int,
    sigma: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if size < 1 or size % 2 != 1:
        raise ValueError("size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    ax = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g1 = torch.exp(-0.5 * (ax / sigma) ** 2)
    k = g1.view(-1, 1) * g1.view(1, -1)
    return k / k.sum()


@torch.no_grad()
def _box_sum(x: Tensor, radius: int) -> Tensor:
    # Sliding-window sum over a (2r+1)^2 box, (H,W) or (B,H,W)
    # Zero-padded so an edge box sums only in-bounds pixels
    if radius < 1:
        return x.clone()
    squeeze = x.ndim == 2
    xb = x.view(1, 1, *x.shape) if squeeze else x.unsqueeze(1)
    k = 2 * radius + 1
    kr = x.new_ones(1, 1, k, 1)
    kc = x.new_ones(1, 1, 1, k)
    xb = F.conv2d(F.pad(xb, (0, 0, radius, radius)), kr)
    xb = F.conv2d(F.pad(xb, (radius, radius, 0, 0)), kc)
    return xb.reshape(x.shape)


@torch.no_grad()
def annulus_count(
    mask: Tensor, inner_radius: int, outer_radius: int, dtype: torch.dtype
) -> Tensor:
    # per-pixel valid-pixel count over the (outer minus inner) box annulus
    m = mask.to(dtype)
    co = _box_sum(m, outer_radius)
    ci = _box_sum(m, inner_radius) if inner_radius >= 1 else torch.zeros_like(co)
    return (co - ci).clamp_min(1.0)


@torch.no_grad()
def local_mean_var(
    residual: Tensor,
    mask: Tensor,
    inner_radius: int,
    outer_radius: int,
    count: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    # Per-pixel mean and variance of residual over the masked annulus
    # inner_radius < |offset|_inf <= outer_radius (outer box minus inner box)
    # Excluding the inner box keeps a peak from biasing its own background.
    # ``count`` (annulus valid-pixel count) may be supplied precomputed, else None.
    m = mask.to(residual.dtype)
    rm = residual * m
    rm2 = residual * rm
    so = _box_sum(rm, outer_radius)
    s2o = _box_sum(rm2, outer_radius)
    if inner_radius >= 1:
        si = _box_sum(rm, inner_radius)
        s2i = _box_sum(rm2, inner_radius)
    else:
        si = torch.zeros_like(so)
        s2i = torch.zeros_like(s2o)
    if count is None:
        co = _box_sum(m, outer_radius)
        ci = _box_sum(m, inner_radius) if inner_radius >= 1 else torch.zeros_like(co)
        count = (co - ci).clamp_min(1.0)
    mean = (so - si) / count
    ex2 = (s2o - s2i) / count
    var = (ex2 - mean * mean).clamp_min(0.0)
    return mean, var


@torch.no_grad()
def matched_filter_z(
    z: Tensor,
    kernel: Tensor,
    mask: Tensor,
    den: Optional[Tensor] = None,
) -> Tensor:
    # Correlate z with the L2-norm
    u = (kernel / kernel.norm().clamp_min(1e-12)).to(z)
    kH, kW = u.shape
    pad = (kW // 2, kW // 2, kH // 2, kH // 2)
    uk = u.view(1, 1, kH, kW)
    squeeze = z.ndim == 2
    zb = z.view(1, 1, *z.shape) if squeeze else z.unsqueeze(1)
    m = mask.to(z)
    num = F.conv2d(F.pad((zb * m.view(1, 1, *mask.shape)), pad), uk)
    if den is None:
        den = F.conv2d(
            F.pad(m.view(1, 1, *mask.shape), pad), (u * u).view(1, 1, kH, kW)
        ).reshape(mask.shape)
    return num.reshape(z.shape) / den.clamp_min(1e-12).sqrt()


@torch.no_grad()
def matched_filter_denominator(mask: Tensor, kernel: Tensor) -> Tensor:
    # Per-pixel valid kernel energy sum(u^2 * mask)
    u = kernel / kernel.norm().clamp_min(1e-12)
    kH, kW = u.shape
    pad = (kW // 2, kW // 2, kH // 2, kH // 2)
    m = mask.to(u).view(1, 1, *mask.shape)
    return F.conv2d(F.pad(m, pad), (u * u).view(1, 1, kH, kW)).reshape(mask.shape)


@torch.no_grad()
def mask_denominator(mask: Tensor, kernel: Tensor) -> Tensor:
    # Per-pixel normaliser for smoothing
    if kernel.ndim != 2:
        raise ValueError("kernel must be 2D")
    kH, kW = kernel.shape
    pad = (kW // 2, kW // 2, kH // 2, kH // 2)
    k = kernel.view(1, 1, kH, kW)
    m = mask.to(kernel).view(1, 1, *mask.shape)
    return (
        F.conv2d(F.pad(m, pad, mode="reflect"), k).reshape(mask.shape).clamp_min(1e-12)
    )


@torch.no_grad()
def smooth_logits(
    logits: Tensor,
    kernel: Tensor,
    mask: Optional[Tensor] = None,
    den: Optional[Tensor] = None,
) -> Tensor:
    # Mask-weighted local average K*(x.mask)/(K*mask)
    if logits.ndim != 2 or kernel.ndim != 2:
        raise ValueError("logits and kernel must be 2D")
    kH, kW = kernel.shape
    pad = (kW // 2, kW // 2, kH // 2, kH // 2)
    k = kernel.to(logits).view(1, 1, kH, kW)
    if mask is None:
        return F.conv2d(
            F.pad(logits.view(1, 1, *logits.shape), pad, mode="reflect"), k
        ).reshape(logits.shape)
    m = mask.to(logits)
    num = F.conv2d(
        F.pad((logits * m).view(1, 1, *logits.shape), pad, mode="reflect"), k
    ).reshape(logits.shape)
    return num / (den if den is not None else mask_denominator(m, k.reshape(kH, kW)))


@torch.no_grad()
def smooth_logits_batch(
    logits: Tensor,
    kernel: Tensor,
    mask: Optional[Tensor] = None,
    den: Optional[Tensor] = None,
) -> Tensor:
    # Batched smooth_logits; logits (B,H,W), den (H,W) broadcasts over the batch
    if logits.ndim != 3 or kernel.ndim != 2:
        raise ValueError("logits must be 3D (B,H,W) and kernel must be 2D")
    kH, kW = kernel.shape
    pad = (kW // 2, kW // 2, kH // 2, kH // 2)
    k = kernel.to(logits).view(1, 1, kH, kW)
    if mask is None:
        return F.conv2d(F.pad(logits.unsqueeze(1), pad, mode="reflect"), k).squeeze(1)
    m = mask.to(logits)
    num = F.conv2d(F.pad((logits * m).unsqueeze(1), pad, mode="reflect"), k).squeeze(1)
    return num / (den if den is not None else mask_denominator(mask, k.reshape(kH, kW)))
