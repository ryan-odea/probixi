from __future__ import annotations

import math

import torch
from torch import Tensor

A_INV_TO_NM_INV = 10.0
MIN_ANNULUS_PIXELS = 10


def snap_positions(
    positions: Tensor, observed: Tensor, radius: float
) -> tuple[Tensor, Tensor]:
    # Move each prediction onto the nearest observed centroid within radius px
    snapped = torch.zeros(len(positions), dtype=torch.bool, device=positions.device)
    if len(positions) and len(observed):
        distance, index = torch.cdist(positions, observed.to(positions)).min(1)
        snapped = distance < radius
        positions = torch.where(
            snapped[:, None], observed.to(positions)[index], positions
        )
    return positions, snapped


def keep_non_overlapping(
    positions: Tensor,
    integration_radius_px: float,
    panels: Tensor | None = None,
) -> Tensor:
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if not math.isfinite(integration_radius_px) or integration_radius_px <= 0:
        raise ValueError("integration radius must be finite and positive")
    keep = torch.ones(len(positions), dtype=torch.bool, device=positions.device)
    positions = positions.float()
    cutoff2 = (1.5 * integration_radius_px) ** 2
    for start in range(0, len(positions), 512):
        block = positions[start : start + 512]
        distance2 = (block[:, None] - positions[None]).square().sum(-1)
        row = torch.arange(len(block), device=positions.device)
        distance2[row, start + row] = float("inf")  # ignore self-distance
        if panels is not None:
            distance2.masked_fill_(
                panels[start : start + 512, None] != panels[None], float("inf")
            )
        keep[start : start + 512] = distance2.amin(1) >= cutoff2
    return keep


@torch.no_grad()
def radial_profile(
    excess: Tensor,
    positions: Tensor,
    pixel_valid: Tensor | None = None,
    max_radius: int = 16,
) -> Tensor:
    out = excess.new_zeros(max_radius + 1)
    if not len(positions):
        return out
    off = torch.arange(-max_radius, max_radius + 1, device=excess.device)
    dr, dc = torch.meshgrid(off, off, indexing="ij")
    dr = dr.flatten()
    dc = dc.flatten()
    rbin = torch.sqrt((dr * dr + dc * dc).to(excess.dtype)).round().long()
    inside = rbin <= max_radius
    centre = positions.round().long()
    rr = centre[:, 0, None] + dr
    cc = centre[:, 1, None] + dc
    ok = (
        inside & (rr >= 0) & (rr < excess.shape[0]) & (cc >= 0) & (cc < excess.shape[1])
    )
    flat = rr.clamp(0, excess.shape[0] - 1) * excess.shape[1] + cc.clamp(
        0, excess.shape[1] - 1
    )
    if pixel_valid is not None:
        ok &= pixel_valid.flatten()[flat]
    values = torch.where(
        ok, excess.flatten()[flat], torch.zeros_like(excess.flatten()[flat])
    )
    # corners of the square stamp exceed max_radius but are already zeroed by
    # `ok`, so clamping their bin index is harmless
    index = rbin.clamp(0, max_radius).expand_as(values)
    total = excess.new_zeros(len(positions), max_radius + 1)
    total.scatter_add_(1, index, values)
    counts = excess.new_zeros(len(positions), max_radius + 1)
    counts.scatter_add_(1, index, ok.to(excess.dtype))
    return (total / counts.clamp_min(1.0)).median(dim=0).values


def radii_from_profile(
    profile: Tensor,
    floor: float = 0.02,
    gap: float = 1.0,
    background_pixels: float = 120.0,
    fit_above: float = 0.1,
) -> tuple[float, float, float] | None:
    if not 0.0 < floor < 1.0:
        raise ValueError("floor must be in (0, 1)")
    if not 0.0 < fit_above < 1.0:
        raise ValueError("fit_above must be in (0, 1)")
    if gap <= 0 or background_pixels <= 0:
        raise ValueError("gap and background_pixels must be positive")
    p = profile.detach().float()
    p = p - p.min()
    centre = float(p[0])
    if not math.isfinite(centre) or centre <= 0.0:
        return None
    r_sig = None
    ok = (p >= fit_above * centre) & (p > 0)
    failed = (~ok).nonzero()
    n = int(failed[0, 0]) if len(failed) else len(p)
    if n >= 3:
        x = torch.arange(n, device=p.device, dtype=p.dtype) ** 2
        y = torch.log(p[:n])
        dx = x - x.mean()
        denom = float((dx * dx).sum())
        if denom > 0.0:
            slope = float((dx * (y - y.mean())).sum()) / denom
            if slope < 0.0:
                sigma = math.sqrt(-1.0 / (2.0 * slope))
                r_sig = sigma * math.sqrt(-2.0 * math.log(floor))
    if r_sig is None:
        below = (p <= floor * centre).nonzero()
        if not len(below):
            return None
        r_sig = float(below[0, 0])
    if not math.isfinite(r_sig):
        return None
    r_sig = min(max(r_sig, 1.0), float(len(p) - 1))
    r_in = r_sig + gap
    r_out = math.sqrt(r_in * r_in + background_pixels / math.pi)
    return r_sig, r_in, r_out


@torch.no_grad()
def integrate_rings(
    pred_positions: Tensor,
    excess: Tensor,
    var: Tensor,
    obs_positions: Tensor,
    snap_radius: float = 5.0,
    mean: Tensor | None = None,
    pixel_valid: Tensor | None = None,
    adu_per_photon: float = 1.0,
    n_bg: float | None = None,
    *,
    radii: tuple[float, float, float],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if mean is None:
        raise ValueError("circular integration requires the background mean image")
    positions, snapped = snap_positions(pred_positions, obs_positions, snap_radius)
    if not len(positions):
        empty = excess.new_empty(0)
        return positions, empty, empty, snapped, empty, empty
    raw = excess + mean
    # gather one (2*ceil(outer)+1)^2 stamp per centre
    extent = math.ceil(radii[2])
    off = torch.arange(-extent, extent + 1, device=raw.device)
    dr, dc = torch.meshgrid(off, off, indexing="ij")
    dr = dr.flatten()
    dc = dc.flatten()
    radius = dr**2 + dc**2
    center = positions.round().long()
    rr = center[:, 0, None] + dr
    cc = center[:, 1, None] + dc
    ok = (rr >= 0) & (rr < raw.shape[0]) & (cc >= 0) & (cc < raw.shape[1])
    flat = rr.clamp(0, raw.shape[0] - 1) * raw.shape[1] + cc.clamp(0, raw.shape[1] - 1)
    if pixel_valid is not None:
        ok &= pixel_valid.flatten()[flat]
    pixels = raw.flatten()[flat]
    # background level and spread from the annulus
    bgmask = ok & (radius >= radii[1] ** 2) & (radius <= radii[2] ** 2)
    nb = bgmask.sum(1)
    background = torch.where(bgmask, pixels, 0).sum(1) / nb.clamp_min(1)
    bgvariance = torch.where(bgmask, (pixels - background[:, None]) ** 2, 0).sum(1) / (
        nb - 1
    ).clamp_min(1)
    use = ok & (radius <= radii[0] ** 2)
    # Sparse nearest-owner reduction prevents double counting overlapping disks.
    unique, inverse = torch.unique(flat.flatten(), return_inverse=True)
    inverse = inverse.reshape_as(flat)
    distance = (rr - positions[:, 0, None]) ** 2 + (cc - positions[:, 1, None]) ** 2
    nearest = torch.full_like(unique, float("inf"), dtype=raw.dtype)
    nearest.scatter_reduce_(
        0,
        inverse.flatten(),
        torch.where(use, distance, float("inf")).flatten(),
        reduce="amin",
    )
    wins = use & (distance <= nearest[inverse] + 1e-6)
    # break ties between equidistant centres by lowest index
    cid = torch.arange(len(positions), device=raw.device)[:, None].expand_as(flat)
    owner = torch.full_like(unique, len(positions))
    owner.scatter_reduce_(
        0,
        inverse.flatten(),
        torch.where(wins, cid, len(positions)).flatten(),
        reduce="amin",
    )
    use = wins & (owner[inverse] == cid)
    n = use.sum(1)
    intensity = torch.where(use, pixels - background[:, None], 0).sum(1)
    totalvar = (
        n * bgvariance * (1 + n / nb.clamp_min(1))
        + intensity.clamp_min(0) * adu_per_photon
    )
    sigma = totalvar.clamp_min(1e-12).sqrt()
    model_var = torch.where(use, var.flatten()[flat], torch.zeros_like(pixels)).sum(1)
    if n_bg is not None and n_bg > 0:
        model_var = model_var * (1.0 + n / float(n_bg))
    fallback = (
        (model_var + intensity.clamp_min(0) * adu_per_photon).clamp_min(1e-12).sqrt()
    )
    sigma = torch.where(nb >= MIN_ANNULUS_PIXELS, sigma, fallback)
    sigma = torch.where(n > 0, sigma, torch.zeros_like(sigma))
    peak = torch.where(use, pixels - background[:, None], float("-inf")).amax(1)
    peak = torch.where(torch.isfinite(peak), peak, 0)
    return positions, intensity, sigma, snapped, peak, background


@torch.no_grad()
def peak_resolution_limit(
    peak_resolution: Tensor,
    percentile: float,
    snr: Tensor | None = None,
    snr_floor: float = 0.0,
) -> float:
    # Per-crystal diffraction limit from a percentile of the indexed peaks' |q|.
    if percentile <= 0.0 or peak_resolution.numel() == 0:
        return float("inf")
    vals = peak_resolution
    if snr is not None and snr_floor > 0.0:
        keep = torch.isfinite(snr) & (snr >= snr_floor)
        if bool(keep.any()):
            vals = peak_resolution[keep]
    q = min(percentile, 1.0)
    vals = torch.sort(vals).values
    idx = min(int(vals.numel()) - 1, int(q * int(vals.numel())))
    return float(vals[idx])


@torch.no_grad()
def falloff_resolution_limit(
    q_nm: Tensor,
    isig: Tensor,
    target: float = 1.0,
    nbins: int = 10,
    min_refl: int = 40,
) -> float | None:
    # Per-crystal diffraction limit (nm^-1): |q| where shell-mean I/sigma crosses target.
    n = int(q_nm.shape[0])
    if n < min_refl:
        return None
    order = torch.argsort(q_nm)
    qs = q_nm[order].tolist()
    iss = isig[order].tolist()
    bq: list[float] = []
    bm: list[float] = []
    for i in range(nbins):
        lo = (i * n) // nbins
        hi = ((i + 1) * n) // nbins
        if hi <= lo:
            continue
        seg = qs[lo:hi]  # already sorted by |q|
        bq.append(seg[len(seg) // 2])  # shell median |q|
        bm.append(sum(iss[lo:hi]) / (hi - lo))  # shell mean I/sigma
    if not bq:
        return None
    if bm[0] < target:  # even the innermost shell is at/below noise
        return bq[0]
    for i in range(1, len(bm)):
        if bm[i] < target:  # interpolate the crossing between shell i-1 and i
            f = (bm[i - 1] - target) / (bm[i - 1] - bm[i])
            return bq[i - 1] + f * (bq[i] - bq[i - 1])
    return qs[-1]  # data-limited: never crosses target


@torch.no_grad()
def spot_enrichment(
    positions: Tensor,
    excess: Tensor,
    var: Tensor,
    z_threshold: float,
    radius: int = 2,
    pixel_valid: Tensor | None = None,
) -> tuple[int, float, float]:
    # Significance of an indexing solution vs the image: n_bright predicted spots
    # clearing z_threshold, enrichment = observed/chance bright-rate (~1 noise, >>1
    # real), and Poisson p-value P(X >= n_bright), X ~ Poisson(M*p) for background
    # bright-rate p.
    z = excess / var.clamp_min(1e-12).sqrt()
    if pixel_valid is not None:
        z = torch.where(pixel_valid, z, z.new_full((), float("-inf")))
    zmax = torch.nn.functional.max_pool2d(
        z[None, None], kernel_size=2 * radius + 1, stride=1, padding=radius
    )[0, 0]
    bright = zmax > z_threshold
    valid = pixel_valid if pixel_valid is not None else torch.ones_like(bright)
    M = positions.shape[0]
    if M == 0:
        return 0, 0.0, 1.0
    centre = torch.round(positions).to(torch.long)
    r = centre[:, 0].clamp(0, z.shape[0] - 1)
    c = centre[:, 1].clamp(0, z.shape[1] - 1)
    keep = valid[r, c]
    # int64 counts -> Python ints
    bright_valid, valid_total, keep_total, bright_keep = torch.stack(
        [
            (bright & valid).sum(),
            valid.sum(),
            keep.sum(),
            (bright[r, c] & keep).sum(),
        ]
    ).tolist()
    p = bright_valid / max(valid_total, 1.0)  # background bright-rate
    n_keep = max(keep_total, 1.0)
    n_bright = int(bright_keep)
    enrichment = (
        (n_bright / n_keep) / p if p > 0.0 else float(n_bright > 0) * float("inf")
    )

    lam = n_keep * p
    if n_bright == 0:
        p_value = 1.0
    elif lam <= 0.0:
        p_value = 0.0
    else:
        k = torch.tensor(float(n_bright), dtype=torch.float32)
        p_value = float(
            torch.special.gammainc(k, torch.tensor(lam, dtype=torch.float32))
        )
    return n_bright, enrichment, p_value
