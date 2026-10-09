from __future__ import annotations

import math
from functools import lru_cache

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
def nearest_neighbour_distance(positions: Tensor) -> Tensor:
    # Distance (px) from each position to its closest other one
    out = positions.new_full((len(positions),), float("inf"))
    for start in range(0, len(positions), 512):
        d2 = (positions[start : start + 512, None] - positions[None]).square().sum(-1)
        row = torch.arange(len(d2), device=positions.device)
        d2[row, start + row] = float("inf")
        out[start : start + 512] = d2.amin(1).sqrt()
    return out


@torch.no_grad()
def ring_sums(
    excess: Tensor,
    positions: Tensor,
    pixel_valid: Tensor | None = None,
    max_radius: int = 16,
    squares: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None]:
    n_pk = len(positions)
    if n_pk == 0:
        z = excess.new_zeros(0, max_radius + 1)
        return z, z.clone(), (z.clone() if squares else None)
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
    # corners of the square stamp exceed max_radius but are already zeroed by
    # `ok`, so clamping their bin index is harmless
    index = rbin.clamp(0, max_radius).expand_as(flat)
    values = torch.where(ok, excess.flatten()[flat], 0.0)

    def gather(v: Tensor) -> Tensor:
        out = excess.new_zeros(n_pk, max_radius + 1)
        return out.scatter_add_(1, index, v.to(out.dtype))

    return (
        gather(values),
        gather(ok.to(excess.dtype)),
        (gather(values * values) if squares else None),
    )


@torch.no_grad()
def radial_profile(
    excess: Tensor,
    positions: Tensor,
    pixel_valid: Tensor | None = None,
    max_radius: int = 16,
) -> Tensor:
    if len(positions) == 0:
        return excess.new_zeros(max_radius + 1)
    total, counts, _ = ring_sums(excess, positions, pixel_valid, max_radius)
    return (total / counts.clamp_min(1.0)).median(dim=0).values


def _ring_pixel_counts(max_radius: int, device) -> Tensor:
    off = torch.arange(-max_radius, max_radius + 1, device=device)
    dr, dc = torch.meshgrid(off, off, indexing="ij")
    rbin = torch.sqrt((dr * dr + dc * dc).float()).round().long().flatten()
    return torch.bincount(rbin[rbin <= max_radius], minlength=max_radius + 1).float()


def snr_disk_radius(profile: Tensor, r_max: float) -> float:
    p = profile.detach().float()
    p = (p - p.min()).clamp_min(0.0)
    n = _ring_pixel_counts(len(p) - 1, p.device)
    s = torch.cumsum(p * n, 0)
    npx = torch.cumsum(n, 0)
    snr = s / npx.sqrt()
    k_max = max(1, min(len(p) - 1, int(math.floor(r_max - 0.5))))
    k = int(torch.argmax(snr[1 : k_max + 1])) + 1
    return k + 0.5


def radii_from_profile(
    profile: Tensor,
    floor: float = 0.02,
    gap: float = 1.0,
    background_pixels: float = 120.0,
    fit_above: float = 0.1,
    snr: bool = False,
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
    if snr:
        # annulus still clears the 2 % radius; only the signal disk shrinks
        r_sig = min(r_sig, snr_disk_radius(profile, r_sig))
    return r_sig, r_in, r_out


@torch.no_grad()
def choose_aperture(
    sums: Tensor,
    squares: Tensor,
    counts: Tensor,
    q: Tensor,
    frame: Tensor,
    radii_snr: tuple[float, float, float],
    radii_flux: tuple[float, float, float],
    adu_per_photon: float = 1.0,
    tolerance: float = 0.02,
    min_isig: float = 10.0,
    min_peaks: int = 50,
    min_crystals: int = 10,
    nn_dist: Tensor | None = None,
) -> tuple[str, dict]:
    """Pick ``"snr"`` or ``"flux"`` from the calibration peaks' ring sums"""
    r_snr, r_flux = radii_snr[0], radii_flux[0]
    diag = dict(r_snr=float(r_snr), r_flux=float(r_flux), n_peaks=len(sums), n=0)
    if r_snr >= r_flux:
        return "flux", dict(diag, reason="snr disk is the flux disk")
    k_s, k_f = int(math.floor(r_snr)), int(round(r_flux))
    k_in, k_out = int(math.ceil(radii_flux[1])), int(math.floor(radii_flux[2]))
    k_out = min(k_out, sums.shape[1] - 1)
    sums, squares, counts = sums.float(), squares.float(), counts.float()
    n_ann = counts[:, k_in : k_out + 1].sum(1)
    s_ann = sums[:, k_in : k_out + 1].sum(1)
    bg = s_ann / n_ann.clamp_min(1)
    bgvar = (squares[:, k_in : k_out + 1].sum(1) - s_ann * bg) / (n_ann - 1).clamp_min(
        1
    )
    n_s, n_f = counts[:, : k_s + 1].sum(1), counts[:, : k_f + 1].sum(1)
    i_s = sums[:, : k_s + 1].sum(1) - n_s * bg
    i_f = sums[:, : k_f + 1].sum(1) - n_f * bg
    shot = i_s.clamp_min(0) * adu_per_photon
    v_s = n_s * bgvar * (1 + n_s / n_ann.clamp_min(1)) + shot
    v_f = (
        n_f * bgvar * (1 + n_f / n_ann.clamp_min(1)) + i_f.clamp_min(0) * adu_per_photon
    )

    cov = n_s * bgvar * (1 + n_f / n_ann.clamp_min(1)) + shot
    strong = (n_ann >= MIN_ANNULUS_PIXELS) & (i_s > 0) & (i_f > 0) & (v_f > 0)
    strong &= i_f / v_f.clamp_min(1e-12).sqrt() >= min_isig
    diag["n_strong"] = int(strong.sum())
    if nn_dist is not None:
        strong &= nn_dist.to(strong.device) >= 2 * r_flux + 1
    diag["n"] = int(strong.sum())
    if diag["n"] < min_peaks:
        return "snr", dict(diag, reason="too few strong peaks")
    i_s, i_f, v_s, v_f, cov = (t[strong] for t in (i_s, i_f, v_s, v_f, cov))
    f = i_s / i_f
    y = torch.log(f)
    w = f * f / ((v_s - 2 * f * cov + f * f * v_f) / (i_f * i_f)).clamp_min(1e-12)
    x = q.float()[strong] ** 2
    diag["f_mean"] = float(f.mean())
    order = torch.argsort(x)
    n_bin = max(3, min(8, len(x) // 20))
    edges = [(i * len(x)) // n_bin for i in range(n_bin + 1)]
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = order[lo:hi]
        bx.append(float((w[sel] * x[sel]).sum() / w[sel].sum()))
        by.append(float((w[sel] * y[sel]).sum() / w[sel].sum()))
    bx_t, by_t = x.new_tensor(bx), x.new_tensor(by)
    pos = torch.bucketize(x, bx_t).clamp(1, len(bx) - 1)
    x0, x1, y0, y1 = bx_t[pos - 1], bx_t[pos], by_t[pos - 1], by_t[pos]
    common = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    span = float(torch.quantile(x, 0.9) - torch.quantile(x, 0.5))
    diag["common_trend"] = by[-1] - by[len(by) // 2]
    y = y - common

    # per-crystal weighted regression of the residual log f on |q|^2
    crystal, c_idx = torch.unique(frame[strong], return_inverse=True)
    J = len(crystal)

    def acc(v: Tensor) -> Tensor:
        return torch.zeros(J, dtype=x.dtype, device=x.device).index_add_(0, c_idx, v)

    sw, swx, swy = acc(w), acc(w * x), acc(w * y)
    xm, ym = swx / sw, swy / sw
    sxx = acc(w * x * x) - sw * xm * xm
    sxy = acc(w * x * y) - sw * xm * ym
    syy = acc(w * y * y) - sw * ym * ym
    n_j = acc(torch.ones_like(w))
    ok = (n_j >= 3) & (sxx > 0)
    diag["n_crystals"] = int(ok.sum())
    if diag["n_crystals"] < min_crystals:
        return "snr", dict(diag, reason="too few crystals with enough peaks")
    b = (sxy / sxx)[ok]
    var_b = (1.0 / sxx)[ok]

    # per-reflection scatter beyond the noise model (sub-pixel centring, shape)
    chi2_within = float(((syy - sxy * sxy / sxx)[ok]).sum() / (n_j[ok] - 2).sum())
    var_b = var_b * max(chi2_within, 1.0)
    b_mean = float((b / var_b).sum() / (1.0 / var_b).sum())
    dof = diag["n_crystals"] - 1
    dev = (b - b_mean) ** 2 / var_b

    capped = dev.clamp_max(9.0)
    excess = max(float(capped.mean()) / 0.995 - 1.0, 0.0) * float(var_b.median())
    diag.update(
        chi2_within=chi2_within,
        residual_trend=b_mean * span,
        crystal_spread=math.sqrt(excess) * span,
        spread_z=(float(capped.sum()) - 0.995 * len(dev)) / math.sqrt(1.90 * len(dev)),
        max_dev=float(dev.max()),
        plain_chi2_red=float(dev.sum()) / dof,
    )
    if diag["crystal_spread"] >= tolerance and diag["spread_z"] >= 3.0:
        return "flux", dict(
            diag, reason="captured fraction's resolution trend differs between crystals"
        )
    return "snr", dict(
        diag, reason="captured fraction's resolution trend is common to all crystals"
    )


@lru_cache(maxsize=8)
def _stamp_offsets(extent: int, device) -> tuple[Tensor, Tensor, Tensor]:
    off = torch.arange(-extent, extent + 1, device=device)
    dr, dc = torch.meshgrid(off, off, indexing="ij")
    dr = dr.flatten()
    dc = dc.flatten()
    return dr, dc, dr**2 + dc**2


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
        return positions, empty, empty, snapped, empty, empty, empty, empty, empty
    raw = excess + mean
    # gather one (2*ceil(outer)+1)^2 stamp per centre
    dr, dc, radius = _stamp_offsets(math.ceil(radii[2]), raw.device)
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
    net = pixels - background[:, None]
    bgvariance = torch.where(bgmask, net**2, 0).sum(1) / (nb - 1).clamp_min(1)
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
    intensity = torch.where(use, net, 0).sum(1)
    totalvar = (
        n * bgvariance * (1 + n / nb.clamp_min(1))
        + intensity.clamp_min(0) * adu_per_photon
    )
    sigma = totalvar.clamp_min(1e-12).sqrt()
    model_var_sum = torch.where(use, var.flatten()[flat], torch.zeros_like(pixels)).sum(
        1
    )
    model_var = model_var_sum
    if n_bg is not None and n_bg > 0:
        model_var = model_var * (1.0 + n / float(n_bg))
    fallback = (
        (model_var + intensity.clamp_min(0) * adu_per_photon).clamp_min(1e-12).sqrt()
    )
    sigma = torch.where(nb >= MIN_ANNULUS_PIXELS, sigma, fallback)
    sigma = torch.where(n > 0, sigma, torch.zeros_like(sigma))
    peak = torch.where(use, net, float("-inf")).amax(1)
    peak = torch.where(torch.isfinite(peak), peak, 0)
    bg_model = torch.where(use, mean.flatten()[flat], torch.zeros_like(pixels)).sum(1)
    if n_bg is not None and n_bg > 0:
        bg_model_var = model_var_sum * n / float(n_bg)
    else:
        bg_model_var = torch.zeros_like(model_var_sum)
    return (
        positions,
        intensity,
        sigma,
        snapped,
        peak,
        background,
        n,
        bg_model,
        bg_model_var,
    )


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
    if positions.shape[0] == 0:
        return 0, 0.0, 1.0
    z = excess / var.clamp_min(1e-12).sqrt()
    if pixel_valid is not None:
        z = torch.where(pixel_valid, z, z.new_full((), float("-inf")))
    zmax = torch.nn.functional.max_pool2d(
        z[None, None], kernel_size=2 * radius + 1, stride=1, padding=radius
    )[0, 0]
    bright = zmax > z_threshold
    valid = pixel_valid if pixel_valid is not None else torch.ones_like(bright)
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
