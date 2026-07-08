from __future__ import annotations

import torch
from torch import Tensor

A_INV_TO_NM_INV = 10.0

# Box integration of background-subtracted intensity at predicted positions. The
# excess map is already background-subtracted, so integrated intensity is the sum
# of excess over a small box and its variance the sum of per-pixel noise
# variances. The box must hold a spot despite prediction scatter yet stay far
# narrower than the inter-spot spacing so neighbours don't leak in.


@torch.no_grad()
def integrate_boxes(
    excess: Tensor,
    var: Tensor,
    positions: Tensor,
    radius: int = 4,
    mean: Tensor | None = None,
    pixel_valid: Tensor | None = None,
    adu_per_photon: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Integrate intensity (sum excess), sigma (sqrt of summed background noise
    # plus signal shot noise), peak (max excess) and background (mean per-pixel
    # noise) in a (2*radius+1)^2 box at each centre
    H, W = excess.shape
    device = excess.device
    M = positions.shape[0]
    if M == 0:
        z = positions.new_zeros(0)
        return z, z, z, z

    centre = torch.round(positions).to(torch.long)
    r0, c0 = centre[:, 0], centre[:, 1]
    # gather the whole (2r+1)^2 box for every centre at once (M, K), vectorised
    off = torch.arange(-radius, radius + 1, device=device)
    off_r, off_c = torch.meshgrid(off, off, indexing="ij")
    off_r = off_r.reshape(-1)
    off_c = off_c.reshape(-1)
    rr = r0[:, None] + off_r[None, :]
    cc = c0[:, None] + off_c[None, :]
    valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    flat = rr.clamp(0, H - 1) * W + cc.clamp(0, W - 1)
    if pixel_valid is not None:
        valid = valid & pixel_valid.reshape(-1)[flat]
    e = excess.reshape(-1)[flat]
    v = var.reshape(-1)[flat]
    zero = torch.zeros_like(e)
    I = torch.where(valid, e, zero).sum(dim=1)  # noqa: E741
    var_sum = torch.where(valid, v, zero).sum(dim=1)
    peak = torch.where(valid, e, torch.full_like(e, float("-inf"))).amax(dim=1)
    # boxes with no valid pixel
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    if mean is not None:
        m = mean.reshape(-1)[flat]
        bg_sum = torch.where(valid, m, zero).sum(dim=1)
        bg_count = valid.to(excess.dtype).sum(dim=1)
        background = bg_sum / bg_count.clamp_min(1.0)
    else:
        background = torch.zeros(M, dtype=excess.dtype, device=device)
    total_var = var_sum.clamp_min(0.0) + I.clamp_min(0.0) * adu_per_photon
    return I, total_var.sqrt(), peak, background


@torch.no_grad()
def integrate_predicted(
    pred_positions: Tensor,
    excess: Tensor,
    var: Tensor,
    obs_positions: Tensor,
    snap_radius: float = 5.0,
    box_radius: int = 3,
    mean: Tensor | None = None,
    pixel_valid: Tensor | None = None,
    adu_per_photon: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    M = pred_positions.shape[0]
    positions = pred_positions
    snapped = torch.zeros(M, dtype=torch.bool, device=pred_positions.device)
    if obs_positions.shape[0] > 0:
        d = torch.cdist(pred_positions, obs_positions.to(pred_positions.dtype))
        nn_dist, nn_idx = d.min(dim=1)
        snapped = nn_dist < snap_radius
        positions = torch.where(
            snapped.unsqueeze(-1),
            obs_positions.to(pred_positions.dtype)[nn_idx],
            pred_positions,
        )
    intensity, sigma, peak, background = integrate_boxes(
        excess,
        var,
        positions,
        box_radius,
        mean=mean,
        pixel_valid=pixel_valid,
        adu_per_photon=adu_per_photon,
    )
    return positions, intensity, sigma, snapped, peak, background


@torch.no_grad()
def peak_resolution_limit(
    peak_resolution: Tensor,
    percentile: float,
    snr: Tensor | None = None,
    snr_floor: float = 0.0,
) -> float:
    """Per-crystal diffraction limit from the indexed peaks' resolution.
    """
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
    """Per-crystal diffraction limit (nm^-1) from the I/sigma-vs-resolution falloff.
    """
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
    """Significance of an indexing solution against the image.

    Parameters
    ----------
    positions : Tensor
        (M, 2) predicted ``(row, col)`` spot centres.
    excess, var : Tensor
        Background-subtracted signal and per-pixel noise variance (H, W).
    z_threshold : float
        Whitened significance a local peak must clear to count as "bright"
    radius : int, default 2
        Half-width (px) of the local-max neighbourhood.
    pixel_valid : Tensor, optional
        Boolean (H, W) mask of usable pixels.

    Returns
    -------
    n_bright : int
        Predicted spots whose neighbourhood clears ``z_threshold``.
    enrichment : float
        Observed bright-rate of predicted spots / background bright-rate
        (= observed / chance). ~1 is a noise indexing, >>1 a real lattice.
    p_value : float
        Probability of seeing this many bright predicted spots by chance under
        the noise-null (predicted positions independent of where signal is):
        ``P(X >= n_bright)`` for ``X ~ Poisson(M * p)``, ``p`` the measured
        background bright-rate.
    """
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
        k = torch.tensor(float(n_bright), dtype=torch.float64)
        p_value = float(
            torch.special.gammainc(k, torch.tensor(lam, dtype=torch.float64))
        )
    return n_bright, enrichment, p_value
