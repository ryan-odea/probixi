from __future__ import annotations

import math
import warnings
from collections.abc import Sized
from dataclasses import dataclass
from itertools import islice
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..peaks.neighborhood import (
    LOCAL_BG_CLIP_K,
    gaussian_kernel_2d,
    local_mean_var,
    matched_filter_z,
)


@dataclass
class CalibrationResult:
    """Blend, variance scale, and peak mixture learned by :func:`calibrate_noise`.

    Attributes
    ----------
    weights : dict
        Convex weights over the background means, keyed by ``pixel``,
        ``rotational``, and ``panel``; zero weights are omitted.
    var_scale : float
        Scale ``s`` on the per-pixel standard deviation that gives the background
        ``z`` unit variance.
    kappa : float
        Peak contrast, the standard deviation of the peak component of ``z``; > 1.
    prior_peak : float
        Prior probability that a pixel carries peak signal.
    background_var : float
        Variance of the background ``z`` under the chosen blend before scaling;
        ``var_scale = sqrt(background_var)``.
    n_pixels : int
        Pixels in the random subsample.
    """

    weights: dict
    var_scale: float
    kappa: float
    prior_peak: float
    background_var: float
    n_pixels: int

    def apply(self, noise_model, finder=None) -> "CalibrationResult":
        """Install the learned parameters and return ``self``.

        ``noise_model`` receives the blend and variance scale; ``finder``, if given,
        receives ``kappa``, ``prior_peak``, and the ``calibrated`` combination.
        """
        noise_model.calibrated_weights = dict(self.weights)
        noise_model.var_scale = float(self.var_scale)
        if finder is not None:
            finder.combination = "calibrated"
            finder.kappa = float(self.kappa)
            finder.prior_peak = float(self.prior_peak)
            finder.log_prior_odds = math.log(self.prior_peak / (1.0 - self.prior_peak))
        return self


def _simplex_grid(step: float) -> list[tuple[float, float, float]]:
    # Convex weights (a, b, c), a+b+c=1, on a grid of spacing `step`.
    n = max(1, round(1.0 / step))
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((i / n, j / n, k / n))
    return pts


def _em_two_gaussian(
    z: Tensor, iters: int, kappa0: float = 10.0, pi0: float = 1e-3
) -> tuple[float, float]:
    # EM for (1-pi) N(0,1) + pi N(0, kappa^2); background pinned to unit var.
    z2 = z * z
    pi, kappa = pi0, kappa0
    for _ in range(iters):
        d_bg = torch.exp(-0.5 * z2)
        d_pk = torch.exp(-0.5 * z2 / (kappa * kappa)) / kappa
        num = pi * d_pk
        r = num / ((1.0 - pi) * d_bg + num + 1e-30)
        pi = float(r.mean().clamp(1e-6, 0.5))
        denom = r.sum().clamp_min(1.0)
        kappa = float(((r * z2).sum() / denom).clamp_min(1.0 + 1e-3).sqrt())
    return kappa, pi


@torch.no_grad()
def calibrate_noise(
    noise_model,
    seed_frames: Iterable[Tensor],
    *,
    warm: bool = True,
    subsample: int = 40000,
    weight_step: float = 0.1,
    em_iters: int = 50,
    robust_z: float = 3.0,
    kurtosis_weight: float = 0.5,
    rng_seed: int = 0,
) -> CalibrationResult:
    """Fit the background blend, variance scale, and peak mixture on seed frames.

    A simplex grid of blend weights is searched for the background ``z`` closest to
    N(0, 1) in location, scale, and excess kurtosis, and EM then fits
    ``(1 - prior_peak) N(0, 1) + prior_peak N(0, kappa^2)`` to the calibrated ``z``.
    See docs/overview/peakfinding.md.

    Parameters
    ----------
    noise_model : NoiseModel
        Built model; warmed on ``seed_frames`` when ``warm`` is set.
    seed_frames : Iterable[Tensor]
        A sequence of ``(H, W)`` frames or a stacked ``(F, H, W)`` tensor; any other
        iterable is first copied into a list.
    warm : bool
        Update the running model on the seed frames first.
    subsample : int
        Valid pixels drawn at random, the same locations on every frame.
    weight_step : float
        Spacing of the simplex grid of blend weights.
    em_iters : int
        EM iterations for the peak mixture.
    robust_z : float
        Half-width of the background set ``|z| < robust_z``, with ``z`` taken from
        the pixel mean alone.
    kurtosis_weight : float
        Weight on the squared excess kurtosis in the blend loss.
    rng_seed : int
        Seed for the pixel subsample.

    Returns
    -------
    CalibrationResult
        The learned blend, variance scale, and peak mixture parameters.
    """
    frames = seed_frames if isinstance(seed_frames, Sized) else list(seed_frames)
    if len(frames) == 0:
        raise ValueError("seed_frames is empty")
    if warm:
        for f in frames:
            noise_model.update(f)

    device = noise_model.pixel.mean_.device
    dtype = noise_model.pixel.mean_.dtype
    mask = noise_model.valid_mask

    mu = {
        "pixel": noise_model.pixel.mean(),
        "rotational": noise_model.rotational.mean(),
        "panel": noise_model.panel.mean(),
    }
    sigma = noise_model.pixel.var().clamp_min(1e-12).sqrt()

    valid_idx = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        raise ValueError("no valid pixels to calibrate on")
    gen = torch.Generator()
    gen.manual_seed(rng_seed)
    take = min(subsample, valid_idx.numel())
    perm = torch.randperm(valid_idx.numel(), generator=gen)[:take]
    idx = valid_idx[perm.to(valid_idx.device)]

    src_order = ["pixel", "rotational", "panel"]
    M = torch.stack([mu[k].flatten()[idx] for k in src_order])  # (3, S)
    sig_s = sigma.flatten()[idx]  # (S,)
    X = torch.stack([f.to(dtype).flatten()[idx] for f in frames])  # (F, S)

    # background set from the pixel mean alone, so it is the same for every blend
    z_pix = (X - M[0][None, :]) / sig_s[None, :]
    bg = z_pix.abs() < robust_z

    # search the convex blend that makes the background z most N(0,1)
    best: Optional[tuple[float, tuple, float]] = None
    for w in _simplex_grid(weight_step):
        wt = torch.tensor(w, dtype=dtype, device=device)
        mean_w = (wt[:, None] * M).sum(0)
        z = (X - mean_w[None, :]) / sig_s[None, :]
        zb = z[bg]
        if zb.numel() < 100:
            continue
        m = zb.mean()
        v = zb.var(unbiased=False)
        kurt = (zb.pow(4).mean() / v.clamp_min(1e-12).pow(2)) - 3.0
        loss = float(m * m + (v - 1.0) ** 2 + kurtosis_weight * kurt * kurt)
        if best is None or loss < best[0]:
            best = (loss, w, float(v))
    assert best is not None
    _, w_best, v_best = best
    var_scale = math.sqrt(max(v_best, 1e-6))

    wt = torch.tensor(w_best, dtype=dtype, device=device)
    mean_w = (wt[:, None] * M).sum(0)
    z_cal = ((X - mean_w[None, :]) / (var_scale * sig_s[None, :])).flatten()
    kappa, prior_peak = _em_two_gaussian(z_cal, em_iters)

    weights = {k: float(w) for k, w in zip(src_order, w_best) if w > 0.0}
    return CalibrationResult(
        weights=weights,
        var_scale=var_scale,
        kappa=kappa,
        prior_peak=prior_peak,
        background_var=v_best,
        n_pixels=int(idx.numel()),
    )


@torch.no_grad()
def fit_photon_transfer(
    noise_model,
    *,
    n_bins: int = 24,
    min_bin_pixels: int = 500,
) -> tuple[float, float]:
    """Fit the photon-transfer curve ``var = read_var + gain * level``.

    The running variance is measured at the run's typical level, and the flux model
    whitens each frame against this curve at its own level instead
    (docs/vignettes/variable_flux.md). The fit is least squares on the medians of
    variance and mean in quantile bins of the calibrated mean, since a fit on raw
    pixels is ill-conditioned near the origin. A negative intercept or non-positive
    slope falls back to ``read_var = 0`` and the median ratio of variance to mean.
    Sets ``noise_model.read_var`` and ``noise_model.gain``.

    Parameters
    ----------
    noise_model : NoiseModel
        Model calibrated by :meth:`CalibrationResult.apply`.
    n_bins : int
        Quantile bins of the background mean.
    min_bin_pixels : int
        Bins with fewer pixels are skipped.

    Returns
    -------
    tuple[float, float]
        ``(read_var, gain)``; ``gain`` is in detector units per photon under
        Poisson statistics.
    """
    pred = noise_model.predict(combination="calibrated")
    mask = pred["mask"]
    mean = pred["mean"][mask].flatten()
    var = pred["var"][mask].clamp_min(1e-12).flatten()

    edges = torch.unique(
        torch.quantile(
            mean, torch.linspace(0, 1, n_bins + 1, dtype=mean.dtype, device=mean.device)
        )
    )
    xs, ys = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        sel = (mean >= lo) & (mean <= hi if last else mean < hi)
        if int(sel.sum()) < min_bin_pixels:
            continue
        xs.append(mean[sel].median())
        ys.append(var[sel].median())

    if len(xs) >= 2:
        x = torch.stack(xs)
        y = torch.stack(ys)
        A = torch.stack([torch.ones_like(x), x], dim=1)
        a, b = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1).tolist()
        if a < 0.0 or b <= 0.0:
            a, b = 0.0, float((y / x.clamp_min(1e-12)).median())
    else:
        a, b = 0.0, float((var / mean.clamp_min(1e-12)).median())

    read_var = max(float(a), 0.0)
    gain = max(float(b), 1e-6)
    noise_model.read_var = read_var
    noise_model.gain = gain
    return read_var, gain


@torch.no_grad()
def background_fluctuation_ratio(
    noise_model,
    frames: Iterable[Tensor],
    local_inner_radius: int,
    local_outer_radius: int,
    *,
    max_frames: int = 16,
    level_quantile: float = 0.75,
) -> float:
    """Return the ratio of across-frame to within-frame background variance.

    The ratio is near 1 for steady exposures and well above 1 when the background
    level changes between shots. Per frame, it is the median, over pixels above the
    ``level_quantile`` of the background level, of the running variance over the
    local annulus variance; the result is the median over frames.

    Parameters
    ----------
    noise_model : NoiseModel
        A calibrated model.
    frames : iterable of torch.Tensor
        Seed frames; the first ``max_frames`` are read.
    local_inner_radius, local_outer_radius : int
        Inner and outer half-widths, in pixels, of the square local annulus.
    max_frames : int
        Frames read from ``frames``.
    level_quantile : float
        Background-level quantile above which pixels count.

    Returns
    -------
    float
        Median ratio, 1.0 if nothing could be measured.
    """
    pred = noise_model.predict(combination="calibrated")
    mu, var, mask = pred["mean"], pred["var"].clamp_min(1e-12), pred["mask"]
    level = mu[mask]
    if level.numel() == 0:
        return 1.0
    cut = level.kthvalue(max(1, int(level_quantile * level.numel()))).values
    bright = mask & (mu >= cut)
    ratios = []
    for f in islice(frames, max_frames):
        f = f.to(dtype=mu.dtype, device=mu.device)
        _, lv = local_mean_var(
            f - mu,
            mask,
            local_inner_radius,
            local_outer_radius,
            clip_hi=LOCAL_BG_CLIP_K * var.sqrt(),
        )
        ok = bright & (lv > 0)
        if ok.any():
            ratios.append((var[ok] / lv[ok]).median())
    return float(torch.stack(ratios).median()) if ratios else 1.0


def flux_variance_floor(
    noise_model,
    level: Tensor,
    local_var: Tensor,
    var0: Tensor,
    floor_coef: float = 0.15,
) -> Tensor:
    # max(read_var + gain * level, floor_coef * var0, local_var), the first if fitted
    rv, g = noise_model.read_var, noise_model.gain
    floor = (float(floor_coef) * var0).clamp_min(1e-12)
    if rv is None or g is None:
        return torch.maximum(local_var, floor)
    base = (rv + g * level.clamp_min(0.0)).clamp_min(1e-12)
    return torch.maximum(torch.maximum(base, floor), local_var)


@dataclass
class ThresholdCalibration:
    """Matched-filter detection threshold learned by :func:`calibrate_threshold`.

    Attributes
    ----------
    mu0 : float
        Empirical-null location of ``T``, the median of the pooled seed values;
        above 0 because ``T`` is a maximum over template widths.
    sigma0 : float
        Empirical-null scale of ``T``, from the values below ``mu0`` reflected
        about it.
    threshold : float
        Detection threshold on ``T``: the smallest grid value whose median count of
        noise peaks per quiet frame is at most ``target_noise_peaks``.
    target_noise_peaks : float
        Noise budget the threshold was solved for, in detections per quiet frame.
    achieved_noise_peaks : float
        Median count of noise peaks above ``threshold`` per quiet frame.
    n_quiet_frames : int
        Seed frames counted as quiet; 0 when the noise was counted on the
        mirrored tail.
    quiet_max_T : float
        Largest ``max(T)`` of a quiet frame, nominally ``mu0 + sigma0 sqrt(2 ln N)``.
    mf_scales : tuple[float, ...]
        Gaussian template widths in pixels.
    """

    mu0: float
    sigma0: float
    threshold: float
    target_noise_peaks: float
    achieved_noise_peaks: float
    n_quiet_frames: int
    quiet_max_T: float
    mf_scales: tuple[float, ...]

    def apply(self, finder) -> "ThresholdCalibration":
        """Install the template bank and threshold on ``finder`` and return ``self``."""
        finder.matched_filter = True
        finder.mf_scales = tuple(float(s) for s in self.mf_scales)
        finder.mf_threshold = float(self.threshold)
        device = finder.noise.valid_mask.device
        dtype = finder.noise.pixel.mean_.dtype
        finder._mf_kernels = [
            gaussian_kernel_2d(
                2 * int(math.ceil(3.0 * s)) + 1, s, device=device, dtype=dtype
            )
            for s in finder.mf_scales
        ]
        finder._pred_key = None  # force MF denominator recompute
        return self


def _empirical_null_central_matching(t_body: Tensor) -> tuple[float, float]:
    # median and reflected lower half: peaks in the right tail enter neither
    mu0 = float(torch.median(t_body))
    left = t_body[t_body < mu0]
    if left.numel() == 0:
        return mu0, 1.0
    reflected = torch.cat([left, 2 * mu0 - left])
    sigma0 = float(reflected.std(unbiased=False).clamp_min(1e-6))
    return mu0, sigma0


@torch.no_grad()
def calibrate_threshold(
    noise_model,
    seed_frames: Iterable[Tensor],
    *,
    target_noise_peaks: float = 10.0,
    quiet_quantile: float = 0.5,
    mf_scales: tuple[float, ...] = (1.0, 1.6, 2.4),
    local_inner_radius: int = 4,
    local_outer_radius: int = 9,
    size_min: int = 2,
    threshold_grid_min: float = 3.0,
    threshold_grid_max: float = 8.0,
    threshold_grid_step: float = 0.1,
    body_subsample: int = 1_000_000,
    flux_variance: bool = False,
    flux_var_floor: float = 0.15,
    rng_seed: int = 0,
    finder=None,
) -> ThresholdCalibration:
    """Learn the matched-filter detection threshold from a noise budget.

    After :func:`calibrate_noise` the background ``z`` is standard normal, but
    ``T = max_s T_s`` is not: the maximum over widths raises its location and the
    template correlation fattens its right tail. The null of ``T`` is therefore
    measured on the seed frames, and the threshold is the smallest grid value whose
    median count of noise peaks per quiet frame is at most ``target_noise_peaks``.
    See docs/overview/peakfinding.md.

    Parameters
    ----------
    noise_model : NoiseModel
        Model calibrated by :meth:`CalibrationResult.apply`.
    seed_frames : Iterable[Tensor]
        A sequence of ``(H, W)`` frames; any other iterable is copied into a list.
    target_noise_peaks : float
        Noise budget, in local maxima above the threshold per quiet frame.
    quiet_quantile : float
        Fraction of frames with the lowest ``max(T)`` taken as quiet. Used only when
        fewer than 4 frames pass the null cut and the mirrored tail ``-min_s T_s``
        already meets the budget at the lowest grid value.
    mf_scales : tuple[float, ...]
        Gaussian template widths in pixels.
    local_inner_radius, local_outer_radius : int
        Inner and outer half-widths, in pixels, of the square per-frame background
        annulus.
    size_min : int
        Pixels above the threshold required in the 3 x 3 neighbourhood of a counted
        maximum.
    threshold_grid_min, threshold_grid_max, threshold_grid_step : float
        Range and step of the threshold sweep.
    body_subsample : int
        Cap on the pooled ``T`` values for the empirical null.
    flux_variance : bool
        Whiten against the flux-scaled variance from :func:`fit_photon_transfer`.
    flux_var_floor : float
        Lower bound on the flux-scaled variance, as a fraction of the running
        variance.
    rng_seed : int
        Seed for the ``T`` subsample.
    finder : PeakFinder, optional
        Receives the result through :meth:`ThresholdCalibration.apply`.

    Returns
    -------
    ThresholdCalibration
        ``mu0``, ``sigma0``, the chosen ``threshold``, and diagnostics. Already
        applied if ``finder`` was given.
    """
    frames = seed_frames if isinstance(seed_frames, Sized) else list(seed_frames)
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("seed_frames is empty")
    if not 0.0 < quiet_quantile <= 1.0:
        raise ValueError("quiet_quantile must be in (0, 1]")
    if target_noise_peaks < 0:
        raise ValueError("target_noise_peaks must be >= 0")
    if threshold_grid_min >= threshold_grid_max:
        raise ValueError("threshold_grid_min must be < threshold_grid_max")

    pred = noise_model.predict(combination="calibrated")
    mu = pred["mean"]
    var = pred["var"].clamp_min(1e-12)
    mask = pred["mask"]
    device = mu.device
    dtype = mu.dtype

    kernels = [
        gaussian_kernel_2d(
            2 * int(math.ceil(3.0 * float(s))) + 1,
            float(s),
            device=device,
            dtype=dtype,
        )
        for s in mf_scales
    ]

    n_steps = (
        int(round((threshold_grid_max - threshold_grid_min) / threshold_grid_step)) + 1
    )
    tgrid = torch.linspace(
        threshold_grid_min, threshold_grid_max, n_steps, device=device, dtype=dtype
    )
    thr_list = tgrid.tolist()

    # keep max(T), a T subsample and the counts per frame; each T map is then freed
    t_max_per_frame: list[float] = []
    body_pieces: list[Tensor] = []
    counts: list[list[int]] = []
    mirrored: list[list[int]] = []
    per_frame_cap = max(1, body_subsample // n_frames)

    def blob_counts(T: Tensor) -> list[int]:
        # per threshold: 3 x 3 local maxima above it, >= size_min of the 9 above too
        pooled = F.max_pool2d(T[None, None], kernel_size=3, stride=1, padding=1)
        is_max = (T >= pooled[0, 0]) & mask
        row: list[Tensor] = []
        for thr in thr_list:
            above = (T > thr) & mask
            peaks = is_max & above
            if size_min > 1:
                local_size = 9.0 * F.avg_pool2d(
                    above[None, None].to(dtype), kernel_size=3, stride=1, padding=1
                )
                peaks = peaks & (local_size[0, 0] >= float(size_min))
            row.append(peaks.sum())
        return torch.stack(row).tolist()

    gen = torch.Generator(device="cpu").manual_seed(rng_seed)
    for f in frames:
        f = f.to(dtype=dtype, device=device)
        lm, lv = local_mean_var(
            f - mu,
            mask,
            local_inner_radius,
            local_outer_radius,
            clip_hi=LOCAL_BG_CLIP_K * var.sqrt(),
        )
        mean_eff = mu + lm
        if flux_variance:
            var_eff = flux_variance_floor(
                noise_model, mean_eff, lv, var, floor_coef=flux_var_floor
            )
        else:
            var_eff = torch.maximum(var, lv)
        z = torch.where(mask, (f - mean_eff) / var_eff.sqrt(), torch.zeros_like(f))
        T: Optional[Tensor] = None
        T_min: Optional[Tensor] = None
        for k in kernels:
            t = matched_filter_z(z, k, mask)
            T = t if T is None else torch.maximum(T, t)
            T_min = t if T_min is None else torch.minimum(T_min, t)
        assert T is not None and T_min is not None
        flat = T[mask]
        t_max_per_frame.append(float(flat.max()))
        if flat.numel() > per_frame_cap:
            idx = torch.randperm(flat.numel(), generator=gen)[:per_frame_cap]
            body_pieces.append(flat[idx.to(device)].cpu())
        else:
            body_pieces.append(flat.cpu())
        counts.append(blob_counts(T))
        mirrored.append(blob_counts(-T_min))

    t_body = torch.cat(body_pieces)
    mu0, sigma0 = _empirical_null_central_matching(t_body)

    tmax_t = torch.tensor(t_max_per_frame)
    # quiet frames: max(T) below the null extreme mu0 + sigma0*sqrt(2 ln N)
    n_valid = int(mask.sum())
    z_extreme = math.sqrt(2.0 * math.log(max(n_valid, 2)))
    quiet_max_T = mu0 + sigma0 * z_extreme
    quiet_idx = [i for i, m in enumerate(t_max_per_frame) if m <= quiet_max_T]
    if len(quiet_idx) < 4:
        # too few quiet frames (hit-only input): count noise on the mirrored tail
        counts_t = torch.tensor(mirrored, dtype=torch.float32).median(dim=0).values
        quiet_idx = []
        if float(counts_t[0]) <= float(target_noise_peaks):  # tail too sparse to use
            order = torch.argsort(tmax_t).tolist()
            quiet_idx = order[: max(4, int(round(quiet_quantile * n_frames)))]
            quiet_max_T = float(tmax_t[quiet_idx[-1]])
    if quiet_idx:
        counts_t = (
            torch.tensor(counts, dtype=torch.float32)[quiet_idx].median(dim=0).values
        )

    ok = counts_t <= float(target_noise_peaks)
    if bool(ok.any()):
        idx = int(torch.nonzero(ok, as_tuple=False)[0])
    else:
        idx = n_steps - 1
        warnings.warn(
            f"detection threshold capped at {threshold_grid_max:g}: "
            f"{float(counts_t[-1]):g} noise blobs per frame remain above it",
            stacklevel=2,
        )

    threshold = float(tgrid[idx])
    achieved = float(counts_t[idx])

    result = ThresholdCalibration(
        mu0=mu0,
        sigma0=sigma0,
        threshold=threshold,
        target_noise_peaks=float(target_noise_peaks),
        achieved_noise_peaks=achieved,
        n_quiet_frames=len(quiet_idx),
        quiet_max_T=quiet_max_T,
        mf_scales=tuple(float(s) for s in mf_scales),
    )
    if finder is not None:
        result.apply(finder)
    return result
