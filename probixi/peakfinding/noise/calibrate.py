from __future__ import annotations

import math
from dataclasses import dataclass
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
    """Learned noise/detection parameters from :func:`calibrate_noise`.

    Attributes
    ----------
    weights : dict
        Convex blend over background-mean sources (keys among
        ``pixel``/``rotational``/``panel``; zero-weight sources dropped).
    var_scale : float
        Multiplier on the per-pixel std so the background ``z`` has unit
        variance (``var`` is scaled by ``var_scale**2``).
    kappa : float
        Learned peak/noise std inflation factor (> 1).
    prior_peak : float
        Learned prior probability that a pixel is peak.
    background_var : float
        Background variance of ``z`` before ``var_scale`` was applied; ``1.0``
        means the chosen blend was already perfectly calibrated.
    n_pixels : int
        Number of subsampled pixels the fit used.
    """

    weights: dict
    var_scale: float
    kappa: float
    prior_peak: float
    background_var: float
    n_pixels: int

    def apply(self, noise_model, finder=None) -> "CalibrationResult":
        """Push learned parameters onto a noise model (blend + var scale) and,
        optionally, a peak finder (kappa + peak prior)."""
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
    """Calibrate ``noise_model`` on ``seed_frames`` (see module docstring).

    Parameters
    ----------
    noise_model : NoiseModel
        A built model whose sources already cover the frame (warmed here when
        ``warm`` is set).
    seed_frames : Iterable[Tensor]
        Calibration frames (the warmup slice); consumed once.
    warm : bool
        Update the running model on the seed frames first.
    subsample : int
        Number of valid pixels to fit on (random, fixed locations).
    weight_step : float
        Grid spacing for the background-mean blend search.
    em_iters : int
        EM iterations for the peak/noise mixture.
    robust_z : float
        Pixels with ``|z| < robust_z`` define the background set scoring a blend.
    kurtosis_weight : float
        Weight on background non-Gaussianity in the blend loss.
    rng_seed : int
        Seed for the pixel subsample (reproducible).

    Returns
    -------
    CalibrationResult
        The learned blend, variance scale, and peak mixture parameters.
    """
    frames = [f for f in seed_frames]
    if not frames:
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

    # background pixels
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

    The running model freezes one per-pixel variance, measured at the full-dose
    background level; whitening a genuinely low-flux frame against that frozen
    variance over-suppresses its real spots. A solution is a variance scaling with
    each frame's own background level, ``var = read_var + gain*signal``. The fit
    runs on binned medians of var vs mean (raw-pixel OLS near the origin is
    ill-conditioned); a negative intercept triggers a non-negative
    through-origin fallback ``var = gain*level`` (pure photon counter).

    Sets ``noise_model.read_var`` and ``noise_model.gain``. Run after
    :func:`calibrate_noise` ``.apply`` so (mean, var) are calibrated.

    Parameters
    ----------
    noise_model : NoiseModel
        A calibrated model.
    n_bins : int
        Number of mean-quantile bins.
    min_bin_pixels : int
        Minimum pixels per bin to include it.

    Returns
    -------
    tuple[float, float]
        ``(read_var, gain)``.
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


def flux_variance_floor(
    noise_model,
    level: Tensor,
    local_var: Tensor,
    var0: Tensor,
    floor_coef: float = 0.15,
) -> Tensor:
    rv, g = noise_model.read_var, noise_model.gain
    floor = (float(floor_coef) * var0).clamp_min(1e-12)
    if rv is None or g is None:
        return torch.maximum(local_var, floor)
    base = (rv + g * level.clamp_min(0.0)).clamp_min(1e-12)
    return torch.maximum(torch.maximum(base, floor), local_var)


@dataclass
class ThresholdCalibration:
    """Learned matched-filter threshold from the meta-T distribution.

    Attributes
    ----------
    mu0 : float
        Empirical-null mean of the matched-filter T body (scale-max biases it
        slightly positive).
    sigma0 : float
        Empirical-null std of the T body (mildly compressed below 1).
    threshold : float
        Raw matched-filter T cutoff: smallest T* whose median quiet-frame blob
        count is at or below ``target_noise_peaks``.
    target_noise_peaks : float
        Tail-rate target the threshold was solved for (blobs/quiet frame).
    achieved_noise_peaks : float
        Median blob count above ``threshold`` across the quiet seed frames.
    n_quiet_frames : int
        Seed frames classified as quiet (signal-free).
    quiet_max_T : float
        The max(T) cutoff separating quiet from active frames.
    mf_scales : tuple[float, ...]
        Matched-filter kernel scales used (passed to the finder).
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
        """Install the matched-filter bank and threshold on ``finder``.

        Enables ``matched_filter``, sets ``mf_scales``/``mf_threshold``, rebuilds
        the kernel bank, and invalidates the cached MF denominator."""
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
    """Learn the matched-filter detection threshold from the meta-T distribution.

    The per-pixel z is N(0,1) under background after :func:`calibrate_noise`, but
    the multi-scale statistic ``T = max_k MF(z; PSF_k)`` is not: the scale-max
    biases the body's location/scale and spatial correlation fattens the right
    tail. This learns both from the seed frames, then picks the smallest T* whose
    median quiet-frame noise-blob count is at or below ``target_noise_peaks`` which
    yields a tail-rate target rather than a hand-set sigma cut.

    Pre-req: ``noise_model`` is already calibrated (run :func:`calibrate_noise`
    + ``.apply`` first).

    Parameters
    ----------
    noise_model : NoiseModel
        A calibrated model (post ``calibrate_noise.apply``).
    seed_frames : Iterable[Tensor]
        The seed frames (consumed once); 16-32 is plenty.
    target_noise_peaks : float
        Target median blobs per quiet frame above the chosen threshold.
    quiet_quantile : float
        Fallback fraction of lowest-max(T) frames used as the quiet set when the
        null-based absolute cut (mu0 + sigma0*sqrt(2 ln N)) yields too few (< 4)
        quiet frames. The primary quiet/hit split is now that absolute cut, so the
        quiet fraction adapts to the hit rate.
    mf_scales : tuple[float, ...]
        Matched-filter kernel scales (Bragg-spot core sigma range).
    local_inner_radius, local_outer_radius : int
        Annulus radii for the per-frame local-background subtraction.
    size_min : int
        Minimum blob pixel count to keep.
    threshold_grid_min, threshold_grid_max, threshold_grid_step : float
        T* sweep grid.
    body_subsample : int
        Cap on pooled body pixels for the empirical-null fit.
    flux_variance : bool
        Use the photon-transfer variance floor for ``var_eff``.
    flux_var_floor : float
        ``floor_coef`` passed to :func:`flux_variance_floor`.
    rng_seed : int
        Subsample seed (reproducible).
    finder : PeakFinder, optional
        If given, install the result onto it in place.

    Returns
    -------
    ThresholdCalibration
        ``mu0``, ``sigma0``, the chosen ``threshold``, and diagnostics. Already
        applied if ``finder`` was given.
    """
    frames = [f for f in seed_frames]
    if not frames:
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

    # Compute T per frame, mirroring the detector's per-frame local-background
    # correction so the meta distribution matches what the finder sees at run time.
    t_maps: list[Tensor] = []
    t_max_per_frame: list[float] = []
    mask_cpu = mask.cpu()
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
        for k in kernels:
            t = matched_filter_z(z, k, mask)
            T = t if T is None else torch.maximum(T, t)
        assert T is not None
        t_max_per_frame.append(float(T[mask].max()))
        t_maps.append(T.cpu())

    # Pool body pixels across frames (subsampled) for the empirical-null fit.
    body_pieces: list[Tensor] = []
    per_frame_cap = max(1, body_subsample // len(t_maps))
    gen = torch.Generator(device="cpu").manual_seed(rng_seed)
    for T in t_maps:  # T is on the host
        flat = T[mask_cpu]
        if flat.numel() > per_frame_cap:
            idx = torch.randperm(flat.numel(), generator=gen)[:per_frame_cap]
            body_pieces.append(flat[idx])
        else:
            body_pieces.append(flat)
    t_body = torch.cat(body_pieces)
    mu0, sigma0 = _empirical_null_central_matching(t_body)

    tmax_t = torch.tensor(t_max_per_frame)
    # quiet frames: max(T) below the null extreme mu0 + sigma0*sqrt(2 ln N)
    n_valid = int(mask.sum())
    z_extreme = math.sqrt(2.0 * math.log(max(n_valid, 2)))
    quiet_max_T = mu0 + sigma0 * z_extreme
    quiet_idx = [i for i, m in enumerate(t_max_per_frame) if m <= quiet_max_T]
    if len(quiet_idx) < 4:
        order = torch.argsort(tmax_t).tolist()
        quiet_idx = order[: max(4, int(round(quiet_quantile * len(frames))))]
        quiet_max_T = float(tmax_t[quiet_idx[-1]])

    n_steps = (
        int(round((threshold_grid_max - threshold_grid_min) / threshold_grid_step)) + 1
    )
    tgrid = torch.linspace(
        threshold_grid_min, threshold_grid_max, n_steps, device=device, dtype=dtype
    )
    thr_list = tgrid.tolist()
    counts_per_thr: list[list[int]] = [[] for _ in thr_list]
    for i in quiet_idx:
        T = t_maps[i].to(device)
        Tb = T.unsqueeze(0).unsqueeze(0)
        pooled = F.max_pool2d(Tb, kernel_size=3, stride=1, padding=1)
        is_max = (Tb.squeeze() >= pooled.squeeze()) & mask
        for j, thr in enumerate(thr_list):
            above = (T > thr) & mask
            peaks = is_max & above
            if size_min > 1:
                Ab = above.unsqueeze(0).unsqueeze(0).to(dtype)
                local_size = (
                    9.0 * F.avg_pool2d(Ab, kernel_size=3, stride=1, padding=1).squeeze()
                )
                peaks = peaks & (local_size >= float(size_min))
            counts_per_thr[j].append(int(peaks.sum()))
    median_counts: list[float] = [
        float(torch.tensor(c, dtype=torch.float32).median()) for c in counts_per_thr
    ]
    counts_t = torch.tensor(median_counts)

    ok = counts_t <= float(target_noise_peaks)
    if bool(ok.any()):
        idx = int(torch.nonzero(ok, as_tuple=False)[0])
    else:
        idx = n_steps - 1

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
