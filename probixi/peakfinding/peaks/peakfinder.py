from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Callable, Iterable, Iterator, Literal, Optional

import torch
from torch import Tensor

from ...kernels import select, tensor_key
from ..noise.model import NoiseModel
from .blobs import (
    BlobStats,
    compute_blob_stats,
    empty_stats,
    filter_blobs,
    label_connected_components,
    mask_touch_counts,
    select_blobs,
)
from .neighborhood import (
    LOCAL_BG_CLIP_K,
    annulus_count,
    gaussian_kernel_2d,
    local_mean_var,
    mask_denominator,
    matched_filter_denominator,
    matched_filter_z,
    smooth_logits,
    smooth_logits_batch,
)

_BLOB_MAPS = ("excess", "z", "log_bf", "posterior", "var_eff", "mean_eff")


@dataclass
class Peak:
    """A single detected peak, reduced to host-side scalars.

    Parameters
    ----------
    row, col : float
        Centroid row and column in pixels, weighted by the positive excess.
    intensity : float
        Summed excess over the background, in detector units.
    sigma : float
        Standard uncertainty of ``intensity``, the root of the summed pixel variance.
    max_intensity : float
        Largest pixel excess.
    z_max : float
        Largest whitened residual.
    log_bf_sum : float
        Summed per-pixel log Bayes factor.
    posterior_mean : float
        Mean peak posterior over the pixels.
    size : int
        Pixels in the blob.
    bbox : tuple of int
        Bounding box ``(r0, r1, c0, c1)``, half-open on the upper bounds.
    eccentricity : float
        Ratio of the larger to the smaller second-moment eigenvalue, >= 1.
    peakedness : float
        ``max_intensity`` over the mean pixel excess.
    frame_index : int, optional
        Index of the source frame, if known.
    label_id : int, optional
        Component label in :attr:`PeakResult.labels`.
    """

    row: float
    col: float
    intensity: float
    sigma: float
    max_intensity: float
    z_max: float
    log_bf_sum: float
    posterior_mean: float
    size: int
    bbox: tuple[int, int, int, int]
    eccentricity: float
    peakedness: float
    frame_index: Optional[int] = None
    label_id: Optional[int] = field(default=None, repr=False)


@dataclass
class PeakResult:
    """Device-resident result of finding peaks in one frame.

    Parameters
    ----------
    frame_index : int, optional
        Index of the source frame, if known.
    scores : dict of str to Tensor
        ``(H, W)`` maps from :meth:`PeakFinder.score`; empty after
        :meth:`drop_scores`.
    labels : Tensor
        ``(H, W)`` connected-component labels, 0 on background.
    stats : BlobStats
        Statistics of every labelled component, kept or not.
    keep : Tensor
        Boolean mask over ``stats`` rows that passed the shape filter.
    var : Tensor, optional
        ``(H, W)`` variance the frame was whitened against; integration propagates
        uncertainties from it.
    valid_mask : Tensor, optional
        ``(H, W)`` validity mask of this frame, including its value flags.
    mean : Tensor, optional
        ``(H, W)`` background subtracted to form ``excess``.
    """

    frame_index: Optional[int]
    scores: dict[str, Tensor]
    labels: Tensor
    stats: BlobStats
    keep: Tensor
    var: Optional[Tensor] = None
    valid_mask: Optional[Tensor] = None
    mean: Optional[Tensor] = None

    @cached_property
    def kept_stats(self) -> BlobStats:
        return select_blobs(self.stats, self.keep)

    def to_peaks(self) -> list[Peak]:
        """Return the kept blobs as host-side ``Peak`` objects; syncs the device."""
        return _stats_to_peaks(self.kept_stats, frame_index=self.frame_index)

    def drop_scores(self) -> "PeakResult":
        """Return a copy without the ``(H, W)`` score maps."""
        return replace(self, scores={})

    def __len__(self) -> int:
        return len(self.kept_stats)


class PeakStream:
    """Lazy, single-use stream of per-frame :class:`PeakResult` objects.

    ``map``, ``filter``, ``tap``, and ``drop_scores`` return new streams;
    ``collect_peaks``, ``count``, ``for_each``, and iteration consume it. No frame is
    scored until a result is pulled.

    Parameters
    ----------
    source : iterable of PeakResult
        The underlying per-frame results to stream over.
    """

    def __init__(self, source: Iterable[PeakResult]):
        self._source: Iterator[PeakResult] = iter(source)

    def map(self, fn: Callable[[PeakResult], PeakResult]) -> "PeakStream":
        """Return a stream of ``fn`` applied to each result."""
        return PeakStream(fn(r) for r in self._source)

    def filter(self, predicate: Callable[[PeakResult], bool]) -> "PeakStream":
        """Keep only results for which ``predicate`` is true."""
        return PeakStream(r for r in self._source if predicate(r))

    def tap(self, fn: Callable[[PeakResult], None]) -> "PeakStream":
        """Run a side effect on each result, yielding it unchanged."""

        def _gen() -> Iterator[PeakResult]:
            for r in self._source:
                fn(r)
                yield r

        return PeakStream(_gen())

    def drop_scores(self) -> "PeakStream":
        """Return a stream whose results carry no ``(H, W)`` score maps."""
        return self.map(lambda r: r.drop_scores())

    def collect_peaks(self) -> list[Peak]:
        """Consume the stream and return every kept peak in one list."""
        return [p for r in self._source for p in r.to_peaks()]

    def count(self) -> int:
        """Consume the stream and return the number of results."""
        return sum(1 for _ in self._source)

    def for_each(self, fn: Callable[[PeakResult], None]) -> None:
        """Consume the stream, applying ``fn`` to each result."""
        for r in self._source:
            fn(r)

    def __iter__(self) -> Iterator[PeakResult]:
        return self._source


class PeakFinder:
    """Bayesian peak finder over a calibrated :class:`NoiseModel`.

    Each frame is whitened against the model and scored by the per-pixel log Bayes
    factor of ``N(mu, (kappa sigma)^2)`` against ``N(mu, sigma^2)`` on upward
    excursions. The Gaussian-smoothed log odds give a posterior per pixel. Pixels
    above the threshold, on the posterior or on the matched-filter maximum ``T``,
    are labelled into connected components and filtered on shape. See
    docs/overview/peakfinding.md.

    Parameters
    ----------
    noise_model : NoiseModel
        Model whose prediction each frame is whitened against.
    kappa : float, default 10.0
        Peak contrast, the standard-deviation inflation of the peak hypothesis; > 1.
    prior_peak : float, default 1e-3
        Prior probability that a pixel carries peak signal; in (0, 1).
    kernel_size : int, default 5
        Side, in pixels, of the Gaussian kernel that smooths the log odds; odd.
    kernel_sigma : float, default 1.0
        Width of that kernel in pixels.
    posterior_threshold : float, default 0.5
        Posterior above which a pixel is a candidate; in (0, 1).
    candidate_threshold : float, optional
        Replaces ``posterior_threshold`` when set, to admit weaker spots; in
        ``(0, posterior_threshold]``.
    matched_filter : bool, default False
        Threshold the matched-filter maximum ``T`` instead of the posterior.
    mf_scales : tuple of float, default (1.0, 1.6, 2.4)
        Gaussian template widths in pixels.
    mf_threshold : float, default 5.0
        Threshold on ``T``, normally set by :func:`calibrate_threshold`.
    size_min, size_max : int, default 2, 30
        Bounds on the pixels per blob. With the matched filter, a bright blob may
        exceed ``size_max`` up to its footprint cap.
    eccentricity_max : float, default 5.0
        Largest second-moment eigenvalue ratio, which rejects streaks.
    peakedness_min : float, default 1.2
        Smallest ratio of maximum to mean excess, which rejects flat regions.
    connectivity : int, default 2
        1 for the 4-neighbourhood, 2 for the 8-neighbourhood.
    combination : str, default 'shrinkage'
        Preset passed to :meth:`NoiseModel.predict`; :meth:`CalibrationResult.apply`
        sets ``calibrated``.
    local_background : bool, default True
        Correct the background mean and variance per frame from a square annulus
        around each pixel, which follows pedestal shifts and smooth gradients.
    local_inner_radius, local_outer_radius : int, default 4, 9
        Inner and outer half-widths of the annulus in pixels;
        ``1 <= inner < outer``.
    flux_variance : bool, default False
        Whiten against ``read_var + gain * level`` at each frame's own level; takes
        effect once :func:`fit_photon_transfer` has run.
    flux_var_floor : float, default 0.15
        Lower bound on the flux-scaled variance, as a fraction of the running
        variance.
    max_peaks : int, optional
        A frame with more components yields no peaks, since its background model
        has failed. Defaults to one per 1000 pixels, at least 1000.
    compile : bool, default False
        ``torch.compile`` the batched scorer when available.
    """

    def __init__(
        self,
        noise_model: NoiseModel,
        kappa: float = 10.0,
        prior_peak: float = 1e-3,
        kernel_size: int = 5,
        kernel_sigma: float = 1.0,
        posterior_threshold: float = 0.5,
        candidate_threshold: Optional[float] = None,
        matched_filter: bool = False,
        mf_scales: tuple[float, ...] = (1.0, 1.6, 2.4),
        mf_threshold: float = 5.0,
        size_min: int = 2,
        size_max: int = 30,
        eccentricity_max: float = 5.0,
        peakedness_min: float = 1.2,
        connectivity: int = 2,
        combination: Literal[
            "pixel", "rotational", "panel", "shrinkage", "calibrated"
        ] = "shrinkage",
        local_background: bool = True,
        local_inner_radius: int = 4,
        local_outer_radius: int = 9,
        flux_variance: bool = False,
        flux_var_floor: float = 0.15,
        max_peaks: Optional[int] = None,
        compile: bool = False,
    ):
        if kappa <= 1.0:
            raise ValueError("kappa must be > 1")
        if local_background and not 1 <= local_inner_radius < local_outer_radius:
            raise ValueError("require 1 <= local_inner_radius < local_outer_radius")
        if not 0.0 < prior_peak < 1.0:
            raise ValueError("prior_peak must be in (0, 1)")
        if not 0.0 < posterior_threshold < 1.0:
            raise ValueError("posterior_threshold must be in (0, 1)")
        if candidate_threshold is not None and not (
            0.0 < candidate_threshold <= posterior_threshold
        ):
            raise ValueError("candidate_threshold must be in (0, posterior_threshold]")
        if matched_filter and (not mf_scales or any(s <= 0 for s in mf_scales)):
            raise ValueError("mf_scales must be non-empty and positive")

        self.noise = noise_model
        self.kappa = float(kappa)
        self.prior_peak = float(prior_peak)
        self.log_prior_odds = math.log(prior_peak / (1.0 - prior_peak))
        self.kernel_size = int(kernel_size)
        self.kernel_sigma = float(kernel_sigma)
        self.posterior_threshold = float(posterior_threshold)
        self.candidate_threshold = (
            float(candidate_threshold) if candidate_threshold is not None else None
        )
        self.matched_filter = bool(matched_filter)
        self.mf_scales = tuple(float(s) for s in mf_scales)
        self.mf_threshold = float(mf_threshold)
        self.size_min = int(size_min)
        self.size_max = int(size_max)
        self.eccentricity_max = float(eccentricity_max)
        self.peakedness_min = float(peakedness_min)
        self.connectivity = int(connectivity)
        self.combination: Literal[
            "pixel", "rotational", "panel", "shrinkage", "calibrated"
        ] = combination
        self.local_background = bool(local_background)
        self.local_inner_radius = int(local_inner_radius)
        self.local_outer_radius = int(local_outer_radius)
        self.flux_variance = bool(flux_variance)
        self.flux_var_floor = float(flux_var_floor)
        self.max_peaks = (
            int(max_peaks)
            if max_peaks is not None
            else max(1000, noise_model.valid_mask.numel() // 1000)
        )

        self._kernel: Tensor = gaussian_kernel_2d(
            self.kernel_size,
            self.kernel_sigma,
            device=noise_model.valid_mask.device,
            dtype=noise_model.pixel.mean_.dtype,
        )

        self._mf_kernels: list[Tensor] = [
            gaussian_kernel_2d(
                2 * int(math.ceil(3.0 * s)) + 1,
                s,
                device=noise_model.valid_mask.device,
                dtype=noise_model.pixel.mean_.dtype,
            )
            for s in self.mf_scales
        ]
        if compile and hasattr(torch, "compile"):
            self.score_batch = torch.compile(self.score_batch)  # type: ignore[method-assign]

        self._pred_cache: Optional[dict[str, Tensor]] = None
        self._pred_key: Optional[int] = None
        self._den_key: Optional[int] = None
        self._smooth_den: Optional[Tensor] = None
        self._mf_bank_den: Optional[list[Tensor]] = None
        self._annulus_count: Optional[Tensor] = None

    def background_annulus_pixels(self) -> Optional[float]:
        """Nominal pixel count of the local annulus; None without local background."""
        if not self.local_background:
            return None
        outer = 2 * self.local_outer_radius + 1
        inner = 2 * self.local_inner_radius + 1
        return float(outer * outer - inner * inner)

    def _pred(self) -> dict[str, Tensor]:
        # prediction cached per model update
        key = self.noise._n_host
        cache = self._pred_cache
        if self._pred_key != key or cache is None:
            cache = self.noise.predict(combination=self.combination)
            self._pred_cache = cache
            self._pred_key = key
        # denominators and annulus count depend only on the mask
        mkey = self.noise._mask_version
        if self._den_key != mkey or self._smooth_den is None:
            self._smooth_den = mask_denominator(cache["mask"], self._kernel)
            if self.matched_filter:
                self._mf_bank_den = [
                    matched_filter_denominator(cache["mask"], k)
                    for k in self._mf_kernels
                ]
            if self.local_background:
                self._annulus_count = annulus_count(
                    cache["mask"],
                    self.local_inner_radius,
                    self.local_outer_radius,
                    dtype=self._kernel.dtype,
                )
            self._den_key = mkey
        return cache

    @torch.no_grad()
    def _eigen_correction(
        self, residual: Tensor, var0: Tensor, modes: Tensor, mask: Tensor
    ) -> Tensor:
        # project the clipped residual onto the orthonormal eigen modes
        cap = 5.0 * var0.sqrt()
        clipped = torch.minimum(residual, cap) * mask
        if residual.ndim == 2:
            coeffs = (modes * clipped).sum(dim=(-2, -1))  # (r,)
            return torch.einsum("r,rhw->hw", coeffs, modes)
        coeffs = torch.einsum("rhw,bhw->br", modes, clipped)  # (B, r)
        return torch.einsum("br,rhw->bhw", coeffs, modes)

    @torch.no_grad()
    def _effective_background(
        self, frame: Tensor, pred: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        # Per-frame (mean, var) after the eigen, local-annulus, and flux corrections
        mean0 = pred["mean"]
        var0 = pred["var"].clamp_min(1e-12)
        modes = getattr(self.noise, "eigen_modes", None)
        if modes is not None:
            mean0 = mean0 + self._eigen_correction(
                frame - mean0, var0, modes, pred["mask"]
            )
        gain = self.noise.gain
        read_var = self.noise.read_var
        flux = self.flux_variance and gain is not None
        if not self.local_background:
            if flux:
                assert gain is not None and read_var is not None
                # Photon-transfer floor read_var + gain*level
                floor = self.flux_var_floor * var0
                base = read_var + gain * mean0.clamp_min(0.0)
                return mean0, torch.maximum(base, floor).clamp_min(1e-12)
            return mean0, var0
        local_mean, local_var = local_mean_var(
            frame - mean0,
            pred["mask"],
            self.local_inner_radius,
            self.local_outer_radius,
            count=self._annulus_count,
            clip_hi=LOCAL_BG_CLIP_K * var0.sqrt(),
        )
        mean_eff = mean0 + local_mean
        if flux:
            assert gain is not None and read_var is not None
            floor = self.flux_var_floor * var0
            base = read_var + gain * mean_eff.clamp_min(0.0)
            base = torch.maximum(base, floor).clamp_min(1e-12)
            return mean_eff, torch.maximum(base, local_var)
        return mean_eff, torch.maximum(var0, local_var)

    @torch.no_grad()
    def _mf_scale_max(self, z: Tensor, mask: Tensor) -> Tensor:
        zb = z.view(1, 1, *z.shape) if z.ndim == 2 else z.unsqueeze(1)
        zm = zb * mask.to(z).view(1, 1, *mask.shape)
        stat: Optional[Tensor] = None
        dens = self._mf_bank_den or [None] * len(self._mf_kernels)
        for k, den in zip(self._mf_kernels, dens):
            t = matched_filter_z(z, k, mask, den=den, zm=zm)
            stat = t if stat is None else torch.maximum(stat, t)
        assert stat is not None, "matched filter requires at least one kernel scale"
        return stat

    @torch.no_grad()
    def score(self, frame: Tensor) -> dict[str, Tensor]:
        """Return the score maps of one frame.

        ``log_bf`` is zero wherever the excess is not positive, and ``posterior`` is
        the logistic function of the Gaussian-smoothed log odds, ``log_bf`` plus the
        log prior odds.

        Parameters
        ----------
        frame : Tensor
            One ``(H, W)`` frame.

        Returns
        -------
        dict of str to Tensor
            ``(H, W)`` maps ``excess``, ``z``, ``log_bf``, ``posterior``,
            ``mean_eff``, ``var_eff``, and ``mf_max`` with the matched filter.
        """
        supported = (
            frame.is_cuda
            and frame.dtype == torch.float32
            and frame.ndim == 2
            and frame.is_contiguous()
            and not frame.requires_grad
            and self.noise.pixel.mean_.dtype == torch.float32
            and frame.device == self.noise.pixel.mean_.device
            and self.local_background
            and self.matched_filter
            and 1 <= self.local_inner_radius < self.local_outer_radius
            and len(self._mf_kernels) == 3
            and [k.shape[0] for k in self._mf_kernels]
            == sorted(k.shape[0] for k in self._mf_kernels)
            and self.noise.eigen_modes is None
            and min(frame.shape) > self._kernel.shape[-1] // 2
        )
        kernel = select("score", supported)
        if kernel is not None:
            key = tuple(tensor_key(k) for k in [self._kernel, *self._mf_kernels])
            if getattr(self, "_kernel_key", None) != key:
                self._kernel_score = kernel.FusedScorer(self)
                self._kernel_key = key
            return self._kernel_score(frame)
        pred = self._pred()
        frame = frame.to(pred["mean"])
        mean, var = self._effective_background(frame, pred)
        excess = frame - mean
        z = excess / var.sqrt()
        log_bf_raw = -math.log(self.kappa) + 0.5 * z * z * (
            1.0 - 1.0 / (self.kappa * self.kappa)
        )
        log_bf = torch.where(pred["mask"] & (frame > mean), log_bf_raw, 0.0)
        logits = log_bf + self.log_prior_odds
        logits_smoothed = smooth_logits(
            logits, self._kernel, mask=pred["mask"], den=self._smooth_den
        )
        posterior = torch.where(pred["mask"], torch.sigmoid(logits_smoothed), 0.0)
        out = {
            "excess": excess,
            "z": z,
            "log_bf": log_bf,
            "posterior": posterior,
            "mean_eff": mean,
            "var_eff": var,
        }
        if self.matched_filter:
            out["mf_max"] = self._mf_scale_max(z, pred["mask"])
        return out

    @torch.no_grad()
    def score_batch(self, batch: Tensor) -> dict[str, Tensor]:
        # score over a (B, H, W) batch
        pred = self._pred()
        mask = pred["mask"].unsqueeze(0)
        batch = batch.to(pred["mean"])
        mean, var = self._effective_background(batch, pred)
        excess = batch - mean
        z = excess / var.sqrt()
        log_bf_raw = -math.log(self.kappa) + 0.5 * z * z * (
            1.0 - 1.0 / (self.kappa * self.kappa)
        )
        log_bf = torch.where(mask & (batch > mean), log_bf_raw, 0.0)
        logits = log_bf + self.log_prior_odds
        logits_smoothed = smooth_logits_batch(
            logits, self._kernel, mask=pred["mask"], den=self._smooth_den
        )
        posterior = torch.sigmoid(logits_smoothed)
        posterior = torch.where(mask, posterior, 0.0)
        out = {
            "excess": excess,
            "z": z,
            "log_bf": log_bf,
            "posterior": posterior,
            "mean_eff": mean,
            "var_eff": var,
        }
        if self.matched_filter:
            out["mf_max"] = self._mf_scale_max(z, pred["mask"])
        return out

    @torch.no_grad()
    def peak_stream(
        self,
        frames: Iterable[Tensor],
        start_index: int = 0,
    ) -> PeakStream:
        """Return a lazy stream of one :class:`PeakResult` per frame.

        A ``(B, H, W)`` batch is scored together and split per frame. A pixel that
        fails a geometry value flag on a frame is masked for that frame.

        Parameters
        ----------
        frames : iterable of Tensor
            ``(H, W)`` frames or ``(B, H, W)`` batches.
        start_index : int, default 0
            Frame index assigned to the first emitted result.

        Returns
        -------
        PeakStream
            A lazy stream of ``PeakResult``, one per frame.
        """

        def _gen() -> Iterator[PeakResult]:
            offset = 0
            for item in frames:
                item, ok = self.noise.clean(item)
                valid = self.noise.valid_mask
                if item.ndim == 3:
                    scores_batch = self.score_batch(item)
                    for b in range(item.shape[0]):
                        idx = start_index + offset
                        yield self._extract(
                            {k: v[b] for k, v in scores_batch.items()},
                            frame_index=idx,
                            mask=valid if ok is None else valid & ok[b],
                        )
                        offset += 1
                else:
                    idx = start_index + offset
                    yield self._extract(
                        self.score(item),
                        frame_index=idx,
                        mask=valid if ok is None else valid & ok,
                    )
                    offset += 1

        return PeakStream(_gen())

    def _extract(
        self,
        scores: dict[str, Tensor],
        frame_index: Optional[int],
        mask: Tensor,
    ) -> PeakResult:
        device = mask.device
        dtype = scores["excess"].dtype
        if self.matched_filter and "mf_max" in scores:
            # T is the maximum over unit-variance responses T_s of the template bank
            binary = (scores["mf_max"] > self.mf_threshold) & mask
        else:
            thr = (
                self.candidate_threshold
                if self.candidate_threshold is not None
                else self.posterior_threshold
            )
            binary = (scores["posterior"] > thr) & mask
        maps = [scores.get(k) for k in _BLOB_MAPS]
        if self.matched_filter:
            maps.append(scores.get("mf_max"))
        kernel = select(
            "blobs",
            binary.is_cuda
            and binary.is_contiguous()
            and all(
                m is not None
                and m.dtype == torch.float32
                and m.is_contiguous()
                and m.shape == binary.shape
                and m.device == binary.device
                for m in maps
            ),
        )
        if kernel is not None:
            result = kernel.extract(self, binary, scores, frame_index, mask)
            if result is not None:
                touch = mask_touch_counts(result.labels, len(result.stats), mask)
                result.stats.n_masked = touch
                result.kept_stats.n_masked = touch[result.keep]
                return result
        labels, n_blobs = label_connected_components(
            binary, connectivity=self.connectivity
        )
        # beyond max_peaks components the background model has failed: emit none
        if n_blobs == 0 or (self.max_peaks > 0 and n_blobs > self.max_peaks):
            return PeakResult(
                frame_index=frame_index,
                scores=scores,
                labels=labels,
                stats=empty_stats(device, dtype),
                keep=torch.zeros(0, dtype=torch.bool, device=device),
            )
        var_eff = scores.get("var_eff", self._pred()["var"])
        mean_eff = scores.get("mean_eff", self._pred()["mean"])
        stats = compute_blob_stats(
            labels,
            n_blobs,
            excess=scores["excess"],
            z=scores["z"],
            log_bf=scores["log_bf"],
            posterior=scores["posterior"],
            var=var_eff,
            mean=mean_eff,
            response=scores.get("mf_max") if self.matched_filter else None,
        )
        stats.n_masked = mask_touch_counts(labels, n_blobs, mask)
        # with the matched filter, the size cap follows each blob's brightness
        footprint = self.matched_filter and "mf_max" in scores
        keep = filter_blobs(
            stats,
            size_min=self.size_min,
            size_max=self.size_max,
            eccentricity_max=self.eccentricity_max,
            peakedness_min=self.peakedness_min,
            footprint_scale=max(self.mf_scales) if footprint else None,
            threshold=self.mf_threshold if footprint else None,
        )
        return PeakResult(
            frame_index=frame_index,
            scores=scores,
            labels=labels,
            stats=stats,
            keep=keep,
            var=var_eff,
            valid_mask=mask,
            mean=mean_eff,
        )


def _stats_to_peaks(stats: BlobStats, frame_index: Optional[int]) -> list[Peak]:
    n = len(stats)
    if n == 0:
        return []
    cols = (
        stats.label_id.tolist(),
        stats.size.tolist(),
        stats.row_centroid.tolist(),
        stats.col_centroid.tolist(),
        stats.bbox_r0.tolist(),
        stats.bbox_r1.tolist(),
        stats.bbox_c0.tolist(),
        stats.bbox_c1.tolist(),
        stats.intensity_sum.tolist(),
        stats.intensity_sigma.tolist(),
        stats.intensity_max.tolist(),
        stats.z_max.tolist(),
        stats.log_bf_sum.tolist(),
        stats.posterior_mean.tolist(),
        stats.eccentricity.tolist(),
        stats.peakedness.tolist(),
    )
    lid, sz, rc, cc, r0, r1, c0, c1, isum, isig, imax, zmax, lbf, post, ecc, peak = cols
    return [
        Peak(
            row=rc[i],
            col=cc[i],
            intensity=isum[i],
            sigma=isig[i],
            max_intensity=imax[i],
            z_max=zmax[i],
            log_bf_sum=lbf[i],
            posterior_mean=post[i],
            size=int(sz[i]),
            bbox=(int(r0[i]), int(r1[i]), int(c0[i]), int(c1[i])),
            eccentricity=float(ecc[i]),
            peakedness=float(peak[i]),
            frame_index=frame_index,
            label_id=int(lid[i]),
        )
        for i in range(n)
    ]
