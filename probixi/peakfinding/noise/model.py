from __future__ import annotations

from typing import Iterable, Literal, Optional

import torch
from torch import Tensor, nn

from ...kernels import select
from ._drift import DriftDiagnostics

# CrystFEL flag_* keys mapped to the test a good pixel passes
FLAG_TESTS = {
    "flag_lessthan": torch.ge,
    "flag_morethan": torch.le,
    "flag_equal": torch.ne,
}


def value_ok(frame: Tensor, flags: Iterable[tuple]) -> Tensor:
    ok = torch.ones_like(frame, dtype=torch.bool)
    for rows, cols, key, value in flags:
        ok[..., rows, cols] &= FLAG_TESTS[key](frame[..., rows, cols], value)
    return ok


class NoiseStats(nn.Module):
    # Streaming base; M2_ is the Welford sum of squares at decay 1, else EWMA variance

    mean_: Tensor
    M2_: Tensor
    n_: Tensor

    def __init__(
        self,
        stat_shape,
        mode: Literal["per_frame", "online"],
        decay: float,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ):
        super().__init__()
        if mode not in ("online", "per_frame"):
            raise ValueError("mode must be 'online' or 'per_frame'")
        self.mode = mode
        self.decay = float(decay)
        self.register_buffer(
            "mean_", torch.zeros(stat_shape, dtype=dtype, device=device)
        )
        self.register_buffer("M2_", torch.zeros(stat_shape, dtype=dtype, device=device))
        self.register_buffer("n_", torch.zeros((), dtype=torch.long, device=device))

    @torch.no_grad()
    def update(self, frame: Tensor, mask: Optional[Tensor] = None) -> None:
        if self.mode == "per_frame":
            self.reset()
        x = self._project(self._coerce(frame), mask)
        self.n_ += 1
        if self.decay == 1.0:
            kernel = select(
                "noise",
                x.is_cuda
                and x.dtype == self.mean_.dtype == torch.float32
                and x.shape == self.mean_.shape
                and x.is_contiguous()
                and self.mean_.is_contiguous()
                and self.M2_.is_contiguous(),
            )
            if kernel is not None:
                kernel.update_stat(self, x)
                return
            # Welford (1962): delta * (x - new mean) avoids float32 cancellation
            delta = x - self.mean_
            self.mean_.add_(delta / self.n_)
            self.M2_.add_(delta * (x - self.mean_))
        else:
            # EWMA with alpha = max(decay, 1/n): a plain running mean until 1/n < decay
            a = (1.0 / self.n_.to(self.mean_.dtype)).clamp_min(self.decay)
            delta = x - self.mean_
            self.mean_.add_(a * delta)
            self.M2_.mul_(1.0 - a).add_((1.0 - a) * a * delta * delta)

    def reset(self) -> None:
        self.mean_.zero_()
        self.M2_.zero_()
        self.n_.zero_()

    def mean(self) -> Tensor:
        return self.mean_

    def var(self) -> Tensor:
        if self.decay == 1.0:
            # unbiased variance M2/(n-1); zero until two observations
            return torch.where(
                self.n_ >= 2,
                self.M2_ / (self.n_ - 1).clamp_min(1),
                torch.zeros_like(self.mean_),
            )
        return self.M2_

    def std(self) -> Tensor:
        return self.var().clamp_min(0.0).sqrt()

    def _project(self, frame: Tensor, mask: Optional[Tensor]) -> Tensor:
        # Subclasses reduce the frame to stat_shape (bins, panels)
        return frame

    def _coerce(self, frame: Tensor) -> Tensor:
        if not isinstance(frame, Tensor):
            frame = torch.as_tensor(frame)
        if tuple(frame.shape) != self.frame_size:
            raise ValueError(
                f"frame shape {tuple(frame.shape)} != frame_size {self.frame_size}"
            )
        return frame.to(self.mean_)


from ._panel import PanelNoise, PanelSpec  # noqa: E402
from ._pixel import PixelNoise  # noqa: E402
from ._radial import RotationalNoise  # noqa: E402


class NoiseModel(nn.Module):
    """Running background model blending pixel, radial, and panel estimators.

    Each estimator keeps a running mean and variance over the frame stream, and
    :meth:`predict` blends their means. With ``robust_update``, each frame is clipped
    from above at ``mean + robust_k * sigma`` before it updates the statistics, which
    keeps Bragg peaks out of the variance that detection tests against. See
    docs/overview/peakfinding.md.

    Parameters
    ----------
    frame_size : tuple[int, int]
        Frame shape ``(rows, cols)``.
    mode : {"online", "per_frame"}
        ``online`` accumulates over the stream; ``per_frame`` resets before every frame.
    decay : float
        Forgetting weight in ``(0, 1]``, a memory of about ``1 / decay`` frames;
        ``1.0`` averages the whole stream.
    beam_center : tuple[float, float], optional
        Centre ``(row, col)`` of the radial annuli in pixels; the frame centre by
        default. Ignored when ``radial_radius`` is given.
    radial_radius : Tensor, optional
        ``(rows, cols)`` lab-frame distance of each pixel from the beam, in pixels.
    radial_bin_width : float
        Width of the radial annuli in pixels.
    panels : PanelSpec, optional
        ``(rows, cols)`` panel id map with -1 outside every panel, or one
        ``(row_slice, col_slice)`` or ``(r0, r1, c0, c1)`` per panel; ``None`` is one
        panel spanning the frame.
    warmup_frames : int
        Frames before the dead-pixel mask is committed and the ``shrinkage`` blend
        reaches the pixel mean.
    drift_log_every : int
        Frames between drift records.
    robust_update : bool
        Clip each frame from above before it updates the statistics; online mode only.
    robust_k : float
        Clip level in standard deviations.
    robust_min_frames : int
        Frames before the clip engages.
    valid_mask : Tensor, optional
        ``(rows, cols)`` static mask, True where a pixel is usable; applied from the
        first frame.
    value_flags : iterable of tuple, optional
        CrystFEL ``flag_lessthan``, ``flag_morethan``, and ``flag_equal`` tests as
        ``(rows, cols, key, value)``. A pixel that fails one on a frame takes its
        running mean on that frame.
    device, dtype
        Placement and precision of the statistics.

    Attributes
    ----------
    read_var, gain : float or None
        Photon-transfer fit ``var = read_var + gain * level``, set by
        :func:`fit_photon_transfer`.
    eigen_modes : Tensor or None
        ``(n_modes, H, W)`` background eigen-images, set by
        :func:`fit_eigen_background`.
    var_scale : float
        Scale on the per-pixel standard deviation that gives the calibrated
        background unit variance.
    calibrated_weights : dict or None
        Convex weights over ``pixel``, ``rotational``, and ``panel``, set by
        :meth:`CalibrationResult.apply`.
    """

    n_frames: Tensor
    valid_mask: Tensor
    static_mask: Tensor

    def __init__(
        self,
        frame_size: tuple[int, int],
        mode: Literal["per_frame", "online"] = "online",
        decay: float = 1.0,
        beam_center: Optional[tuple[float, float]] = None,
        radial_radius: Optional[Tensor] = None,
        radial_bin_width: float = 2.0,
        panels: Optional[PanelSpec] = None,
        warmup_frames: int = 16,
        drift_log_every: int = 1,
        robust_update: bool = True,
        robust_k: float = 5.0,
        robust_min_frames: int = 4,
        valid_mask: Optional[Tensor] = None,
        value_flags: Iterable[tuple] = (),
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if mode not in ("online", "per_frame"):
            raise ValueError("mode must be 'online' or 'per_frame'")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if warmup_frames < 1 or drift_log_every < 1:
            raise ValueError("warmup_frames and drift_log_every must be >= 1")
        if robust_k <= 0:
            raise ValueError("robust_k must be > 0")

        self.frame_size = tuple(int(s) for s in frame_size)
        self.mode = mode
        self.decay = float(decay)
        self.warmup_frames = int(warmup_frames)
        self.drift_log_every = int(drift_log_every)
        self.robust_update = bool(robust_update)
        self.robust_k = float(robust_k)
        self.robust_min_frames = int(robust_min_frames)
        self.value_flags = list(value_flags)

        self.pixel = PixelNoise(
            frame_size=frame_size, mode=mode, decay=decay, device=device, dtype=dtype
        )
        self.rotational = RotationalNoise(
            frame_size=frame_size,
            beam_center=beam_center,
            radius=radial_radius,
            bin_width=radial_bin_width,
            mode=mode,
            decay=decay,
            device=device,
            dtype=dtype,
        )
        self.panel = PanelNoise(
            frame_size=frame_size,
            panels=panels,
            mode=mode,
            decay=decay,
            device=device,
            dtype=dtype,
        )

        self.register_buffer(
            "n_frames", torch.zeros((), dtype=torch.long, device=device)
        )
        # static mask keeps bad pixels out of the radial and panel means from frame 1
        if valid_mask is not None:
            static = valid_mask.to(dtype=torch.bool, device=device)
            if tuple(static.shape) != self.frame_size:
                raise ValueError(
                    f"valid_mask shape {tuple(static.shape)} != frame_size "
                    f"{self.frame_size}"
                )
        else:
            static = torch.ones(self.frame_size, dtype=torch.bool, device=device)
        self.register_buffer("static_mask", static.clone())
        self.register_buffer("valid_mask", static.clone())

        self.drift = DriftDiagnostics()
        self.record_drift = True
        self._prev_mean: Optional[Tensor] = None
        self._prev_var: Optional[Tensor] = None
        self._mask_committed = False
        # host mirror of n_frames; version bumped whenever valid_mask changes
        self._n_host = 0
        self._mask_version = 0
        self.calibrated_weights: Optional[dict[str, float]] = None
        # None updates every source; pixel always updates (clip, mask, drift read it)
        self.active_update_sources: Optional[set] = None
        self.var_scale: float = 1.0
        self.eigen_modes: Optional[Tensor] = None
        self.read_var: Optional[float] = None
        self.gain: Optional[float] = None

    @torch.no_grad()
    def _robust_clip(self, frame: Tensor) -> Tensor:
        # Upper clip at mean + k * sigma keeps peaks out of the stats that detect them
        if not self.robust_update or self.mode != "online":
            return frame
        if self._n_host < self.robust_min_frames:
            return frame
        frame = frame.to(self.pixel.mean_)
        std = self.pixel.std()
        cap = self.pixel.mean() + self.robust_k * std
        # leave pixels with no spread yet (std == 0) untouched
        return torch.where(std > 0, torch.minimum(frame, cap), frame)

    def _fused_pixel(self, frame: Tensor) -> bool:
        p = self.pixel
        return (
            self.mode == "online"
            and p.decay == 1.0
            and isinstance(frame, Tensor)
            and frame.is_cuda
            and frame.dtype == p.mean_.dtype == torch.float32
            and frame.shape == p.mean_.shape
            and frame.device == p.mean_.device
            and frame.is_contiguous()
            and not frame.requires_grad
            and p.mean_.is_contiguous()
            and p.M2_.is_contiguous()
        )

    @torch.no_grad()
    def clean(self, frame: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        if not self.value_flags:
            return frame, None
        frame = frame.to(self.pixel.mean_)
        ok = value_ok(frame, self.value_flags)
        return torch.where(ok, frame, self.pixel.mean_), ok

    @torch.no_grad()
    def update(self, frame: Tensor) -> None:
        frame = self.clean(frame)[0]
        if self.mode == "per_frame":
            self._soft_reset()
        kernel = select("noise", self._fused_pixel(frame))
        if kernel is not None:
            frame_for_stats = kernel.update_pixel(self, frame)
        else:
            frame_for_stats = self._robust_clip(frame)
            self.pixel.update(frame_for_stats)
        feed_mask = self.valid_mask
        active = self.active_update_sources
        if active is None or "rotational" in active:
            self.rotational.update(frame_for_stats, mask=feed_mask)
        if active is None or "panel" in active:
            self.panel.update(frame_for_stats, mask=feed_mask)
        self.n_frames += 1
        self._n_host += 1
        n = self._n_host
        if (
            self.mode == "online"
            and not self._mask_committed
            and n >= self.warmup_frames
        ):
            self._commit_mask()
        if (
            self.record_drift
            and self.mode == "online"
            and n % self.drift_log_every == 0
        ):
            self._record_drift(n)

    def set_active_update_sources(self, sources: Optional[Iterable[str]]) -> None:
        """Update only ``sources`` plus ``pixel`` per frame; ``None`` updates all."""
        if sources is None:
            self.active_update_sources = None
        else:
            self.active_update_sources = set(sources) | {"pixel"}

    def fit(self, frames: Iterable[Tensor]) -> "NoiseModel":
        for item in frames:
            if isinstance(item, Tensor) and item.ndim == 3:
                for frame in item:
                    self.update(frame)
            else:
                self.update(item)
        return self

    def diagnostics(self, frames: Iterable[Tensor], path, **kwargs):
        """Update the model on ``frames`` and write an animated GIF of its state.

        Each step of the animation follows one batch of frames and shows the standard
        error and mean of the background, its radial profile, and the mean absolute
        change of the pixel mean per batch.

        Parameters
        ----------
        frames : iterable of Tensor
            ``(H, W)`` frames or ``(B, H, W)`` batches; every frame updates the model.
        path : str or Path
            Output ``.gif``; parent directories are created.
        **kwargs
            ``batch_size`` (frames per step, default 15), ``fps``, ``dpi``,
            ``max_radius``, ``figsize``, and ``assemble`` with ``assembled_beam`` to
            draw an assembled detector image.

        Returns
        -------
        Path
            The written ``.gif`` path.
        """
        from ._diagnostics import animate_noise_diagnostics

        return animate_noise_diagnostics(self, frames, path, **kwargs)

    def _preset_weights(
        self,
        combination: Literal["pixel", "rotational", "panel", "shrinkage", "calibrated"],
    ) -> tuple[dict[str, float], str]:
        # Return (mean_weights, var_source) for a named preset.
        if combination == "pixel":
            return {"pixel": 1.0}, "pixel"
        if combination == "rotational":
            return {"rotational": 1.0}, "rotational"
        if combination == "panel":
            return {"panel": 1.0}, "panel"
        if combination == "shrinkage":
            # linear shrinkage from the radial mean to the pixel mean over warmup
            n = float(self._n_host)
            w = min(1.0, n / max(1.0, float(self.warmup_frames)))
            return {"pixel": w, "rotational": 1.0 - w}, "pixel"
        if combination == "calibrated":
            if self.calibrated_weights is None:
                raise ValueError("model not calibrated; call calibrate() first")
            return dict(self.calibrated_weights), "pixel"
        raise ValueError(f"unknown combination: {combination!r}")

    @torch.no_grad()
    def predict(
        self,
        combination: Literal[
            "pixel", "rotational", "panel", "shrinkage", "calibrated"
        ] = "shrinkage",
        weights: Optional[dict[str, float]] = None,
        var_source: Optional[str] = None,
    ) -> dict[str, Tensor]:
        """Return the blended background mean and variance.

        Parameters
        ----------
        combination : {"pixel", "rotational", "panel", "shrinkage", "calibrated"}
            Preset for the mean blend and the variance source.
        weights : dict, optional
            Weights over ``pixel``, ``rotational``, and ``panel``, normalised to sum
            to 1; replaces the preset blend.
        var_source : str, optional
            Source of the variance; the preset's source by default.

        Returns
        -------
        dict
            ``mean``, ``var``, and ``mask``, each ``(rows, cols)``. Under the
            ``calibrated`` preset the variance includes ``var_scale**2``.
        """
        sources = {
            "pixel": self.pixel,
            "rotational": self.rotational,
            "panel": self.panel,
        }
        use_preset = weights is None
        if use_preset:
            weights, preset_var = self._preset_weights(combination)
        else:
            preset_var = "pixel"
        unknown = set(weights) - sources.keys()
        if unknown:
            raise ValueError(f"unknown weight keys: {sorted(unknown)}")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("weights must sum to > 0")
        chosen_var = var_source or preset_var
        if chosen_var not in sources:
            raise ValueError(f"unknown var_source: {chosen_var!r}")

        p = self.pixel
        kernel = select(
            "noise",
            chosen_var == "pixel"
            and p.decay == 1.0
            and p.mean_.is_cuda
            and p.mean_.dtype == torch.float32
            and p.mean_.is_contiguous()
            and p.M2_.is_contiguous(),
        )
        if kernel is not None:
            mean, var = kernel.predict_maps(
                self, weights, total, combination == "calibrated" and use_preset
            )
            return {"mean": mean, "var": var, "mask": self.valid_mask}

        contributions = [
            (w / total) * sources[k].mean() for k, w in weights.items() if w > 0
        ]
        mean = (
            contributions[0]
            if len(contributions) == 1
            else torch.stack(contributions).sum(dim=0)
        )
        var = sources[chosen_var].var()
        if combination == "calibrated" and use_preset:
            var = var * (self.var_scale**2)
        return {"mean": mean, "var": var, "mask": self.valid_mask}

    @torch.no_grad()
    def z_score(
        self,
        frame: Tensor,
        combination: Literal[
            "pixel", "rotational", "panel", "shrinkage", "calibrated"
        ] = "shrinkage",
        weights: Optional[dict[str, float]] = None,
        var_source: Optional[str] = None,
    ) -> Tensor:
        frame = self._coerce(frame)
        pred = self.predict(
            combination=combination, weights=weights, var_source=var_source
        )
        std = pred["var"].clamp_min(1e-12).sqrt()
        return torch.where(self.valid_mask, (frame - pred["mean"]) / std, 0.0)

    def reset(self) -> None:
        self.pixel.reset()
        self.rotational.reset()
        self.panel.reset()
        self.n_frames.zero_()
        self._n_host = 0
        self.valid_mask.copy_(self.static_mask)
        self._mask_version += 1
        self.drift = DriftDiagnostics()
        self._prev_mean = None
        self._prev_var = None
        self._mask_committed = False

    def _soft_reset(self) -> None:
        self.pixel.reset()
        self.rotational.reset()
        self.panel.reset()
        self.n_frames.zero_()
        self._n_host = 0

    @torch.no_grad()
    def _commit_mask(self) -> None:
        # Combine the warmup dead-pixel mask with the static a-priori mask.
        self.pixel.commit_dead_pixel_mask(tol=0.0)
        self.valid_mask.copy_(self.pixel.valid_mask & self.static_mask)
        self._mask_version += 1
        self._mask_committed = True

    @torch.no_grad()
    def _record_drift(self, step: int) -> None:
        mean = self.pixel.mean()
        var = self.pixel.var().clamp_min(1e-12)
        n_masked_t = (~self.valid_mask).sum()
        if self._prev_mean is None or self._prev_var is None:
            zero = mean.new_zeros(())
            mean_shift = var_ratio = kl = zero
        else:
            prev_mean, prev_var = self._prev_mean, self._prev_var.clamp_min(1e-12)
            valid = self.valid_mask
            denom = valid.sum().clamp_min(1).to(mean.dtype)
            mean_shift = (((mean - prev_mean).abs()) * valid).sum() / denom
            var_ratio = (torch.log(var / prev_var) * valid).sum() / denom
            # per-pixel KL(N(mu, s^2) || N(mu_p, s_p^2)), averaged over valid pixels
            kl_map = 0.5 * (
                torch.log(prev_var / var)
                + (var + (mean - prev_mean) ** 2) / prev_var
                - 1.0
            )
            kl = (kl_map * valid).sum() / denom

        ms, vr, klv, nm = (
            torch.stack([mean_shift, var_ratio, kl, n_masked_t.to(mean_shift.dtype)])
            .cpu()
            .tolist()
        )
        self.drift.step.append(step)
        self.drift.mean_shift.append(ms)
        self.drift.var_ratio_log.append(vr)
        self.drift.kl_gaussian.append(klv)
        self.drift.effective_n.append(
            float(step) if self.decay == 1.0 else 1.0 / self.decay
        )
        self.drift.n_masked.append(int(nm))
        self._prev_mean = mean.detach().clone()
        self._prev_var = var.detach().clone()

    def _coerce(self, frame: Tensor) -> Tensor:
        if not isinstance(frame, Tensor):
            frame = torch.as_tensor(frame)
        if tuple(frame.shape) != self.frame_size:
            raise ValueError(
                f"frame shape {tuple(frame.shape)} != frame_size {self.frame_size}"
            )
        return frame.to(self.pixel.mean_)
