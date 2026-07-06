from __future__ import annotations

from typing import Iterable, Literal, Optional

import torch
from torch import Tensor, nn

from ._drift import DriftDiagnostics


# NOISE STATES MOTHER ===================================================
class NoiseStats(nn.Module):
    """Streaming mean/variance base class

    Attributes:
        mean_: Running mean, shape ``stat_shape``.
        M2_: Welford sum of squared deviations (``decay == 1``) or EWMA variance
            directly (``decay < 1``).
        n_: Frames seen since the last reset.
    """

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
            # Welford: mean += delta/n; M2 += delta*(x - mean_new). var divides by n-1.
            delta = x - self.mean_
            self.mean_.add_(delta / self.n_)
            self.M2_.add_(delta * (x - self.mean_))
        else:
            # EWMA with alpha = decay; M2 holds the variance estimate directly.
            delta = x - self.mean_
            self.mean_.add_(self.decay * delta)
            self.M2_.mul_(1.0 - self.decay).add_(
                (1.0 - self.decay) * self.decay * delta * delta
            )

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
        # Map a coerced frame to stat-shape; identity (full-frame) by default.
        return frame

    def _coerce(self, frame: Tensor) -> Tensor:
        if not isinstance(frame, Tensor):
            frame = torch.as_tensor(frame)
        if tuple(frame.shape) != self.frame_size:
            raise ValueError(
                f"frame shape {tuple(frame.shape)} != frame_size {self.frame_size}"
            )
        return frame.to(self.mean_)


# NOISE MODEL ====================================================

from ._panel import PanelNoise, PanelSpec  # noqa: E402
from ._pixel import PixelNoise  # noqa: E402
from ._radial import RotationalNoise  # noqa: E402


class NoiseModel(nn.Module):
    """Composite noise model: pixel + radial + panel.

    Blends per-pixel, radial (rotational), and panel background estimates into
    a single mean+variance prediction for detection and calibration. With
    ``robust_update`` each frame is upper-clipped at ``mean + robust_k*sigma``
    before folding into the running statistics, so Bragg peaks do not inflate
    the per-pixel variance that detection relies on.

    Parameters
    ----------
    frame_size : tuple[int, int]
        Detector frame shape ``(rows, cols)``.
    mode : {"per_frame", "online"}
        ``online`` keeps running stats; ``per_frame`` resets each frame.
    decay : float
        EWMA weight in ``(0, 1]``; ``1.0`` is an unweighted running estimate.
    beam_center : tuple[float, float], optional
        Radial-bin center; defaults to the geometric frame center.
    radial_bin_width : float
        Width of the radial annuli in pixels.
    panels : PanelSpec, optional
        Panel layout (id map or slice/box specs); ``None`` is one panel.
    warmup_frames : int
        Frames before the dead-pixel mask is committed and shrinkage saturates.
    drift_log_every : int
        Record drift diagnostics every this many frames.
    robust_update : bool
        Upper-clip peaks before updating running stats.
    robust_k : float
        Clip level in sigma for ``robust_update``.
    robust_min_frames : int
        Frames before robust clipping engages.
    valid_mask : Tensor, optional
        Static a-priori bad-pixel mask, applied from the first frame.
    device, dtype
        Tensor placement and precision.

    Attributes
    ----------
    read_var, gain : float or None
        Photon-transfer curve ``var = read_var + gain*level`` (fit later).
    eigen_modes : Tensor or None
        Low-rank background eigen-images ``(n_modes, H, W)`` (fit later).
    var_scale : float
        Multiplier pinning the calibrated background z to unit variance.
    calibrated_weights : dict or None
        Learned convex blend over background-mean sources.
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
        radial_bin_width: float = 2.0,
        panels: Optional[PanelSpec] = None,
        warmup_frames: int = 16,
        drift_log_every: int = 1,
        robust_update: bool = True,
        robust_k: float = 5.0,
        robust_min_frames: int = 4,
        valid_mask: Optional[Tensor] = None,
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

        self.pixel = PixelNoise(
            frame_size=frame_size, mode=mode, decay=decay, device=device, dtype=dtype
        )
        self.rotational = RotationalNoise(
            frame_size=frame_size,
            beam_center=beam_center,
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
        # Static a-priori mask fed to the spatial estimators from the first
        # frame, so bad pixels never poison a radial bin / panel.
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
        # Set by calibrate_noise().apply(); used by the "calibrated" combination.
        self.calibrated_weights: Optional[dict[str, float]] = None
        # None -> update every source; else only these (pixel always updates, as
        # robust clip, mask commit, and drift all read it). Set post-calibration
        # so sources the active prediction never blends aren't computed per frame.
        self.active_update_sources: Optional[set] = None
        self.var_scale: float = 1.0
        self.eigen_modes: Optional[Tensor] = None
        self.read_var: Optional[float] = None
        self.gain: Optional[float] = None

    @torch.no_grad()
    def _robust_clip(self, frame: Tensor) -> Tensor:
        # Upper-clip at mean + k*sigma so peaks cannot poison the stats used to detect them.
        if not self.robust_update or self.mode != "online":
            return frame
        if self._n_host < self.robust_min_frames:
            return frame
        frame = frame.to(self.pixel.mean_)
        std = self.pixel.std()
        cap = self.pixel.mean() + self.robust_k * std
        # leave pixels with no spread yet (std == 0) untouched
        return torch.where(std > 0, torch.minimum(frame, cap), frame)

    @torch.no_grad()
    def update(self, frame: Tensor) -> None:
        if self.mode == "per_frame":
            self._soft_reset()
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
        # Restrict per-frame updates to `sources` (plus pixel, always required).
        # Pass None to restore updating every source (e.g. before recalibration).
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
        """Fit over ``frames`` in batches and write a diagnostic GIF.

        Renders the running mean background, its radial profile, and the
        per-batch drift as an animation over batches.

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
            # Linear shrinkage from radial prior toward per-pixel as evidence
            # accrues: w = min(1, n/warmup); mean = w*pixel + (1-w)*rotational.
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
        """Blend pixel/rotational/panel sources into a single mean+var prediction.

        Parameters
        ----------
        combination : {"pixel", "rotational", "panel", "shrinkage", "calibrated"}
            Named preset for the mean blend and variance source.
        weights : dict, optional
            Custom convex weights over sources; overrides ``combination``'s blend.
        var_source : str, optional
            Source whose variance to use; defaults to the preset's choice.

        Returns
        -------
        dict
            ``{"mean": Tensor, "var": Tensor, "mask": Tensor}``.
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
        # z = (x - mu) / sigma
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
            # per-pixel KL( N(mu,s^2) || N(mu_p,s_p^2) ), averaged over valid pixels.
            kl_map = 0.5 * (
                torch.log(prev_var / var)
                + (var + (mean - prev_mean) ** 2) / prev_var
                - 1.0
            )
            kl = (kl_map * valid).sum() / denom

        ms, vr, klv, nm = (
            torch.stack([mean_shift, var_ratio, kl, n_masked_t.to(mean_shift.dtype)])
            .cpu()
            .to(torch.float64)
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
