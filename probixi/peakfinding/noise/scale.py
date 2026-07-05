from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class FrameScale:
    """Per-frame relative intensity scale, inferred from the background.

    Attributes
    ----------
    frame_index : int or None
        Index of the source frame.
    scale : float
        Multiplicative scale ``g`` of this frame's background relative to the
        live model's current background (1.0 = same fluence as the model's
        running mean). Robust WLS slope of ``frame ~ offset + g * background``
        over peak-free pixels.
    sigma : float
        1-sigma uncertainty on ``scale`` from the WLS covariance and the
        per-pixel variance model.
    offset : float
        Fitted additive pedestal ``a`` (absorbs any constant offset; ~0 for a
        photon-counting detector).
    """

    frame_index: Optional[int]
    scale: float
    sigma: float
    offset: float


class ScaleReference:
    """Live background reference for per-frame scale inference.

    Wraps a (typically online) ``NoiseModel`` and, on each :meth:`estimate`,
    regresses the frame against the model's *current* background. Because the
    reference follows the live model, the per-frame scale is always measured
    relative to the latest drift-tracked background instead of a snapshot frozen
    at calibration.

    A fixed random subsample of valid pixels is chosen on the first estimate
    (reproducible, for speed); the background mean/variance over those pixels are
    re-read from the live model every call.

    Parameters
    ----------
    noise_model : NoiseModel
        The live model whose background each frame is regressed against.
    combination : str, optional
        Background blend passed to ``noise_model.predict``. ``None`` (default)
        auto-selects ``"calibrated"`` once the model is calibrated, else
        ``"shrinkage"``.
    robust_k : float, default 5.0
        Upper-clip level (in background sigma) to exclude Bragg peaks from the fit.
    subsample : int, default 40000
        Number of valid pixels the fit uses (random, fixed locations).
    rng_seed : int, default 0
        Seed for the pixel subsample (reproducible).
    """

    def __init__(
        self,
        noise_model,
        *,
        combination: Optional[str] = None,
        robust_k: float = 5.0,
        subsample: int = 40000,
        rng_seed: int = 0,
    ):
        self.noise_model = noise_model
        self.combination = combination
        self.robust_k = float(robust_k)
        self.subsample = int(subsample)
        self.rng_seed = int(rng_seed)
        self._idx: Optional[Tensor] = None

    @classmethod
    def from_noise_model(cls, noise_model, **kwargs) -> "ScaleReference":
        """Bind a live reference to ``noise_model`` (kept for call-site clarity)."""
        return cls(noise_model, **kwargs)

    def _combination(self) -> str:
        if self.combination is not None:
            return self.combination
        return (
            "calibrated"
            if self.noise_model.calibrated_weights is not None
            else "shrinkage"
        )

    def _select_idx(self, mask: Tensor) -> Tensor:
        if self._idx is not None:
            return self._idx
        valid = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
        if valid.numel() > self.subsample:
            gen = torch.Generator(device="cpu").manual_seed(self.rng_seed)
            sel = torch.randperm(valid.numel(), generator=gen)[: self.subsample]
            valid = valid[sel.to(valid.device)]
        self._idx = valid
        return valid

    @torch.no_grad()
    def estimate(self, frame: Tensor, frame_index: Optional[int] = None) -> FrameScale:
        """Infer the relative scale of ``frame`` against the live background.

        Parameters
        ----------
        frame : Tensor
            (H, W) raw frame.
        frame_index : int, optional
            Index carried through to the result.

        Returns
        -------
        FrameScale
            The fitted ``(scale, sigma, offset)``.
        """
        pred = self.noise_model.predict(combination=self._combination())
        idx = self._select_idx(pred["mask"])
        t = pred["mean"].flatten()[idx]
        v = pred["var"].clamp_min(1e-12).flatten()[idx]
        sigma = v.sqrt()
        y = frame.to(t).flatten()[idx]
        w = 1.0 / v
        # exclude Bragg peaks from the background fit
        keep = y < (t + self.robust_k * sigma)
        y, t, w = y[keep], t[keep], w[keep]
        sw = w.sum()
        swt = (w * t).sum()
        swtt = (w * t * t).sum()
        swy = (w * y).sum()
        swty = (w * t * y).sum()
        det = (sw * swtt - swt * swt).clamp_min(1e-12)
        g = (sw * swty - swt * swy) / det
        a = (swtt * swy - swt * swty) / det
        var_g = (sw / det).clamp_min(0.0)
        scale, sigma, offset = torch.stack([g, var_g.sqrt(), a]).tolist()
        return FrameScale(
            frame_index=frame_index,
            scale=scale,
            sigma=sigma,
            offset=offset,
        )
