from __future__ import annotations

import weakref
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

_MAX_PENDING = 64


@dataclass
class FrameScale:
    """Per-frame relative intensity scale, inferred from the background.

    Attributes
    ----------
    frame_index : int or None
        Index of the source frame.
    scale : float
        Slope ``g`` of ``frame ~ offset + g * background``, fitted by weighted least
        squares over pixels without Bragg signal; 1.0 is the fluence of the model's
        running mean.
    sigma : float
        Standard uncertainty of ``scale`` from the weighted least-squares covariance.
    offset : float
        Fitted additive pedestal, near 0 on a photon-counting detector.
    """

    frame_index: Optional[int]
    scale: float
    sigma: float
    offset: float


class PendingScale:
    # FrameScale proxy; the fit runs on the first public attribute access
    def __init__(self, ref, frame_index, sample):
        self._ref, self._frame_index, self._sample = ref, frame_index, sample
        self._done = None

    def resolve(self) -> FrameScale:
        if self._done is None:
            self._done = self._ref._fit(*self._sample, self._frame_index)
            self._ref = self._sample = None
        return self._done

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.resolve(), name)


class ScaleReference:
    """Live background reference for per-frame scale inference.

    Each :meth:`estimate` regresses the frame against the model's current
    background, so the scale follows the drift that the running model tracks. The
    pixel subsample is drawn once, on the first call, and the background over it is
    read from the model on every call.

    Parameters
    ----------
    noise_model : NoiseModel
        The live model whose background each frame is regressed against.
    combination : str, optional
        Blend passed to ``noise_model.predict``; by default ``calibrated`` once the
        model is calibrated, else ``shrinkage``.
    robust_k : float, default 5.0
        Pixels above ``mean + robust_k * sigma`` are excluded from the fit.
    subsample : int, default 40000
        Valid pixels drawn at random for the fit.
    rng_seed : int, default 0
        Seed for the pixel subsample.
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

    def __getstate__(self):
        # the weakrefs in _pending cannot be pickled
        return {k: v for k, v in self.__dict__.items() if k != "_pending"}

    @classmethod
    def from_noise_model(cls, noise_model, **kwargs) -> "ScaleReference":
        """Return a reference bound to ``noise_model``, as the constructor does."""
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
        return self._fit(*self._sample(frame, pred), frame_index)

    @torch.no_grad()
    def estimate_later(
        self, frame: Tensor, frame_index: Optional[int] = None, pred=None
    ) -> PendingScale:
        # Fit deferred until read; past _MAX_PENDING fits the oldest resolves now
        if pred is None:
            pred = self.noise_model.predict(combination=self._combination())
        pending = PendingScale(self, frame_index, self._sample(frame, pred))
        queue = getattr(self, "_pending", None)
        if queue is None:
            queue = self._pending = deque()
        queue.append(weakref.ref(pending))
        if len(queue) > _MAX_PENDING:
            oldest = queue.popleft()()
            if oldest is not None:
                oldest.resolve()
        return pending

    def _sample(self, frame: Tensor, pred):
        idx = self._select_idx(pred["mask"])
        t = pred["mean"].flatten()[idx]
        v = pred["var"].flatten()[idx]
        y = frame.to(t).flatten()[idx]
        return t, v, y

    @torch.no_grad()
    def _fit(self, t: Tensor, v: Tensor, y: Tensor, frame_index: Optional[int]):
        v = v.clamp_min(1e-12)
        sigma = v.sqrt()
        w = 1.0 / v
        # exclude Bragg peaks from the background fit
        keep = y < (t + self.robust_k * sigma)
        y, t, w = y[keep], t[keep], w[keep]
        sw = w.sum()
        swt = (w * t).sum()
        swtt = (w * t * t).sum()
        swy = (w * y).sum()
        swty = (w * t * y).sum()
        # closed-form weighted least squares for y = a + g t with weights 1 / v
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
