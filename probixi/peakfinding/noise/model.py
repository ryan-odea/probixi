from __future__ import annotations

from typing import Iterable, Literal, Optional

import torch
from torch import Tensor, nn


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
            n = int(self.n_)
            delta = x - self.mean_
            self.mean_.add_(delta / n)
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
            n = int(self.n_)
            if n < 2:
                return torch.zeros_like(self.mean_)
            return self.M2_ / (n - 1)
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

