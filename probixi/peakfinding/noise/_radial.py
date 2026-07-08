from typing import Literal, Optional

import torch
from torch import Tensor

from .model import NoiseStats


class RotationalNoise(NoiseStats):
    """Radial-bin running stats for rotationally-symmetric background.

    Attributes:
        bin_idx: (rows, cols) radial-bin index per pixel.
        pixels_per_bin: Pixels falling in each radial bin.
        valid_mask: (rows, cols) bool mask of usable pixels.
    """

    bin_idx: Tensor
    pixels_per_bin: Tensor
    valid_mask: Tensor

    def __init__(
        self,
        frame_size: tuple[int, int],
        beam_center: Optional[tuple[float, float]] = None,
        bin_width: float = 2.0,
        mode: Literal["per_frame", "online"] = "online",
        decay: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if bin_width <= 0:
            raise ValueError("bin_width must be > 0")
        self.frame_size = tuple(int(s) for s in frame_size)
        self.bin_width = float(bin_width)

        rows, cols = self.frame_size
        if beam_center is None:
            beam_center = ((rows - 1) / 2.0, (cols - 1) / 2.0)
        self.beam_center = (float(beam_center[0]), float(beam_center[1]))

        # bin = floor(radius / bin_width); pixels at one radius share a stat.
        rr = torch.arange(rows, dtype=dtype, device=device).view(-1, 1)
        cc = torch.arange(cols, dtype=dtype, device=device).view(1, -1)
        radius = torch.sqrt(
            (rr - self.beam_center[0]) ** 2 + (cc - self.beam_center[1]) ** 2
        )
        bin_idx = torch.floor(radius / self.bin_width).long()
        n_bins = int(bin_idx.max().item()) + 1
        self.n_bins = n_bins

        super().__init__((n_bins,), mode, decay, device, dtype)

        pixels_per_bin = torch.bincount(bin_idx.flatten(), minlength=n_bins).to(dtype)
        self.register_buffer("bin_idx", bin_idx)
        self.register_buffer("pixels_per_bin", pixels_per_bin)
        self.register_buffer(
            "valid_mask",
            torch.ones(self.frame_size, dtype=torch.bool, device=device),
        )

    def reset(self) -> None:
        super().reset()
        self.valid_mask.fill_(True)

    def _project(self, frame: Tensor, mask: Optional[Tensor]) -> Tensor:
        # Mean over each bin's (valid) pixels: xbar_b = sum_b x / count_b.
        flat_idx = self.bin_idx.flatten()
        sum_per_bin = torch.zeros_like(self.mean_)
        if mask is None:
            sum_per_bin.index_add_(0, flat_idx, frame.flatten())
            counts = self.pixels_per_bin
        else:
            mask = mask.to(device=frame.device, dtype=frame.dtype)
            sum_per_bin.index_add_(0, flat_idx, (frame * mask).flatten())
            counts = torch.zeros_like(self.mean_)
            counts.index_add_(0, flat_idx, mask.flatten())
        return sum_per_bin / counts.clamp_min(1.0)

    def mean(self) -> Tensor:
        return self.mean_[self.bin_idx]

    def var(self) -> Tensor:
        return super().var()[self.bin_idx] * self.pixels_per_bin[self.bin_idx]
