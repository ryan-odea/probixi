from typing import Literal, Optional

import torch
from torch import Tensor

from .model import NoiseStats


class PixelNoise(NoiseStats):
    # Per-pixel running mean/variance over the full frame.

    valid_mask: Tensor

    def __init__(
        self,
        frame_size: tuple[int, int],
        mode: Literal["per_frame", "online"],
        decay: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.frame_size = tuple(int(s) for s in frame_size)
        super().__init__(self.frame_size, mode, decay, device, dtype)
        self.register_buffer(
            "valid_mask",
            torch.ones(self.frame_size, dtype=torch.bool, device=device),
        )

    def reset(self) -> None:
        super().reset()
        self.valid_mask.fill_(True)

    @torch.no_grad()
    def commit_dead_pixel_mask(self, tol: float = 0.0) -> Tensor:
        self.valid_mask.copy_(~(self.var() <= tol))
        return self.valid_mask
