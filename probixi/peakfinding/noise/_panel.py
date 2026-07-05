from typing import Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .model import NoiseStats

PanelSpec = Union[
    Tensor,
    Sequence[Tuple[slice, slice]],
    Sequence[Tuple[int, int, int, int]],
]


class PanelNoise(NoiseStats):
    """Panel-level running mean/variance.

    Attributes:
        panel_map: (rows, cols) panel id per pixel; -1 marks no-panel pixels.
        pixels_per_panel: Valid pixels assigned to each panel.
        valid_mask: (rows, cols) bool mask of pixels belonging to some panel.
    """

    panel_map: Tensor
    pixels_per_panel: Tensor
    valid_mask: Tensor

    def __init__(
        self,
        frame_size: tuple[int, int],
        panels: Optional[PanelSpec] = None,
        mode: Literal["per_frame", "online"] = "online",
        decay: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.frame_size = tuple(int(s) for s in frame_size)

        panel_map = self._build_panel_map(panels, device)
        n_panels = int(panel_map.max().item()) + 1 if panel_map.numel() else 1
        self.n_panels = n_panels

        super().__init__((n_panels,), mode, decay, device, dtype)

        valid = panel_map >= 0
        pixels_per_panel = torch.bincount(
            panel_map[valid].flatten().to(torch.long), minlength=n_panels
        ).to(dtype)

        self.register_buffer("panel_map", panel_map)
        self.register_buffer("pixels_per_panel", pixels_per_panel)
        self.register_buffer("valid_mask", valid.to(device=device, dtype=torch.bool))
        # per-pixel panel index; no-panel pixels fold to 0 and are zeroed by valid_f
        self.register_buffer("_pid_flat", panel_map.clamp_min(0).flatten())

    def _build_panel_map(
        self, panels: Optional[PanelSpec], device: Optional[torch.device]
    ) -> Tensor:
        rows, cols = self.frame_size
        if panels is None:
            return torch.zeros((rows, cols), dtype=torch.long, device=device)
        if isinstance(panels, Tensor):
            if tuple(panels.shape) != self.frame_size:
                raise ValueError(
                    f"panel_map shape {tuple(panels.shape)} != frame_size {self.frame_size}"
                )
            return panels.to(device=device, dtype=torch.long)

        panel_map = torch.full((rows, cols), -1, dtype=torch.long, device=device)
        for pid, spec in enumerate(panels):
            if len(spec) == 2:
                rs, cs = spec
                panel_map[rs, cs] = pid
            elif len(spec) == 4:
                r0, r1, c0, c1 = spec
                panel_map[r0:r1, c0:c1] = pid
            else:
                raise ValueError("panel spec must be 2- or 4-tuple")
        return panel_map

    def _project(self, frame: Tensor, mask: Optional[Tensor]) -> Tensor:
        # Mean over each panel's valid pixels: xbar_p = sum_p mask*x / sum_p mask.
        valid = self.valid_mask
        if mask is not None:
            valid = valid & mask.to(device=valid.device, dtype=torch.bool)
        valid_f = valid.to(frame.dtype)
        pid = self._pid_flat

        sum_per_panel = torch.zeros_like(self.mean_)
        sum_per_panel.index_add_(0, pid, (frame * valid_f).flatten())
        count_per_panel = torch.zeros_like(self.mean_)
        count_per_panel.index_add_(0, pid, valid_f.flatten())
        return sum_per_panel / count_per_panel.clamp_min(1.0)

    def mean(self) -> Tensor:
        safe = self.panel_map.clamp_min(0)
        z = torch.zeros((), dtype=self.mean_.dtype, device=self.mean_.device)
        return torch.where(self.valid_mask, self.mean_[safe], z)

    def var(self) -> Tensor:
        safe = self.panel_map.clamp_min(0)
        z = torch.zeros((), dtype=self.M2_.dtype, device=self.M2_.device)
        return torch.where(self.valid_mask, super().var()[safe], z)
