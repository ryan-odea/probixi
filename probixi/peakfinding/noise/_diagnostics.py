from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Union

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .model import NoiseModel

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize

PathLike = Union[str, Path]


def _iter_single(frames: Iterable[Tensor]):
    # Flatten (N, H, W) batches into single (H, W) frames; pass 2-D through.
    for item in frames:
        if isinstance(item, Tensor) and item.ndim == 3:
            for sub in item:
                yield sub
        else:
            yield item


class _Snapshot:
    __slots__ = ("batch", "frames_seen", "mean", "radius", "profile", "drift")

    def __init__(self, batch, frames_seen, mean, radius, profile, drift):
        self.batch = batch
        self.frames_seen = frames_seen
        self.mean = mean
        self.radius = radius
        self.profile = profile
        self.drift = drift


def _capture(model: "NoiseModel", batch: int, frames_seen: int, prev_mean: Tensor):
    # Snapshot the model's current background state plus the per-batch drift
    mean = model.pixel.mean().detach()
    mask = model.valid_mask.detach()
    denom = mask.sum().clamp_min(1).to(mean.dtype)
    drift = float((((mean - prev_mean).abs()) * mask).sum() / denom)

    masked = mean.clone()
    masked[~mask] = float("nan")

    prof = model.rotational.mean_.detach().cpu()
    ppb = model.rotational.pixels_per_bin.detach().cpu()
    keep = ppb > 0
    radius = ((torch.arange(prof.numel()) + 0.5) * model.rotational.bin_width)[keep]

    snap = _Snapshot(
        batch=batch,
        frames_seen=frames_seen,
        mean=masked.cpu().numpy(),
        radius=radius.numpy(),
        profile=prof[keep].numpy(),
        drift=drift,
    )
    return snap, mean.clone()


def animate_noise_diagnostics(
    model: "NoiseModel",
    frames: Iterable[Tensor],
    path: PathLike,
    *,
    batch_size: int = 15,
    fps: int = 4,
    cmap: str = "viridis",
    dpi: int = 100,
    max_radius: Optional[float] = None,
    figsize: tuple[float, float] = (8.4, 5.2),
) -> Path:
    """Drive ``model`` over ``frames`` in batches and write a diagnostic GIF.

    Updates the (online) model frame by frame, snapshotting after every
    ``batch_size`` frames, then renders a 3-panel animation: running mean
    background, radial background profile, and per-batch drift (log scale).

    Parameters
    ----------
    model : NoiseModel
        Live model to fit and visualise; it is updated in place.
    frames : iterable of Tensor
        Frames to feed (2-D frames or (N, H, W) batches).
    path : str or Path
        Output ``.gif`` path.
    batch_size : int, default 15
        Frames folded in between snapshots (one animation frame per batch).
    fps : int, default 4
        Playback frame rate of the GIF.
    cmap : str, default "viridis"
        Colormap for the mean-background panel.
    dpi : int, default 100
        Render resolution.
    max_radius : float, optional
        Clip the radial-profile x-axis to this radius (px); full range if None.
    figsize : tuple, default (8.4, 5.2)
        Figure size in inches.

    Returns
    -------
    Path
        The written GIF path.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    snaps: list[_Snapshot] = []
    prev_mean = torch.zeros_like(model.pixel.mean())
    seen = 0
    pending = 0
    for frame in _iter_single(frames):
        model.update(frame)
        seen += 1
        pending += 1
        if pending >= batch_size:
            snap, prev_mean = _capture(model, len(snaps) + 1, seen, prev_mean)
            snaps.append(snap)
            pending = 0
    if pending > 0:  # trailing partial batch
        snap, prev_mean = _capture(model, len(snaps) + 1, seen, prev_mean)
        snaps.append(snap)

    if not snaps:
        raise ValueError("no frames provided; nothing to animate")

    out = Path(path)
    decay_label = "cumulative" if model.decay == 1.0 else f"{model.decay:g}"
    n_batches = len(snaps)

    # Stable color scale from the final mean so the colorbar does not flicker.
    final = snaps[-1].mean
    finite = final[~_isnan(final)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(_percentile(finite, 99.0)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_rad = fig.add_subplot(gs[0, 1])
    ax_drift = fig.add_subplot(gs[1, :])

    rows, cols = model.frame_size
    bc = model.rotational.beam_center

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("0.6")

    ax_img.set_title("running mean background")
    im = ax_img.imshow(snaps[0].mean, origin="upper", cmap=cmap_obj, norm=norm)
    ax_img.axvline(bc[1], color="magenta", lw=0.4, alpha=0.5)
    ax_img.axhline(bc[0], color="magenta", lw=0.4, alpha=0.5)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
    cbar.set_label("mean counts")

    ax_rad.set_title("radial background profile")
    (rad_line,) = ax_rad.plot(snaps[0].radius, snaps[0].profile, color="C0")
    ax_rad.set_xlabel("radius (px from beam center)")
    ax_rad.set_ylabel("mean counts")
    rad_xmax = max_radius if max_radius is not None else float(snaps[-1].radius.max())
    ax_rad.set_xlim(0, rad_xmax)
    prof_max = max(float(s.profile.max()) for s in snaps if s.profile.size)
    ax_rad.set_ylim(0, prof_max * 1.05)

    ax_drift.set_title("per-batch drift (log scale)")
    (drift_line,) = ax_drift.plot([], [], "o-", color="red", ms=4, lw=1.0)
    ax_drift.set_yscale("log")
    ax_drift.set_xlabel("batch")
    ax_drift.set_ylabel("mean |mean shift|")
    ax_drift.set_xlim(0.5, n_batches + 0.5)
    drifts = [s.drift for s in snaps if s.drift > 0]
    if drifts:
        ax_drift.set_ylim(min(drifts) * 0.6, max(drifts) * 1.6)

    suptitle = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def _draw(i: int):
        s = snaps[i]
        im.set_data(s.mean)
        rad_line.set_data(s.radius, s.profile)
        xs = [snaps[j].batch for j in range(i + 1)]
        ys = [snaps[j].drift for j in range(i + 1)]
        drift_line.set_data(xs, ys)
        suptitle.set_text(
            f"mean  batch {s.batch}/{n_batches}   "
            f"frames seen: {s.frames_seen}   decay={decay_label}"
        )
        return im, rad_line, drift_line, suptitle

    anim = FuncAnimation(fig, _draw, frames=n_batches, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out


def _isnan(arr):
    import numpy as np

    return np.isnan(arr)


def _percentile(arr, q: float):
    import numpy as np

    return np.percentile(arr, q)
