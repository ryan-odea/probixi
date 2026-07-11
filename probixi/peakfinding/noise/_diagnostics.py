from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Union

import numpy as np
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
    __slots__ = ("batch", "frames_seen", "mean", "se", "radius", "profile", "drift")

    def __init__(self, batch, frames_seen, mean, se, radius, profile, drift):
        self.batch = batch
        self.frames_seen = frames_seen
        self.mean = mean
        self.se = se
        self.radius = radius
        self.profile = profile
        self.drift = drift


def _radial_profile(img, beam_yx, bin_width: float):
    # Azimuthal mean of an assembled physical image
    h, w = img.shape
    yy = np.arange(h)[:, None] - float(beam_yx[0])
    xx = np.arange(w)[None, :] - float(beam_yx[1])
    r = np.hypot(yy, xx)
    finite = np.isfinite(img)
    rv, vv = r[finite], img[finite]
    if rv.size == 0:
        return np.zeros(0), np.zeros(0)
    bw = max(float(bin_width), 1.0)
    idx = (rv / bw).astype(int)
    counts = np.bincount(idx)
    sums = np.bincount(idx, weights=vv, minlength=counts.size)
    keep = counts > 0
    radius = (np.arange(counts.size) + 0.5) * bw
    return radius[keep], sums[keep] / counts[keep]


def _to_image(field: Tensor, mask: Tensor, assemble):
    masked = field.clone()
    masked[~mask] = float("nan")
    arr = masked.cpu().numpy()
    return assemble(arr) if assemble is not None else arr


def _capture(
    model: "NoiseModel",
    batch: int,
    frames_seen: int,
    prev_mean: Tensor,
    assemble=None,
    assembled_beam=None,
):
    mean = model.pixel.mean().detach()
    mask = model.valid_mask.detach()
    denom = mask.sum().clamp_min(1).to(mean.dtype)
    drift = float((((mean - prev_mean).abs()) * mask).sum() / denom)

    mean_img = _to_image(mean, mask, assemble)
    n = max(int(model.pixel.n_), 1)
    se = (model.pixel.var().detach() / n).clamp_min(0.0).sqrt()
    se_img = _to_image(se, mask, assemble)

    if assemble is not None:
        radius, profile = _radial_profile(
            mean_img, assembled_beam, float(model.rotational.bin_width)
        )
    else:
        prof = model.rotational.mean_.detach().cpu()
        ppb = model.rotational.pixels_per_bin.detach().cpu()
        keep = ppb > 0
        radius = ((torch.arange(prof.numel()) + 0.5) * model.rotational.bin_width)[
            keep
        ].numpy()
        profile = prof[keep].numpy()

    snap = _Snapshot(batch, frames_seen, mean_img, se_img, radius, profile, drift)
    return snap, mean.clone()


def animate_noise_diagnostics(
    model: "NoiseModel",
    frames: Iterable[Tensor],
    path: PathLike,
    *,
    batch_size: int = 15,
    fps: int = 4,
    dpi: int = 100,
    max_radius: Optional[float] = None,
    figsize: tuple[float, float] = (11.5, 6.0),
    assemble=None,
    assembled_beam=None,
) -> Path:
    # Drive model over frames in batches and write a diagnostic GIF
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
            snap, prev_mean = _capture(
                model, len(snaps) + 1, seen, prev_mean, assemble, assembled_beam
            )
            snaps.append(snap)
            pending = 0
    if pending > 0:  # trailing partial batch
        snap, prev_mean = _capture(
            model, len(snaps) + 1, seen, prev_mean, assemble, assembled_beam
        )
        snaps.append(snap)

    if not snaps:
        raise ValueError("no frames provided; nothing to animate")

    out = Path(path)
    n_batches = len(snaps)

    def _robust_max(arr):
        fin = arr[np.isfinite(arr)]
        return float(np.percentile(fin, 99.0)) if fin.size else 1.0

    prof_max = max(
        (float(s.profile.max()) for s in snaps if s.profile.size), default=1.0
    )
    mean_max = max((_robust_max(s.mean) for s in snaps), default=1.0)
    shared_vmax = max(mean_max, prof_max) or 1.0
    mean_norm = Normalize(vmin=0.0, vmax=shared_vmax)
    se_vmax = max((_robust_max(s.se) for s in snaps), default=1.0) or 1.0
    se_norm = Normalize(vmin=0.0, vmax=se_vmax)

    viridis = plt.get_cmap("viridis").copy()
    viridis.set_bad("none")  # NaN gaps transparent

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0])
    ax_se = fig.add_subplot(gs[0, 0])
    ax_img = fig.add_subplot(gs[0, 1])
    ax_rad = fig.add_subplot(gs[0, 2])
    ax_drift = fig.add_subplot(gs[1, :])

    # Left: per-pixel standard error of the mean
    ax_se.set_title("Background Standard Error")
    im_se = ax_se.imshow(snaps[0].se, origin="upper", cmap=viridis, norm=se_norm)
    ax_se.set_xticks([])
    ax_se.set_yticks([])
    cb_se = fig.colorbar(im_se, ax=ax_se, location="left", fraction=0.046, pad=0.12)
    cb_se.set_label("Photon Standard Error")

    # Middle: mean background
    ax_img.set_title("Mean Background")
    im_mean = ax_img.imshow(snaps[0].mean, origin="upper", cmap=viridis, norm=mean_norm)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    fig.colorbar(im_mean, ax=ax_img, location="right", fraction=0.046, pad=0.04)

    # Right: radial profile
    ax_rad.set_title("Radial Background Profile")
    ghost_lines = [
        ax_rad.plot([], [], color="0.8", lw=0.8, zorder=1)[0] for _ in range(n_batches)
    ]
    (rad_line,) = ax_rad.plot(
        snaps[0].radius, snaps[0].profile, color="black", lw=1.6, zorder=3
    )
    ax_rad.set_xlabel("Radius (Pixels from Beam Center)")
    rad_xmax = max_radius if max_radius is not None else float(snaps[-1].radius.max())
    ax_rad.set_xlim(0, rad_xmax)
    ax_rad.set_ylim(0.0, shared_vmax)
    ax_rad.set_yticks([])
    ax_rad.spines["left"].set_visible(False)
    ax_rad.spines["right"].set_visible(False)
    ax_rad.spines["top"].set_visible(False)

    # Bottom: per-batch drift
    ax_drift.set_title("Per-batch Drift (log-scale)")
    (drift_line,) = ax_drift.plot([], [], "o-", color="black", ms=4, lw=1.0)
    ax_drift.set_yscale("log")
    ax_drift.set_xlabel("Batch")
    ax_drift.set_ylabel("|Mean Shift|")
    ax_drift.set_xlim(0.5, n_batches + 0.5)
    drifts = [s.drift for s in snaps if s.drift > 0]
    if drifts:
        ax_drift.set_ylim(min(drifts) * 0.6, max(drifts) * 1.6)

    suptitle = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def _draw(i: int):
        s = snaps[i]
        im_se.set_data(s.se)
        im_mean.set_data(s.mean)
        rad_line.set_data(s.radius, s.profile)
        for j, ghost in enumerate(ghost_lines):
            if j < i:  # past batches trail behind the current (black) profile
                ghost.set_data(snaps[j].radius, snaps[j].profile)
            else:
                ghost.set_data([], [])
        xs = [snaps[j].batch for j in range(i + 1)]
        ys = [snaps[j].drift for j in range(i + 1)]
        drift_line.set_data(xs, ys)
        suptitle.set_text(
            f"Running Mean: Batch {s.batch} / {n_batches}; "
            f"Frames Seen: {s.frames_seen}"
        )
        return im_se, im_mean, rad_line, *ghost_lines, drift_line, suptitle

    anim = FuncAnimation(fig, _draw, frames=n_batches, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out
