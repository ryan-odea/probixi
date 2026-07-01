from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib
import numpy as np

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt

PathLike = Union[str, Path]


def _to_numpy(x) -> np.ndarray:
    arr = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return arr


def _xy(points) -> Optional[np.ndarray]:
    # (N, 2) (row, col) -> (N, 2) (x=col, y=row); None/empty -> None
    if points is None:
        return None
    arr = _to_numpy(points)
    if arr.size == 0:
        return None
    arr = arr.reshape(-1, 2)
    return np.stack([arr[:, 1], arr[:, 0]], axis=-1)


def render_frame(
    image,
    *,
    path: PathLike,
    peaks=None,
    reflections=None,
    mask=None,
    title: Optional[str] = None,
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.5,
    cmap: str = "gray",
    dpi: int = 150,
) -> Path:
    """Render a detector frame with peak/reflection overlays to an image file.

    Parameters
    ----------
    image : array-like
        ``(H, W)`` detector frame (torch tensor or numpy array).
    path : str or Path
        Output image path (extension picks the format, e.g. ``.png``).
    peaks : array-like, optional
        ``(N, 2)`` detected peak ``(row, col)`` positions, drawn as open circles.
    reflections : array-like, optional
        ``(M, 2)`` predicted/indexed reflection ``(row, col)`` positions, drawn
        as crosses.
    mask : array-like, optional
        ``(H, W)`` bool array of valid pixels; invalid pixels are excluded from
        the intensity scaling.
    title : str, optional
        Title drawn above the frame.
    vmin_pct, vmax_pct : float
        Percentiles of the valid pixels used as the display intensity range.
    cmap : str, default "gray"
        Matplotlib colormap.
    dpi : int, default 150
        Output resolution.

    Returns
    -------
    Path
        The written image path.
    """
    img = _to_numpy(image).astype(np.float64)
    valid = _to_numpy(mask).astype(bool) if mask is not None else np.isfinite(img)
    finite = img[valid & np.isfinite(img)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [vmin_pct, vmax_pct])
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(8, 8 * img.shape[0] / max(img.shape[1], 1)))
    ax.imshow(
        img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="none"
    )

    pk = _xy(peaks)
    if pk is not None:
        ax.scatter(
            pk[:, 0],
            pk[:, 1],
            s=80,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.0,
            label=f"peaks ({len(pk)})",
        )
    rf = _xy(reflections)
    if rf is not None:
        ax.scatter(
            rf[:, 0],
            rf[:, 1],
            s=24,
            c="red",
            marker="+",
            linewidths=0.6,
            label=f"reflections ({len(rf)})",
        )

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_xlabel("fs / px")
    ax.set_ylabel("ss / px")
    if title:
        ax.set_title(title, fontsize=10)
    if pk is not None or rf is not None:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)

    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
