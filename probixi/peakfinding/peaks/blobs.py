from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import Tensor

# cached frame-invariant index grids, keyed by (H, W, device[, dtype])
_SEED_CACHE: dict = {}
_COORD_CACHE: dict = {}


def _seed_grid(H: int, W: int, device: torch.device) -> Tensor:
    key = (H, W, device)
    grid = _SEED_CACHE.get(key)
    if grid is None:
        grid = (torch.arange(H * W, dtype=torch.long, device=device) + 1).reshape(H, W)
        _SEED_CACHE[key] = grid
    return grid


def _coord_grids(
    H: int, W: int, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    key = (H, W, device, dtype)
    got = _COORD_CACHE.get(key)
    if got is None:
        flat_rows = (
            torch.arange(H, device=device, dtype=torch.long)
            .view(-1, 1)
            .expand(H, W)
            .flatten()
        )
        flat_cols = (
            torch.arange(W, device=device, dtype=torch.long)
            .view(1, -1)
            .expand(H, W)
            .flatten()
        )
        got = (flat_rows, flat_cols, flat_rows.to(dtype), flat_cols.to(dtype))
        _COORD_CACHE[key] = got
    return got


@dataclass
class BlobStats:
    """Per peak-blob statistics as parallel ``(n_blobs,)`` device tensors.

    Fields: label_id (1-based contiguous), size, row/col_centroid
    (intensity-weighted), bbox_{r0,r1,c0,c1} (r1/c1 exclusive), intensity_sum
    (excess), intensity_sigma (sqrt of summed per-pixel variance), intensity_max,
    z_max, log_bf_sum, posterior_mean, eccentricity (lambda_max/lambda_min >= 1),
    peakedness (intensity_max / mean intensity).
    """

    label_id: Tensor
    size: Tensor
    row_centroid: Tensor
    col_centroid: Tensor
    bbox_r0: Tensor
    bbox_r1: Tensor
    bbox_c0: Tensor
    bbox_c1: Tensor
    intensity_sum: Tensor
    intensity_sigma: Tensor
    intensity_max: Tensor
    z_max: Tensor
    log_bf_sum: Tensor
    posterior_mean: Tensor
    eccentricity: Tensor
    peakedness: Tensor

    def __len__(self) -> int:
        return int(self.label_id.numel())


@torch.no_grad()
def label_connected_components(
    mask: Tensor,
    connectivity: int = 2,
    max_iters: int = 64,
) -> tuple[Tensor, int]:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    H, W = mask.shape
    device = mask.device
    if not mask.any():
        return torch.zeros(H, W, dtype=torch.long, device=device), 0

    # Seed each foreground pixel with a unique id, then iterate min over self +
    # neighbors until the per-component global minimum floods to convergence
    INF = H * W + 2
    seeds = _seed_grid(H, W, device)
    labels = seeds * mask
    offsets = (
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 1
        else [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    )

    for it in range(max_iters):
        prev = labels
        cur = labels.masked_fill(~mask, INF)
        padded = F.pad(cur, (1, 1, 1, 1), value=INF)
        acc = cur
        for dy, dx in offsets:
            acc = torch.minimum(acc, padded[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W])
        labels = torch.where(mask, acc, 0)
        # poll for convergence every 8th iteration
        if it % 8 == 7 and labels.equal(prev):
            break

    # Compress sparse surviving seed ids to a dense 1:n_blobs range
    unique = labels[mask].unique()
    n_blobs = int(unique.numel())
    remap = torch.zeros(int(labels.max()) + 1, dtype=torch.long, device=device)
    remap[unique] = torch.arange(1, n_blobs + 1, dtype=torch.long, device=device)
    return remap[labels], n_blobs


def empty_stats(device: torch.device, dtype: torch.dtype) -> BlobStats:
    el = torch.zeros(0, dtype=torch.long, device=device)
    ef = torch.zeros(0, dtype=dtype, device=device)
    return BlobStats(
        label_id=el,
        size=el,
        row_centroid=ef,
        col_centroid=ef,
        bbox_r0=el,
        bbox_r1=el,
        bbox_c0=el,
        bbox_c1=el,
        intensity_sum=ef,
        intensity_sigma=ef,
        intensity_max=ef,
        z_max=ef,
        log_bf_sum=ef,
        posterior_mean=ef,
        eccentricity=ef,
        peakedness=ef,
    )


@torch.no_grad()
def select_blobs(stats: BlobStats, keep: Tensor) -> BlobStats:
    idx = (
        torch.nonzero(keep, as_tuple=False).flatten()
        if keep.dtype == torch.bool
        else keep
    )
    return BlobStats(**{f.name: getattr(stats, f.name)[idx] for f in fields(stats)})


@torch.no_grad()
def compute_blob_stats(
    labels: Tensor,
    n_blobs: int,
    excess: Tensor,
    z: Tensor,
    log_bf: Tensor,
    posterior: Tensor,
    var: Tensor,
) -> BlobStats:
    if n_blobs == 0:
        return empty_stats(labels.device, excess.dtype)

    H, W = labels.shape
    device = labels.device
    dtype = excess.dtype
    n_total = n_blobs + 1

    # weights provide brightness-weighted moments
    flat_labels = labels.flatten()
    flat_excess = excess.flatten()
    flat_z = z.flatten()
    flat_lbf = log_bf.flatten()
    flat_post = posterior.flatten()
    flat_var = var.flatten()
    w = flat_excess.clamp_min(0)

    size = torch.zeros(n_total, dtype=torch.long, device=device)
    size.scatter_add_(0, flat_labels, torch.ones_like(flat_labels))
    w_sum = torch.zeros(n_total, dtype=dtype, device=device)
    w_sum.scatter_add_(0, flat_labels, w)

    flat_rows, flat_cols, rows, cols = _coord_grids(H, W, device, dtype)
    w_row = torch.zeros(n_total, dtype=dtype, device=device)
    w_col = torch.zeros(n_total, dtype=dtype, device=device)
    g_row = torch.zeros(n_total, dtype=dtype, device=device)
    g_col = torch.zeros(n_total, dtype=dtype, device=device)
    w_row.scatter_add_(0, flat_labels, w * rows)
    w_col.scatter_add_(0, flat_labels, w * cols)
    g_row.scatter_add_(0, flat_labels, rows)
    g_col.scatter_add_(0, flat_labels, cols)

    has_w = w_sum > 0
    size_f = size.clamp_min(1).to(dtype)
    row_centroid = torch.where(has_w, w_row / w_sum.clamp_min(1e-12), g_row / size_f)
    col_centroid = torch.where(has_w, w_col / w_sum.clamp_min(1e-12), g_col / size_f)

    # Weighted second central moments -> 2x2 covariance (inertia) matrix
    dr = rows - row_centroid[flat_labels]
    dc = cols - col_centroid[flat_labels]
    Crr = torch.zeros(n_total, dtype=dtype, device=device)
    Ccc = torch.zeros(n_total, dtype=dtype, device=device)
    Crc = torch.zeros(n_total, dtype=dtype, device=device)
    Crr.scatter_add_(0, flat_labels, w * dr * dr)
    Ccc.scatter_add_(0, flat_labels, w * dc * dc)
    Crc.scatter_add_(0, flat_labels, w * dr * dc)
    w_safe = w_sum.clamp_min(1e-12)
    Crr, Ccc, Crc = Crr / w_safe, Ccc / w_safe, Crc / w_safe
    # Closed-form symmetric-2x2 eigenvalues; ratio = eccentricity. disc >=0
    tr = Crr + Ccc
    disc = (tr * tr - 4.0 * (Crr * Ccc - Crc * Crc)).clamp_min(0.0).sqrt()
    lam_max = 0.5 * (tr + disc)
    lam_min = 0.5 * (tr - disc)
    eccentricity = torch.where(
        lam_min > 1e-12,
        lam_max / lam_min.clamp_min(1e-12),
        torch.full_like(lam_max, float("inf")),
    )

    intensity_sum = torch.zeros(n_total, dtype=dtype, device=device)
    intensity_sum.scatter_add_(0, flat_labels, flat_excess)
    # sigma(I) = sqrt(sum var): per-pixel noise treated independent over the peak blob
    var_sum = torch.zeros(n_total, dtype=dtype, device=device)
    var_sum.scatter_add_(0, flat_labels, flat_var.clamp_min(0))
    intensity_sigma = var_sum.sqrt()
    log_bf_sum = torch.zeros(n_total, dtype=dtype, device=device)
    log_bf_sum.scatter_add_(0, flat_labels, flat_lbf)
    post_sum = torch.zeros(n_total, dtype=dtype, device=device)
    post_sum.scatter_add_(0, flat_labels, flat_post)

    intensity_max = torch.full((n_total,), float("-inf"), dtype=dtype, device=device)
    intensity_max = intensity_max.scatter_reduce(
        0, flat_labels, flat_excess, reduce="amax", include_self=True
    )
    z_max = torch.full((n_total,), float("-inf"), dtype=dtype, device=device)
    z_max = z_max.scatter_reduce(
        0, flat_labels, flat_z, reduce="amax", include_self=True
    )

    bbox_r0 = torch.full((n_total,), H, dtype=torch.long, device=device)
    bbox_r0.scatter_reduce_(0, flat_labels, flat_rows, reduce="amin", include_self=True)
    bbox_r1 = torch.zeros(n_total, dtype=torch.long, device=device)
    bbox_r1.scatter_reduce_(
        0, flat_labels, flat_rows + 1, reduce="amax", include_self=True
    )
    bbox_c0 = torch.full((n_total,), W, dtype=torch.long, device=device)
    bbox_c0.scatter_reduce_(0, flat_labels, flat_cols, reduce="amin", include_self=True)
    bbox_c1 = torch.zeros(n_total, dtype=torch.long, device=device)
    bbox_c1.scatter_reduce_(
        0, flat_labels, flat_cols + 1, reduce="amax", include_self=True
    )

    intensity_mean = intensity_sum / size_f
    peakedness = torch.where(
        intensity_mean.abs() > 1e-12,
        intensity_max / intensity_mean.clamp_min(1e-12),
        torch.zeros_like(intensity_mean),
    )

    idx = torch.arange(1, n_total, device=device)
    return BlobStats(
        label_id=idx,
        size=size[idx],
        row_centroid=row_centroid[idx],
        col_centroid=col_centroid[idx],
        bbox_r0=bbox_r0[idx],
        bbox_r1=bbox_r1[idx],
        bbox_c0=bbox_c0[idx],
        bbox_c1=bbox_c1[idx],
        intensity_sum=intensity_sum[idx],
        intensity_sigma=intensity_sigma[idx],
        intensity_max=intensity_max[idx],
        z_max=z_max[idx],
        log_bf_sum=log_bf_sum[idx],
        posterior_mean=post_sum[idx] / size_f[idx],
        eccentricity=eccentricity[idx],
        peakedness=peakedness[idx],
    )


def filter_blobs(
    stats: BlobStats,
    size_min: int = 2,
    size_max: int = 30,
    eccentricity_max: float = 5.0,
    peakedness_min: float = 1.2,
) -> Tensor:
    if size_min < 1:
        raise ValueError("size_min must be >= 1")
    return (
        (stats.size >= size_min)
        & (stats.size <= size_max)
        & (stats.eccentricity <= eccentricity_max)
        & (stats.peakedness >= peakedness_min)
    )
