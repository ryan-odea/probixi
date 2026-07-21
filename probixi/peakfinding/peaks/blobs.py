from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

# cached frame-invariant index grids, keyed by (H, W, device, dtype)
_COORD_CACHE: dict = {}


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
    out = torch.zeros(H, W, dtype=torch.long, device=device)

    # Work only on foreground pixels: the detector is extremely sparse, so a dense
    # per-pixel flood over H*W is wasteful. Build the fg adjacency graph and flood
    # component labels over the ~10^2 fg nodes instead.
    flat = mask.reshape(-1)
    fg = flat.nonzero(as_tuple=True)[0]
    K = int(fg.numel())
    if K == 0:
        return out, 0

    rows = fg // W
    cols = fg % W
    if connectivity == 1:
        doff = torch.tensor([-1, 1, 0, 0], dtype=torch.long, device=device)
        coff = torch.tensor([0, 0, -1, 1], dtype=torch.long, device=device)
    else:
        doff = torch.tensor(
            [-1, -1, -1, 0, 0, 1, 1, 1], dtype=torch.long, device=device
        )
        coff = torch.tensor(
            [-1, 0, 1, -1, 1, -1, 0, 1], dtype=torch.long, device=device
        )

    r2 = rows[:, None] + doff[None, :]
    c2 = cols[:, None] + coff[None, :]
    valid = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
    nflat = (r2 * W + c2).clamp_(0, H * W - 1)
    nbr_c = torch.searchsorted(fg, nflat).clamp_(max=K - 1)  # in [0, K-1]
    nbr_valid = valid & (fg[nbr_c] == nflat)

    labels = torch.arange(K, dtype=torch.long, device=device)
    big = torch.full((), K, dtype=torch.long, device=device)
    # Convergence needs a host sync (torch.equal)
    check_every = 4
    for it in range(max_iters):
        gathered = torch.where(nbr_valid, labels[nbr_c], big)
        cand = torch.minimum(labels, gathered.amin(dim=1))
        converged = it % check_every == check_every - 1 and torch.equal(cand, labels)
        labels = cand
        if converged:
            break

    uniq, inv = torch.unique(labels, return_inverse=True)
    n_blobs = int(uniq.numel())
    out.reshape(-1)[fg] = inv + 1
    return out, n_blobs


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

    # only foreground pixels (label > 0) contribute; gather them once so every
    # scatter runs over the sparse set, not all H*W (bin 0 stays empty, sliced off)
    flat_labels_full = labels.flatten()
    fg = flat_labels_full.nonzero(as_tuple=True)[0]
    lbl = flat_labels_full[fg]
    flat_excess = excess.flatten()[fg]
    flat_z = z.flatten()[fg]
    flat_lbf = log_bf.flatten()[fg]
    flat_post = posterior.flatten()[fg]
    flat_var = var.flatten()[fg]
    w = flat_excess.clamp_min(0)

    _, _, grid_rows_f, grid_cols_f = _coord_grids(H, W, device, dtype)
    rows = grid_rows_f[fg]
    cols = grid_cols_f[fg]

    # Fused per-blob reductions: batch each group of same-index scatters into one call over a stacked source.
    lbl2 = lbl.unsqueeze(1)
    size = torch.zeros(n_total, dtype=torch.long, device=device)
    size.scatter_add_(0, lbl, torch.ones_like(lbl))

    add1_src = torch.stack(
        [
            w,
            w * rows,
            w * cols,
            rows,
            cols,
            flat_excess,
            flat_var.clamp_min(0),
            flat_lbf,
            flat_post,
        ],
        dim=1,
    )
    add1 = torch.zeros(n_total, add1_src.shape[1], dtype=dtype, device=device)
    add1.scatter_add_(0, lbl2.expand_as(add1_src), add1_src)
    w_sum, w_row, w_col, g_row, g_col = (
        add1[:, 0],
        add1[:, 1],
        add1[:, 2],
        add1[:, 3],
        add1[:, 4],
    )
    intensity_sum, var_sum, log_bf_sum, post_sum = (
        add1[:, 5],
        add1[:, 6],
        add1[:, 7],
        add1[:, 8],
    )

    has_w = w_sum > 0
    size_f = size.clamp_min(1).to(dtype)
    w_safe = w_sum.clamp_min(1e-12)
    row_centroid = torch.where(has_w, w_row / w_safe, g_row / size_f)
    col_centroid = torch.where(has_w, w_col / w_safe, g_col / size_f)

    # Weighted second central moments -> 2x2 covariance (inertia) matrix
    dr = rows - row_centroid[lbl]
    dc = cols - col_centroid[lbl]
    add2_src = torch.stack([w * dr * dr, w * dc * dc, w * dr * dc], dim=1)
    add2 = torch.zeros(n_total, 3, dtype=dtype, device=device)
    add2.scatter_add_(0, lbl2.expand_as(add2_src), add2_src)
    Crr, Ccc, Crc = add2[:, 0] / w_safe, add2[:, 1] / w_safe, add2[:, 2] / w_safe
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
    # sigma(I) = sqrt(sum var)
    intensity_sigma = var_sum.sqrt()

    # per-blob maxima of (excess, z), fused into one amax scatter
    amax = torch.full((n_total, 2), float("-inf"), dtype=dtype, device=device)
    amax.scatter_reduce_(
        0,
        lbl2.expand(-1, 2),
        torch.stack([flat_excess, flat_z], dim=1),
        reduce="amax",
        include_self=True,
    )
    intensity_max, z_max = amax[:, 0], amax[:, 1]

    # bbox min/max in float (row/col indices exact in float32), cast to long.
    bbox_lo = torch.empty(n_total, 2, dtype=dtype, device=device)
    bbox_lo[:, 0] = float(H)
    bbox_lo[:, 1] = float(W)
    bbox_lo.scatter_reduce_(
        0,
        lbl2.expand(-1, 2),
        torch.stack([rows, cols], dim=1),
        reduce="amin",
        include_self=True,
    )
    bbox_hi = torch.zeros(n_total, 2, dtype=dtype, device=device)
    bbox_hi.scatter_reduce_(
        0,
        lbl2.expand(-1, 2),
        torch.stack([rows + 1, cols + 1], dim=1),
        reduce="amax",
        include_self=True,
    )
    bbox_r0 = bbox_lo[:, 0].to(torch.long)
    bbox_c0 = bbox_lo[:, 1].to(torch.long)
    bbox_r1 = bbox_hi[:, 0].to(torch.long)
    bbox_c1 = bbox_hi[:, 1].to(torch.long)

    intensity_mean = intensity_sum / size_f
    # peakedness = max/mean (>= 1); gate to 0 for non-positive mean (undefined)
    peakedness = torch.where(
        intensity_mean > 1e-12,
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
