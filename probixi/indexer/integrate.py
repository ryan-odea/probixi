from __future__ import annotations

import torch
from torch import Tensor

# Box integration of background-subtracted intensity at predicted positions. The
# excess map is already background-subtracted, so integrated intensity is the sum
# of excess over a small box and its variance the sum of per-pixel noise
# variances. The box must hold a spot despite prediction scatter yet stay far
# narrower than the inter-spot spacing so neighbours don't leak in.


@torch.no_grad()
def integrate_boxes(
    excess: Tensor,
    var: Tensor,
    positions: Tensor,
    radius: int = 4,
    mean: Tensor | None = None,
    pixel_valid: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Integrate intensity (sum excess), sigma (sqrt sum var), peak (max excess)
    # and background (mean per-pixel noise) in a (2*radius+1)^2 box at each
    # centre
    H, W = excess.shape
    device = excess.device
    M = positions.shape[0]
    if M == 0:
        z = positions.new_zeros(0)
        return z, z, z, z

    centre = torch.round(positions).to(torch.long)
    r0, c0 = centre[:, 0], centre[:, 1]
    I = torch.zeros(M, dtype=excess.dtype, device=device) # noqa: E741
    var_sum = torch.zeros(M, dtype=excess.dtype, device=device)
    peak = torch.full((M,), float("-inf"), dtype=excess.dtype, device=device)
    bg_sum = torch.zeros(M, dtype=excess.dtype, device=device)
    bg_count = torch.zeros(M, dtype=excess.dtype, device=device)
    for dr in range(-radius, radius + 1):
        rr = r0 + dr
        in_r = (rr >= 0) & (rr < H)
        rr_c = rr.clamp(0, H - 1)
        for dc in range(-radius, radius + 1):
            cc = c0 + dc
            valid = in_r & (cc >= 0) & (cc < W)
            cc_c = cc.clamp(0, W - 1)
            if pixel_valid is not None:
                valid = valid & pixel_valid[rr_c, cc_c]
            e = excess[rr_c, cc_c]
            v = var[rr_c, cc_c]
            zero = torch.zeros_like(e)
            I = I + torch.where(valid, e, zero) # noqa: E741
            var_sum = var_sum + torch.where(valid, v, zero)
            peak = torch.where(valid, torch.maximum(peak, e), peak)
            if mean is not None:
                m = mean[rr_c, cc_c]
                bg_sum = bg_sum + torch.where(valid, m, zero)
                bg_count = bg_count + valid.to(excess.dtype)
    # boxes with no valid pixel
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    background = bg_sum / bg_count.clamp_min(1.0)
    return I, var_sum.clamp_min(0.0).sqrt(), peak, background


@torch.no_grad()
def integrate_predicted(
    pred_positions: Tensor,
    excess: Tensor,
    var: Tensor,
    obs_positions: Tensor,
    obs_intensity: Tensor,
    obs_sigma: Tensor,
    snap_radius: float = 5.0,
    box_radius: int = 3,
    mean: Tensor | None = None,
    pixel_valid: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    M = pred_positions.shape[0]
    positions = pred_positions
    snapped = torch.zeros(M, dtype=torch.bool, device=pred_positions.device)
    if obs_positions.shape[0] > 0:
        d = torch.cdist(pred_positions, obs_positions.to(pred_positions.dtype))
        nn_dist, nn_idx = d.min(dim=1)
        snapped = nn_dist < snap_radius
        positions = torch.where(
            snapped.unsqueeze(-1),
            obs_positions.to(pred_positions.dtype)[nn_idx],
            pred_positions,
        )
    intensity, sigma, peak, background = integrate_boxes(
        excess, var, positions, box_radius, mean=mean, pixel_valid=pixel_valid
    )
    return positions, intensity, sigma, snapped, peak, background
