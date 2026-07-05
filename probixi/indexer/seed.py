from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor

from .refine import _axis_angle_to_rotation

# TORO-style seeding over orientation with the cell fixed
# TORO: https://doi.org/10.1107/S1600576724003182
#
# A rotation has 3 DOF:
# the rotated real a-axis is a known-length vector (2 DOF direction) plus a roll
# about it (1 DOF)
# 1. Sample a-axis directions on a Fibonacci hemisphere,
# 2. score each by the integer-projection fitness sum_i cos(2*pi t.q_i)
# (maximal when t = La*dir is a true lattice vector)
# 3. spin the best directions and score each full orientation by induced-hkl inliers.


@lru_cache(maxsize=None)
def _fibonacci_hemisphere(n: int, device, dtype: torch.dtype) -> Tensor:
    i = torch.arange(n, device=device, dtype=dtype)
    z = (i + 0.5) / n
    r = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    phi = i * (math.pi * (3.0 - math.sqrt(5.0)))  # golden angle
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)


def _rotations_mapping(a: Tensor, b: Tensor) -> Tensor:
    # Batched shortest-arc rotations carrying unit vector a (3,) onto each unit
    # vector in b (M, 3)- R @ a = b
    M = b.shape[0]
    device, dtype = b.device, b.dtype
    a_exp = a.view(1, 3).expand(M, 3)
    v = torch.linalg.cross(a_exp, b, dim=-1)
    c = (a_exp * b).sum(dim=-1)
    zero = torch.zeros(M, device=device, dtype=dtype)
    K = torch.stack(
        [
            torch.stack([zero, -v[:, 2], v[:, 1]], dim=-1),
            torch.stack([v[:, 2], zero, -v[:, 0]], dim=-1),
            torch.stack([-v[:, 1], v[:, 0], zero], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=device, dtype=dtype).expand(M, 3, 3)
    coeff = (1.0 / (1.0 + c).clamp_min(1e-12)).view(M, 1, 1)
    R = eye + K + (K @ K) * coeff
    # antiparallel jic: rotate pi about any axis perpendicular to a
    anti = c < -1.0 + 1e-6
    if bool(anti.any()):
        perp = torch.linalg.cross(
            a_exp,
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand(M, 3),
        )
        small = torch.linalg.vector_norm(perp, dim=-1) < 1e-6
        if bool(small.any()):
            perp[small] = torch.linalg.cross(
                a_exp[small],
                torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand(
                    int(small.sum()), 3
                ),
            )
        perp = perp / torch.linalg.vector_norm(perp, dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        R_anti = _axis_angle_to_rotation(perp * math.pi)
        R = torch.where(anti.view(M, 1, 1), R_anti, R)
    return R


def _induced_inlier_counts(
    A: Tensor, q: Tensor, q_tolerance: float, weights: Tensor | None = None
) -> Tensor:
    # Per candidate basis A (C, 3, 3): score observed q whose induced hkl
    # round(A^-1 q) predicts q back within q_tolerance
    A_inv = torch.linalg.inv(A)
    hkl = torch.round(torch.einsum("cij,nj->cni", A_inv, q))
    q_pred = torch.einsum("cij,cnj->cni", A, hkl)
    diff = q_pred - q.unsqueeze(0)
    sq = (diff * diff).sum(dim=-1)
    matched = sq < q_tolerance * q_tolerance
    if weights is None:
        return matched.sum(dim=-1)
    return (matched.to(weights.dtype) * weights.unsqueeze(0)).sum(dim=-1)


def sphere_seed_candidates(
    q_obs: Tensor,
    B_target: Tensor,
    q_tolerance: float = 0.02,
    n_directions: int = 6000,
    n_spin: int = 120,
    top_directions: int = 32,
    top_k: int = 64,
    weights: Tensor | None = None,
) -> Tensor:
    # Seed candidate A = U @ B_target by a Fibonacci-sphere search over the rotated
    # a-axis direction plus a roll about it.
    #
    # weights (N,) are per-peak detection confidences
    if q_obs.ndim != 2 or q_obs.shape[-1] != 3:
        raise ValueError("q_obs must be (N, 3)")
    if B_target.shape != (3, 3):
        raise ValueError("B_target must be (3, 3)")
    N = q_obs.shape[0]
    if N < 2:
        return q_obs.new_empty(0, 3, 3)

    device, out_dtype = q_obs.device, q_obs.dtype
    work = torch.float32
    B = B_target.to(device=device, dtype=work)
    q = q_obs.to(device=device, dtype=work)
    w = None if weights is None else weights.to(device=device, dtype=work)

    # columns of B^-T are the direct lattice vectors a, b, c
    A_direct = torch.linalg.inv(B).transpose(-1, -2)
    a_real = A_direct[:, 0]
    La = torch.linalg.vector_norm(a_real).clamp_min(1e-12)
    a_hat = a_real / La

    # 1/2 from aboev: score a-axis directions by integer-projection fitness
    dirs = _fibonacci_hemisphere(n_directions, device, work)
    proj = (La * dirs) @ q.T  # (D, N) = t . q over observed peaks
    cos = torch.cos(2.0 * math.pi * proj)
    if w is None:
        fitness = cos.mean(dim=1)
    else:
        fitness = (cos * w.unsqueeze(0)).sum(dim=1) / w.sum().clamp_min(1e-12)
    M = min(top_directions, dirs.shape[0])
    top_dirs = dirs[torch.topk(fitness, k=M).indices]

    # 3 from above: land the reference a-axis on each kept direction, then spin through
    # n_spin rolls; score each full orientation A = U @ B by induced hkls.
    R0 = _rotations_mapping(a_hat, top_dirs)  # R0 @ a_hat = dir
    thetas = torch.arange(n_spin, device=device, dtype=work) * (2.0 * math.pi / n_spin)
    omega = top_dirs.unsqueeze(1) * thetas.view(1, n_spin, 1)  # (M, S, 3)
    R_spin = _axis_angle_to_rotation(omega)  # (M, S, 3, 3)
    U = R_spin @ R0.unsqueeze(1)
    A_cand = (U @ B).reshape(M * n_spin, 3, 3)

    scores = _induced_inlier_counts(A_cand, q, q_tolerance, weights=w)
    keep = torch.topk(scores, k=min(top_k, A_cand.shape[0])).indices
    return A_cand[keep].to(out_dtype)
