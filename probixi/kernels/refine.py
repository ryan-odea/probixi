import math
from functools import lru_cache

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as lib

from probixi.indexer.refine import RefineResult


@triton.jit
def rotation(x, y, z):
    norm2 = x * x + y * y + z * z
    theta = lib.sqrt(tl.maximum(norm2, 1.0e-24))
    sn = lib.sin(theta)
    cs = lib.cos(theta)
    a = sn / theta
    b = (1.0 - cs) / (theta * theta)
    return (
        1.0 - b * (y * y + z * z),
        b * x * y - a * z,
        b * x * z + a * y,
        b * x * y + a * z,
        1.0 - b * (x * x + z * z),
        b * y * z - a * x,
        b * x * z - a * y,
        b * y * z + a * x,
        1.0 - b * (x * x + y * y),
        a,
        b,
        theta,
        sn,
        cs,
        norm2,
    )


@triton.jit
def assign(a00, a01, a02, a10, a11, a12, a20, a21, a22, qx, qy, qz, valid, tol):
    c00 = a11 * a22 - a12 * a21
    c01 = a02 * a21 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    c10 = a12 * a20 - a10 * a22
    c11 = a00 * a22 - a02 * a20
    c12 = a02 * a10 - a00 * a12
    c20 = a10 * a21 - a11 * a20
    c21 = a01 * a20 - a00 * a21
    c22 = a00 * a11 - a01 * a10
    det = a00 * c00 + a01 * c10 + a02 * c20
    h = lib.nearbyint((c00 * qx + c01 * qy + c02 * qz) / det)
    k = lib.nearbyint((c10 * qx + c11 * qy + c12 * qz) / det)
    ell = lib.nearbyint((c20 * qx + c21 * qy + c22 * qz) / det)
    dx = a00 * h + a01 * k + a02 * ell - qx
    dy = a10 * h + a11 * k + a12 * ell - qy
    dz = a20 * h + a21 * k + a22 * ell - qz
    match = (dx * dx + dy * dy + dz * dz < tol * tol) & valid
    return h, k, ell, match


@triton.jit
def kernel(
    A,
    Q,
    W,
    FID,
    NOBS,
    BC1,
    BC2,
    AO,
    HK,
    IDX,
    RMS,
    SCORE,
    NI,
    HISTORY,
    N: tl.constexpr,
    STEPS: tl.constexpr,
    REASSIGN: tl.constexpr,
    MIN_INDEXED: tl.constexpr,
    TOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per orientation candidate, running every Adam step in-register.
    c = tl.program_id(0)
    f = tl.load(FID + c)
    i = tl.arange(0, BLOCK)
    valid = i < tl.load(NOBS + f)
    qx = tl.load(Q + (f * N + i) * 3, valid, 0.0)
    qy = tl.load(Q + (f * N + i) * 3 + 1, valid, 0.0)
    qz = tl.load(Q + (f * N + i) * 3 + 2, valid, 0.0)
    w = tl.load(W + f * N + i, valid, 0.0)
    b00 = tl.load(A + c * 9)
    b01 = tl.load(A + c * 9 + 1)
    b02 = tl.load(A + c * 9 + 2)
    b10 = tl.load(A + c * 9 + 3)
    b11 = tl.load(A + c * 9 + 4)
    b12 = tl.load(A + c * 9 + 5)
    b20 = tl.load(A + c * 9 + 6)
    b21 = tl.load(A + c * 9 + 7)
    b22 = tl.load(A + c * 9 + 8)
    x = tl.full((), 0.0, tl.float32)
    y = x
    z = x
    mx = x
    my = x
    mz = x
    vx = x
    vy = x
    vz = x
    h, k, ell, match = assign(
        b00, b01, b02, b10, b11, b12, b20, b21, b22, qx, qy, qz, valid, TOL
    )
    for step in range(STEPS):
        r00, r01, r02, r10, r11, r12, r20, r21, r22, a, b, theta, sn, cs, norm2 = (
            rotation(x, y, z)
        )
        a00 = r00 * b00 + r01 * b10 + r02 * b20
        a01 = r00 * b01 + r01 * b11 + r02 * b21
        a02 = r00 * b02 + r01 * b12 + r02 * b22
        a10 = r10 * b00 + r11 * b10 + r12 * b20
        a11 = r10 * b01 + r11 * b11 + r12 * b21
        a12 = r10 * b02 + r11 * b12 + r12 * b22
        a20 = r20 * b00 + r21 * b10 + r22 * b20
        a21 = r20 * b01 + r21 * b11 + r22 * b21
        a22 = r20 * b02 + r21 * b12 + r22 * b22
        if step > 0 and step % REASSIGN == 0:
            h, k, ell, match = assign(
                a00, a01, a02, a10, a11, a12, a20, a21, a22, qx, qy, qz, valid, TOL
            )
        dx = a00 * h + a01 * k + a02 * ell - qx
        dy = a10 * h + a11 * k + a12 * ell - qy
        dz = a20 * h + a21 * k + a22 * ell - qz
        wm = tl.where(match, w, 0.0)
        per_w = tl.maximum(tl.sum(wm, 0), 1.0)
        count = tl.sum(match.to(tl.int32), 0)
        loss = tl.sum((dx * dx + dy * dy + dz * dz) * wm, 0) / per_w + (
            count < MIN_INDEXED
        ).to(tl.float32)
        tl.store(HISTORY + c * STEPS + step, loss)
        # Gradient of R(omega) @ v, with v = A_anchor @ h.
        ux = b00 * h + b01 * k + b02 * ell
        uy = b10 * h + b11 * k + b12 * ell
        uz = b20 * h + b21 * k + b22 * ell
        cx = y * uz - z * uy
        cy = z * ux - x * uz
        cz = x * uy - y * ux
        ddx = y * cz - z * cy
        ddy = z * cx - x * cz
        ddz = x * cy - y * cx
        gx = 2.0 * wm / per_w * dx
        gy = 2.0 * wm / per_w * dy
        gz = 2.0 * wm / per_w * dz
        vg_x = uy * gz - uz * gy
        vg_y = uz * gx - ux * gz
        vg_z = ux * gy - uy * gx
        cg_x = cy * gz - cz * gy
        cg_y = cz * gx - cx * gz
        cg_z = cx * gy - cy * gx
        go_x = gy * z - gz * y
        go_y = gz * x - gx * z
        go_z = gx * y - gy * x
        vgo_x = uy * go_z - uz * go_y
        vgo_y = uz * go_x - ux * go_z
        vgo_z = ux * go_y - uy * go_x
        adot = (theta * cs - sn) / (theta * theta)
        bdot = (theta * sn - 2.0 * (1.0 - cs)) / (theta * theta * theta)
        radial = tl.where(
            norm2 > 1.0e-24,
            (
                adot * tl.sum(cx * gx + cy * gy + cz * gz, 0)
                + bdot * tl.sum(ddx * gx + ddy * gy + ddz * gz, 0)
            )
            / theta,
            0.0,
        )
        gradx = tl.sum(a * vg_x + b * (cg_x + vgo_x), 0) + x * radial
        grady = tl.sum(a * vg_y + b * (cg_y + vgo_y), 0) + y * radial
        gradz = tl.sum(a * vg_z + b * (cg_z + vgo_z), 0) + z * radial
        mx = mx + (gradx - mx) * 0.1
        my = my + (grady - my) * 0.1
        mz = mz + (gradz - mz) * 0.1
        vx = vx * 0.999 + gradx * gradx * 0.001
        vy = vy * 0.999 + grady * grady * 0.001
        vz = vz * 0.999 + gradz * gradz * 0.001
        step_size = tl.load(BC1 + step)
        bias_sqrt = tl.load(BC2 + step)
        x = x - step_size * mx / (lib.sqrt(vx) / bias_sqrt + 1.0e-8)
        y = y - step_size * my / (lib.sqrt(vy) / bias_sqrt + 1.0e-8)
        z = z - step_size * mz / (lib.sqrt(vz) / bias_sqrt + 1.0e-8)
    r00, r01, r02, r10, r11, r12, r20, r21, r22, _, _, _, _, _, _ = rotation(x, y, z)
    a00 = r00 * b00 + r01 * b10 + r02 * b20
    a01 = r00 * b01 + r01 * b11 + r02 * b21
    a02 = r00 * b02 + r01 * b12 + r02 * b22
    a10 = r10 * b00 + r11 * b10 + r12 * b20
    a11 = r10 * b01 + r11 * b11 + r12 * b21
    a12 = r10 * b02 + r11 * b12 + r12 * b22
    a20 = r20 * b00 + r21 * b10 + r22 * b20
    a21 = r20 * b01 + r21 * b11 + r22 * b21
    a22 = r20 * b02 + r21 * b12 + r22 * b22
    h, k, ell, match = assign(
        a00, a01, a02, a10, a11, a12, a20, a21, a22, qx, qy, qz, valid, TOL
    )
    dx = a00 * h + a01 * k + a02 * ell - qx
    dy = a10 * h + a11 * k + a12 * ell - qy
    dz = a20 * h + a21 * k + a22 * ell - qz
    count = tl.sum(match.to(tl.int32), 0)
    rms = lib.sqrt(
        tl.sum(tl.where(match, dx * dx + dy * dy + dz * dz, 0.0), 0)
        / tl.maximum(count, 1)
    )
    tl.store(AO + c * 9, a00)
    tl.store(AO + c * 9 + 1, a01)
    tl.store(AO + c * 9 + 2, a02)
    tl.store(AO + c * 9 + 3, a10)
    tl.store(AO + c * 9 + 4, a11)
    tl.store(AO + c * 9 + 5, a12)
    tl.store(AO + c * 9 + 6, a20)
    tl.store(AO + c * 9 + 7, a21)
    tl.store(AO + c * 9 + 8, a22)
    tl.store(HK + (c * N + i) * 3, h.to(tl.int64), i < N)
    tl.store(HK + (c * N + i) * 3 + 1, k.to(tl.int64), i < N)
    tl.store(HK + (c * N + i) * 3 + 2, ell.to(tl.int64), i < N)
    tl.store(IDX + c * N + i, match, i < N)
    tl.store(RMS + c, rms)
    tl.store(NI + c, count)
    tl.store(SCORE + c, tl.sum(tl.where(match, w, 0.0), 0))


@lru_cache(maxsize=4)
def _bias_corrections(lr, max_iters, device):
    bc1 = torch.tensor(
        [lr / (1 - 0.9 ** (s + 1)) for s in range(max_iters)],
        device=device,
        dtype=torch.float32,
    )
    bc2 = torch.tensor(
        [math.sqrt(1 - 0.999 ** (s + 1)) for s in range(max_iters)],
        device=device,
        dtype=torch.float32,
    )
    return bc1, bc2


def refine_triton(
    A_init_per_frame,
    q_obs_per_frame,
    *,
    q_tolerance=0.02,
    lr=0.001,
    max_iters=200,
    reassign_every=10,
    min_indexed=6,
    weights_per_frame=None,
):
    assert A_init_per_frame and max_iters > 0
    assert all(a.dtype == torch.float32 and a.is_cuda for a in A_init_per_frame)
    counts = [len(a) for a in A_init_per_frame]
    ns = [len(q) for q in q_obs_per_frame]
    F = len(counts)
    N = triton.next_power_of_2(max(ns))
    C = sum(counts)
    dev = A_init_per_frame[0].device
    A = torch.cat(A_init_per_frame).contiguous()
    Q = torch.zeros(F, N, 3, device=dev, dtype=torch.float32)
    W = torch.zeros(F, N, device=dev, dtype=torch.float32)
    for f, (q, n) in enumerate(zip(q_obs_per_frame, ns)):
        Q[f, :n] = q
        W[f, :n] = (
            weights_per_frame[f]
            if weights_per_frame is not None and weights_per_frame[f] is not None
            else 1.0
        )
    ids = torch.tensor(
        [f for f, k in enumerate(counts) for _ in range(k)],
        device=dev,
        dtype=torch.int32,
    )
    nobs = torch.tensor(ns, device=dev, dtype=torch.int32)
    bc1, bc2 = _bias_corrections(lr, max_iters, dev)
    AO = torch.empty_like(A)
    HK = torch.empty(C, N, 3, device=dev, dtype=torch.int64)
    IDX = torch.empty(C, N, device=dev, dtype=torch.bool)
    RMS = torch.empty(C, device=dev, dtype=torch.float32)
    SCORE = torch.empty_like(RMS)
    NI = torch.empty(C, device=dev, dtype=torch.int64)
    HISTORY = torch.empty(C, max_iters, device=dev, dtype=torch.float32)
    kernel[(C,)](
        A,
        Q,
        W,
        ids,
        nobs,
        bc1,
        bc2,
        AO,
        HK,
        IDX,
        RMS,
        SCORE,
        NI,
        HISTORY,
        N=N,
        STEPS=max_iters,
        REASSIGN=reassign_every,
        MIN_INDEXED=min_indexed,
        TOL=q_tolerance,
        BLOCK=triton.next_power_of_2(N),
        num_warps=4,
        enable_fp_fusion=False,
    )
    history = HISTORY.sum(0).to(device="cpu", dtype=torch.float32)
    results = []
    offset = 0
    for k, n in zip(counts, ns):
        sl = slice(offset, offset + k)
        results.append(
            RefineResult(
                A=AO[sl],
                rmsd=RMS[sl],
                n_indexed=NI[sl],
                soft_score=SCORE[sl],
                indexed=IDX[sl, :n],
                hkl=HK[sl, :n],
                history=history,
            )
        )
        offset += k
    return results
