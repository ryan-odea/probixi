import math

import torch
import triton
import triton.language as tl

from probixi.peakfinding.peaks.neighborhood import _separable_1d


@triton.jit
def prepare(F, MEAN, VAR, MASK, RM, RM2, N: tl.constexpr, B: tl.constexpr):
    x = tl.program_id(0) * B + tl.arange(0, B)
    f = tl.load(F + x, x < N, 0).to(tl.float32)
    m = tl.load(MEAN + x, x < N, 0)
    v = tl.maximum(tl.load(VAR + x, x < N, 0), 1.0e-12)
    valid = tl.load(MASK + x, x < N, 0)
    r = tl.minimum(f - m, 5.0 * tl.sqrt(v))
    rm = r * valid.to(tl.float32)
    tl.store(RM + x, rm, x < N)
    tl.store(RM2 + x, r * rm, x < N)


@triton.jit
def background_vertical(
    RM,
    RM2,
    SO,
    SI,
    S2O,
    S2I,
    H: tl.constexpr,
    W: tl.constexpr,
    INNER: tl.constexpr,
    OUTER: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row = i // W
    so = tl.full((B,), 0, tl.float32)
    si = tl.full((B,), 0, tl.float32)
    s2o = tl.full((B,), 0, tl.float32)
    s2i = tl.full((B,), 0, tl.float32)
    for d in tl.static_range(-OUTER, OUTER + 1):
        valid = (i < H * W) & (row + d >= 0) & (row + d < H)
        a = tl.load(RM + i + d * W, valid, 0)
        b = tl.load(RM2 + i + d * W, valid, 0)
        so += a
        s2o += b
        if (d >= -INNER) & (d <= INNER):
            si += a
            s2i += b
    tl.store(SO + i, so, i < H * W)
    tl.store(SI + i, si, i < H * W)
    tl.store(S2O + i, s2o, i < H * W)
    tl.store(S2I + i, s2i, i < H * W)


@triton.jit
def background_score(
    F,
    MEAN,
    VAR,
    MASK,
    COUNT,
    SO,
    SI,
    S2O,
    S2I,
    MEAN_EFF,
    VAR_EFF,
    EXCESS,
    Z,
    BF,
    LOGITS,
    ZM,
    H: tl.constexpr,
    W: tl.constexpr,
    INNER: tl.constexpr,
    OUTER: tl.constexpr,
    FLUX: tl.constexpr,
    GAIN,
    READ,
    FLOOR,
    LOGK,
    BF_FACTOR,
    PRIOR,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    col = i % W
    so = tl.full((B,), 0, tl.float32)
    si = tl.full((B,), 0, tl.float32)
    s2o = tl.full((B,), 0, tl.float32)
    s2i = tl.full((B,), 0, tl.float32)
    for d in tl.static_range(-OUTER, OUTER + 1):
        valid = (i < H * W) & (col + d >= 0) & (col + d < W)
        so += tl.load(SO + i + d, valid, 0)
        s2o += tl.load(S2O + i + d, valid, 0)
        if (d >= -INNER) & (d <= INNER):
            si += tl.load(SI + i + d, valid, 0)
            s2i += tl.load(S2I + i + d, valid, 0)
    count = tl.load(COUNT + i, i < H * W, 1)
    lm = tl.div_rn(so - si, count)
    ex2 = tl.div_rn(s2o - s2i, count)
    lv = tl.maximum(ex2 - lm * lm, 0.0)
    mean = tl.load(MEAN + i, i < H * W, 0) + lm
    var0 = tl.maximum(tl.load(VAR + i, i < H * W, 0), 1.0e-12)
    var = tl.maximum(var0, lv)
    if FLUX:
        base = READ + GAIN * tl.maximum(mean, 0.0)
        base = tl.maximum(tl.maximum(base, FLOOR * var0), 1.0e-12)
        var = tl.maximum(base, lv)
    f = tl.load(F + i, i < H * W, 0).to(tl.float32)
    mask = tl.load(MASK + i, i < H * W, 0)
    excess = f - mean
    z = tl.div_rn(excess, tl.sqrt(var))
    raw = -LOGK + ((0.5 * z) * z) * BF_FACTOR
    bf = tl.where(mask & (f > mean), raw, 0.0)
    tl.store(MEAN_EFF + i, mean, i < H * W)
    tl.store(VAR_EFF + i, var, i < H * W)
    tl.store(EXCESS + i, excess, i < H * W)
    tl.store(Z + i, z, i < H * W)
    tl.store(BF + i, bf, i < H * W)
    tl.store(LOGITS + i, (bf + PRIOR) * mask.to(tl.float32), i < H * W)
    tl.store(ZM + i, z * mask.to(tl.float32), i < H * W)


@triton.jit
def filter_vertical(
    ZM,
    LOGITS,
    A0,
    A1,
    A2,
    AP,
    T0,
    T1,
    T2,
    TP,
    H: tl.constexpr,
    W: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    RP: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row = i // W
    s0 = tl.full((B,), 0, tl.float32)
    s1 = tl.full((B,), 0, tl.float32)
    s2 = tl.full((B,), 0, tl.float32)
    sp = tl.full((B,), 0, tl.float32)
    # Matched filters zero-pad at the border, posterior smoothing reflects.
    for d in tl.static_range(-R2, R2 + 1):
        v = tl.load(ZM + i + d * W, (i < H * W) & (row + d >= 0) & (row + d < H), 0)
        if (d >= -R0) & (d <= R0):
            s0 += v * tl.load(A0 + d + R0)
        if (d >= -R1) & (d <= R1):
            s1 += v * tl.load(A1 + d + R1)
        s2 += v * tl.load(A2 + d + R2)
    for d in tl.static_range(-RP, RP + 1):
        r = row + d
        r = tl.where(r < 0, -r, tl.where(r >= H, 2 * H - 2 - r, r))
        v = tl.load(LOGITS + r * W + i % W, i < H * W, 0)
        sp += v * tl.load(AP + d + RP)
    tl.store(T0 + i, s0, i < H * W)
    tl.store(T1 + i, s1, i < H * W)
    tl.store(T2 + i, s2, i < H * W)
    tl.store(TP + i, sp, i < H * W)


@triton.jit
def filter_horizontal(
    T0,
    T1,
    T2,
    TP,
    A0,
    A1,
    A2,
    AP,
    D0,
    D1,
    D2,
    DP,
    MASK,
    MF,
    POST,
    H: tl.constexpr,
    W: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    RP: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    col = i % W
    s0 = tl.full((B,), 0, tl.float32)
    s1 = tl.full((B,), 0, tl.float32)
    s2 = tl.full((B,), 0, tl.float32)
    sp = tl.full((B,), 0, tl.float32)
    # Matched filters zero-pad at the border, posterior smoothing reflects.
    for d in tl.static_range(-R2, R2 + 1):
        valid = (i < H * W) & (col + d >= 0) & (col + d < W)
        if (d >= -R0) & (d <= R0):
            s0 += tl.load(T0 + i + d, valid, 0) * tl.load(A0 + d + R0)
        if (d >= -R1) & (d <= R1):
            s1 += tl.load(T1 + i + d, valid, 0) * tl.load(A1 + d + R1)
        s2 += tl.load(T2 + i + d, valid, 0) * tl.load(A2 + d + R2)
    for d in tl.static_range(-RP, RP + 1):
        c = col + d
        c = tl.where(c < 0, -c, tl.where(c >= W, 2 * W - 2 - c, c))
        sp += tl.load(TP + (i // W) * W + c, i < H * W, 0) * tl.load(AP + d + RP)
    z0 = tl.div_rn(s0, tl.sqrt(tl.maximum(tl.load(D0 + i, i < H * W, 1), 1.0e-12)))
    z1 = tl.div_rn(s1, tl.sqrt(tl.maximum(tl.load(D1 + i, i < H * W, 1), 1.0e-12)))
    z2 = tl.div_rn(s2, tl.sqrt(tl.maximum(tl.load(D2 + i, i < H * W, 1), 1.0e-12)))
    logit = tl.div_rn(sp, tl.load(DP + i, i < H * W, 1))
    post = tl.div_rn(1.0, 1.0 + tl.exp(-logit))
    mask = tl.load(MASK + i, i < H * W, 0)
    tl.store(MF + i, tl.maximum(tl.maximum(z0, z1), z2), i < H * W)
    tl.store(POST + i, tl.where(mask, post, 0.0), i < H * W)


class FusedScorer:
    def __init__(self, finder):
        self.finder = finder
        assert finder.local_background and finder.matched_filter
        assert 1 <= finder.local_inner_radius < finder.local_outer_radius
        assert len(finder._mf_kernels) == 3
        assert finder.noise.eigen_modes is None
        self.a = [
            _separable_1d(k / k.norm().clamp_min(1.0e-12)) for k in finder._mf_kernels
        ]
        self.ap = _separable_1d(finder._kernel)
        self.r = [a.numel() // 2 for a in self.a]
        assert self.r == sorted(self.r)

    @torch.no_grad()
    def __call__(self, frame):
        f = self.finder
        pred = f._pred()
        assert frame.ndim == 2 and frame.is_contiguous() and frame.is_cuda
        assert pred["mean"].dtype == torch.float32
        h, w = frame.shape
        assert min(h, w) > self.ap.numel() // 2

        def alloc():
            return torch.empty((h, w), device=frame.device, dtype=torch.float32)

        rm, rm2 = alloc(), alloc()
        so, si, s2o, s2i = [alloc() for _ in range(4)]
        maps = {
            k: alloc()
            for k in [
                "mean_eff",
                "var_eff",
                "excess",
                "z",
                "log_bf",
                "mf_max",
                "posterior",
            ]
        }
        logits, zm = alloc(), alloc()
        ts = [alloc() for _ in range(4)]

        def grid(meta):
            return (triton.cdiv(h * w, meta["B"]),)

        opts = {"enable_fp_fusion": False}
        prepare[grid](
            frame,
            pred["mean"],
            pred["var"],
            pred["mask"],
            rm,
            rm2,
            h * w,
            B=256,
            **opts,
        )
        background_vertical[grid](
            rm,
            rm2,
            so,
            si,
            s2o,
            s2i,
            h,
            w,
            f.local_inner_radius,
            f.local_outer_radius,
            B=128,
            **opts,
        )
        flux = f.flux_variance and f.noise.gain is not None
        background_score[grid](
            frame,
            pred["mean"],
            pred["var"],
            pred["mask"],
            f._annulus_count,
            so,
            si,
            s2o,
            s2i,
            maps["mean_eff"],
            maps["var_eff"],
            maps["excess"],
            maps["z"],
            maps["log_bf"],
            logits,
            zm,
            h,
            w,
            f.local_inner_radius,
            f.local_outer_radius,
            flux,
            f.noise.gain or 0.0,
            f.noise.read_var or 0.0,
            f.flux_var_floor,
            math.log(f.kappa),
            1.0 - 1.0 / (f.kappa * f.kappa),
            f.log_prior_odds,
            B=128,
            **opts,
        )
        radii = {
            "H": h,
            "W": w,
            "R0": self.r[0],
            "R1": self.r[1],
            "R2": self.r[2],
            "RP": self.ap.numel() // 2,
            "B": 128,
        }
        filter_vertical[grid](zm, logits, *self.a, self.ap, *ts, **radii, **opts)
        filter_horizontal[grid](
            *ts,
            *self.a,
            self.ap,
            *f._mf_bank_den,
            f._smooth_den,
            pred["mask"],
            maps["mf_max"],
            maps["posterior"],
            **radii,
            **opts,
        )
        return maps
