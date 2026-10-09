import torch
import triton
import triton.language as tl

from . import tensor_key

NAN = tl.constexpr(tl.PropagateNan.ALL)


@triton.jit
def pixel_update(
    F, MEAN, M2, N, CLIP, n_elem, robust_k, ROBUST: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    live = i < n_elem
    x = tl.load(F + i, live, 0.0)
    m = tl.load(MEAN + i, live, 0.0)
    s = tl.load(M2 + i, live, 0.0)
    n = tl.load(N)
    if ROBUST:
        var = tl.where(n >= 2, tl.div_rn(s, tl.maximum(n - 1, 1).to(tl.float32)), 0.0)
        std = tl.sqrt_rn(tl.maximum(var, 0.0, propagate_nan=NAN))
        x = tl.where(std > 0, tl.minimum(x, m + robust_k * std, propagate_nan=NAN), x)
        tl.store(CLIP + i, x, live)
    delta = x - m
    m = m + tl.div_rn(delta, (n + 1).to(tl.float32))
    tl.store(MEAN + i, m, live)
    tl.store(M2 + i, s + delta * (x - m), live)


@triton.jit
def welford(X, MEAN, M2, N, n_elem, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    live = i < n_elem
    x = tl.load(X + i, live, 0.0)
    m = tl.load(MEAN + i, live, 0.0)
    s = tl.load(M2 + i, live, 0.0)
    delta = x - m
    m = m + tl.div_rn(delta, tl.load(N).to(tl.float32))
    tl.store(MEAN + i, m, live)
    tl.store(M2 + i, s + delta * (x - m), live)


@triton.jit
def source_mean(
    code: tl.constexpr,
    coef,
    i,
    live,
    PM,
    RM,
    BIN,
    AM,
    AVALID,
    APID,
    SINGLE: tl.constexpr,
):
    if code == 0:
        v = tl.load(PM + i, live, 0.0)
    elif code == 1:
        v = tl.load(RM + tl.load(BIN + i, live, 0), live, 0.0)
    else:
        if SINGLE:
            a = tl.load(AM)
        else:
            a = tl.load(AM + tl.load(APID + i, live, 0), live, 0.0)
        v = tl.where(tl.load(AVALID + i, live, 0), a, 0.0)
    return coef * v


@triton.jit
def predict(
    PM,
    PS,
    PN,
    RM,
    BIN,
    AM,
    AVALID,
    APID,
    MEAN,
    VAR,
    n_elem,
    w0,
    w1,
    w2,
    var_scale,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    SINGLE: tl.constexpr,
    SCALE: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    live = i < n_elem
    mean = source_mean(C0, w0, i, live, PM, RM, BIN, AM, AVALID, APID, SINGLE)
    if C1 >= 0:
        mean = mean + source_mean(
            C1, w1, i, live, PM, RM, BIN, AM, AVALID, APID, SINGLE
        )
    if C2 >= 0:
        mean = mean + source_mean(
            C2, w2, i, live, PM, RM, BIN, AM, AVALID, APID, SINGLE
        )
    n = tl.load(PN)
    s = tl.load(PS + i, live, 0.0)
    var = tl.where(n >= 2, tl.div_rn(s, tl.maximum(n - 1, 1).to(tl.float32)), 0.0)
    if SCALE:
        var = var * var_scale
    tl.store(MEAN + i, mean, live)
    tl.store(VAR + i, var, live)


def update_pixel(model, frame):
    p = model.pixel
    robust = model.robust_update and model._n_host >= model.robust_min_frames
    clip = torch.empty_like(frame) if robust else frame
    n = frame.numel()
    pixel_update[(triton.cdiv(n, 1024),)](
        frame,
        p.mean_,
        p.M2_,
        p.n_,
        clip,
        n,
        model.robust_k,
        ROBUST=robust,
        B=1024,
        enable_fp_fusion=False,
    )
    p.n_ += 1
    return clip


def update_stat(stat, x):
    n = x.numel()
    welford[(triton.cdiv(n, 1024),)](
        x, stat.mean_, stat.M2_, stat.n_, n, B=1024, enable_fp_fusion=False
    )


def cached(owner, name, tensor):
    key = tensor_key(tensor)
    hit = getattr(owner, "_i32_cache", {}).get(name)
    if hit is None or hit[0] != key:
        hit = (key, tensor.to(torch.int32).contiguous())
        if not hasattr(owner, "_i32_cache"):
            owner._i32_cache = {}
        owner._i32_cache[name] = hit
    return hit[1]


CODES = {"pixel": 0, "rotational": 1, "panel": 2}


def predict_maps(model, weights, total, scaled):
    p, r, a = model.pixel, model.rotational, model.panel
    live = [(CODES[k], w / total) for k, w in weights.items() if w > 0]
    codes = [c for c, _ in live] + [-1] * (3 - len(live))
    coefs = [w for _, w in live] + [0.0] * (3 - len(live))
    single = a.n_panels == 1
    mean = torch.empty_like(p.mean_)
    var = torch.empty_like(p.mean_)
    n = mean.numel()
    predict[(triton.cdiv(n, 1024),)](
        p.mean_,
        p.M2_,
        p.n_,
        r.mean_,
        cached(r, "bin", r.bin_idx) if 1 in codes else r.mean_,
        a.mean_,
        a.valid_mask.contiguous(),
        a._pid_flat if single or 2 not in codes else cached(a, "pid", a._pid_flat),
        mean,
        var,
        n,
        *coefs,
        model.var_scale**2,
        *codes,
        SINGLE=single,
        SCALE=scaled,
        B=1024,
        enable_fp_fusion=False,
    )
    return mean, var
