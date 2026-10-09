import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..peakfinding.peaks.blobs import BlobStats, empty_stats

# fast-path limits on foreground pixels, blobs and pixels per blob; torch beyond
KCAP = 4096
BCAP = 1024
MAX_SIZE = 2048
NAN = tl.constexpr(tl.PropagateNan.ALL)


@triton.jit
def neighbour(
    OUT,
    fg,
    r,
    c,
    i,
    live,
    dr: tl.constexpr,
    dc: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    ok = live & (r + dr >= 0) & (r + dr < H) & (c + dc >= 0) & (c + dc < W)
    node = tl.load(OUT + fg + (dr * W + dc), ok, 0).to(tl.int32)
    return tl.where(node > 0, node - 1, i)


@triton.jit(do_not_specialize=["K"])
def label_nodes(
    FG,
    K,
    OUT,
    START,
    META,
    P,
    NV,
    EXC,
    Z,
    LBF,
    POST,
    VAR,
    MEAN,
    RESP,
    H: tl.constexpr,
    W: tl.constexpr,
    NBLK: tl.constexpr,
    LOGN: tl.constexpr,
    CONN: tl.constexpr,
    ITERS: tl.constexpr,
    HAS_RESP: tl.constexpr,
):
    i = tl.arange(0, NBLK)
    live = i < K
    fg = tl.load(FG + i, live, 0).to(tl.int32)
    r = fg // W
    c = fg - r * W
    tl.store(OUT + fg, (i + 1).to(tl.int64), live)
    tl.debug_barrier()
    n0 = neighbour(OUT, fg, r, c, i, live, -1, 0, H, W)
    n1 = neighbour(OUT, fg, r, c, i, live, 1, 0, H, W)
    n2 = neighbour(OUT, fg, r, c, i, live, 0, -1, H, W)
    n3 = neighbour(OUT, fg, r, c, i, live, 0, 1, H, W)
    if CONN == 2:
        n4 = neighbour(OUT, fg, r, c, i, live, -1, -1, H, W)
        n5 = neighbour(OUT, fg, r, c, i, live, -1, 1, H, W)
        n6 = neighbour(OUT, fg, r, c, i, live, 1, -1, H, W)
        n7 = neighbour(OUT, fg, r, c, i, live, 1, 1, H, W)
    lab = i
    going = 1
    it = 0
    while going > 0:
        new = tl.minimum(lab, tl.gather(lab, n0, 0))
        new = tl.minimum(new, tl.gather(lab, n1, 0))
        new = tl.minimum(new, tl.gather(lab, n2, 0))
        new = tl.minimum(new, tl.gather(lab, n3, 0))
        if CONN == 2:
            new = tl.minimum(new, tl.gather(lab, n4, 0))
            new = tl.minimum(new, tl.gather(lab, n5, 0))
            new = tl.minimum(new, tl.gather(lab, n6, 0))
            new = tl.minimum(new, tl.gather(lab, n7, 0))
        changed = tl.max((new != lab).to(tl.int32), 0)
        lab = new
        it += 1
        going = tl.where(it < ITERS, changed, 0)
    flag = (tl.histogram(lab, NBLK, mask=live) > 0).to(tl.int32)
    bid = tl.gather(tl.cumsum(flag, 0), lab, 0)
    skey = tl.sort(tl.where(live, (bid << LOGN) | i, 2147483647))
    sbid = skey >> LOGN
    first = live & ((i == 0) | (sbid != tl.gather(sbid, tl.maximum(i - 1, 0), 0)))
    nb = tl.sum(flag, 0)
    tl.store(START + sbid - 1, i, first)
    tl.store(START + nb, K)
    tl.store(OUT + fg, bid.to(tl.int64), live)
    tl.store(META, nb)
    # per-pixel inputs of the statistics, blob-major in raster order
    p = tl.gather(fg, skey & (NBLK - 1), 0)
    tl.store(P + i, p, live)
    tl.store(NV + i, tl.load(EXC + p, live, 0.0), live)
    tl.store(NV + NBLK + i, tl.load(Z + p, live, 0.0), live)
    tl.store(NV + 2 * NBLK + i, tl.load(LBF + p, live, 0.0), live)
    tl.store(NV + 3 * NBLK + i, tl.load(POST + p, live, 0.0), live)
    tl.store(NV + 4 * NBLK + i, tl.load(VAR + p, live, 0.0), live)
    tl.store(NV + 5 * NBLK + i, tl.load(MEAN + p, live, 0.0), live)
    if HAS_RESP:
        tl.store(NV + 6 * NBLK + i, tl.load(RESP + p, live, 0.0), live)


@triton.jit(do_not_specialize=["size_min", "size_cap", "size_max", "max_size"])
def blob_stats(
    START,
    META,
    P,
    NV,
    FOUT,
    IOUT,
    KEEP,
    KFOUT,
    KIOUT,
    max_size,
    size_min,
    size_max,
    size_cap,
    ecc_max,
    peak_min,
    inv_thr,
    foot_c,
    H: tl.constexpr,
    W: tl.constexpr,
    NODES: tl.constexpr,
    NBLK: tl.constexpr,
    HAS_RESP: tl.constexpr,
    FOOT: tl.constexpr,
):
    b = tl.arange(0, NBLK)
    nb = tl.load(META)
    lane = b < nb
    start = tl.load(START + b, lane, 0)
    cnt = tl.load(START + b + 1, lane, 0) - start
    maxsz = tl.minimum(tl.max(cnt, 0), max_size)
    tl.store(META + 2, tl.max(cnt, 0) > max_size)
    zero = tl.zeros([NBLK], tl.float32)
    ninf = tl.full([NBLK], float("-inf"), tl.float32)
    s_w = zero
    s_wr = zero
    s_wc = zero
    s_gr = zero
    s_gc = zero
    s_e = zero
    s_v = zero
    s_l = zero
    s_p = zero
    s_m = zero
    e_max = ninf
    z_max = ninf
    r_max = ninf
    r0 = tl.full([NBLK], H, tl.int32)
    r1 = tl.zeros([NBLK], tl.int32)
    c0 = tl.full([NBLK], W, tl.int32)
    c1 = tl.zeros([NBLK], tl.int32)
    for t in range(0, maxsz):
        act = lane & (t < cnt)
        pos = start + t
        p = tl.load(P + pos, act, 0)
        ri = p // W
        ci = p - ri * W
        rf = ri.to(tl.float32)
        cf = ci.to(tl.float32)
        e = tl.load(NV + pos, act, 0.0)
        zv = tl.load(NV + NODES + pos, act, 0.0)
        lb = tl.load(NV + 2 * NODES + pos, act, 0.0)
        po = tl.load(NV + 3 * NODES + pos, act, 0.0)
        va = tl.load(NV + 4 * NODES + pos, act, 0.0)
        me = tl.load(NV + 5 * NODES + pos, act, 0.0)
        w = tl.maximum(e, 0.0, propagate_nan=NAN)
        s_w = tl.where(act, s_w + w, s_w)
        s_wr = tl.where(act, s_wr + w * rf, s_wr)
        s_wc = tl.where(act, s_wc + w * cf, s_wc)
        s_gr = tl.where(act, s_gr + rf, s_gr)
        s_gc = tl.where(act, s_gc + cf, s_gc)
        s_e = tl.where(act, s_e + e, s_e)
        s_v = tl.where(act, s_v + tl.maximum(va, 0.0, propagate_nan=NAN), s_v)
        s_l = tl.where(act, s_l + lb, s_l)
        s_p = tl.where(act, s_p + po, s_p)
        s_m = tl.where(act, s_m + me, s_m)
        e_max = tl.where(act, tl.maximum(e_max, e, propagate_nan=NAN), e_max)
        z_max = tl.where(act, tl.maximum(z_max, zv, propagate_nan=NAN), z_max)
        if HAS_RESP:
            rv = tl.load(NV + 6 * NODES + pos, act, 0.0)
            r_max = tl.where(act, tl.maximum(r_max, rv, propagate_nan=NAN), r_max)
        r0 = tl.where(act, tl.minimum(r0, ri), r0)
        r1 = tl.where(act, tl.maximum(r1, ri + 1), r1)
        c0 = tl.where(act, tl.minimum(c0, ci), c0)
        c1 = tl.where(act, tl.maximum(c1, ci + 1), c1)
    size_f = tl.maximum(cnt, 1).to(tl.float32)
    w_safe = tl.maximum(s_w, 1.0e-12, propagate_nan=NAN)
    has_w = s_w > 0
    rc = tl.where(has_w, tl.div_rn(s_wr, w_safe), tl.div_rn(s_gr, size_f))
    cc = tl.where(has_w, tl.div_rn(s_wc, w_safe), tl.div_rn(s_gc, size_f))
    q_rr = zero
    q_cc = zero
    q_rc = zero
    for t in range(0, maxsz):
        act = lane & (t < cnt)
        pos = start + t
        p = tl.load(P + pos, act, 0)
        ri = p // W
        ci = p - ri * W
        e = tl.load(NV + pos, act, 0.0)
        w = tl.maximum(e, 0.0, propagate_nan=NAN)
        dr = ri.to(tl.float32) - rc
        dc = ci.to(tl.float32) - cc
        wdr = w * dr
        wdc = w * dc
        q_rr = tl.where(act, q_rr + wdr * dr, q_rr)
        q_cc = tl.where(act, q_cc + wdc * dc, q_cc)
        q_rc = tl.where(act, q_rc + wdr * dc, q_rc)
    crr = tl.div_rn(q_rr, w_safe)
    ccc = tl.div_rn(q_cc, w_safe)
    crc = tl.div_rn(q_rc, w_safe)
    tr = crr + ccc
    disc = tl.sqrt_rn(
        tl.maximum(
            (tr * tr) - 4.0 * ((crr * ccc) - (crc * crc)), 0.0, propagate_nan=NAN
        )
    )
    lam_max = 0.5 * (tr + disc)
    lam_min = 0.5 * (tr - disc)
    ecc = tl.where(
        lam_min > 1.0e-12,
        tl.div_rn(lam_max, tl.maximum(lam_min, 1.0e-12, propagate_nan=NAN)),
        float("inf"),
    )
    mean_i = tl.div_rn(s_e, size_f)
    peaked = tl.where(
        mean_i > 1.0e-12,
        tl.div_rn(e_max, tl.maximum(mean_i, 1.0e-12, propagate_nan=NAN)),
        0.0,
    )
    sigma = tl.sqrt_rn(s_v)
    post_mean = tl.div_rn(s_p, size_f)
    if FOOT:
        ratio = tl.maximum(r_max * inv_thr, 1.0, propagate_nan=NAN)
        cap = tl.maximum(size_cap, foot_c * libdevice.log(ratio), propagate_nan=NAN)
        size_ok = cnt.to(tl.float32) <= cap
    else:
        size_ok = cnt <= size_max
    keep = (cnt >= size_min) & size_ok & (ecc <= ecc_max) & (peaked >= peak_min) & lane
    tl.store(KEEP + b, keep.to(tl.int8), lane)
    kidx = tl.cumsum(keep.to(tl.int32), 0) - 1
    tl.store(META + 1, tl.sum(keep.to(tl.int32), 0))
    tl.store(IOUT + b, b + 1, lane)
    tl.store(IOUT + NBLK + b, cnt, lane)
    tl.store(IOUT + 2 * NBLK + b, r0, lane)
    tl.store(IOUT + 3 * NBLK + b, r1, lane)
    tl.store(IOUT + 4 * NBLK + b, c0, lane)
    tl.store(IOUT + 5 * NBLK + b, c1, lane)
    tl.store(KIOUT + kidx, b + 1, keep)
    tl.store(KIOUT + NBLK + kidx, cnt, keep)
    tl.store(KIOUT + 2 * NBLK + kidx, r0, keep)
    tl.store(KIOUT + 3 * NBLK + kidx, r1, keep)
    tl.store(KIOUT + 4 * NBLK + kidx, c0, keep)
    tl.store(KIOUT + 5 * NBLK + kidx, c1, keep)
    tl.store(FOUT + b, rc, lane)
    tl.store(FOUT + NBLK + b, cc, lane)
    tl.store(FOUT + 2 * NBLK + b, s_e, lane)
    tl.store(FOUT + 3 * NBLK + b, sigma, lane)
    tl.store(FOUT + 4 * NBLK + b, e_max, lane)
    tl.store(FOUT + 5 * NBLK + b, z_max, lane)
    tl.store(FOUT + 6 * NBLK + b, s_l, lane)
    tl.store(FOUT + 7 * NBLK + b, post_mean, lane)
    tl.store(FOUT + 8 * NBLK + b, ecc, lane)
    tl.store(FOUT + 9 * NBLK + b, peaked, lane)
    tl.store(FOUT + 10 * NBLK + b, s_m, lane)
    tl.store(KFOUT + kidx, rc, keep)
    tl.store(KFOUT + NBLK + kidx, cc, keep)
    tl.store(KFOUT + 2 * NBLK + kidx, s_e, keep)
    tl.store(KFOUT + 3 * NBLK + kidx, sigma, keep)
    tl.store(KFOUT + 4 * NBLK + kidx, e_max, keep)
    tl.store(KFOUT + 5 * NBLK + kidx, z_max, keep)
    tl.store(KFOUT + 6 * NBLK + kidx, s_l, keep)
    tl.store(KFOUT + 7 * NBLK + kidx, post_mean, keep)
    tl.store(KFOUT + 8 * NBLK + kidx, ecc, keep)
    tl.store(KFOUT + 9 * NBLK + kidx, peaked, keep)
    tl.store(KFOUT + 10 * NBLK + kidx, s_m, keep)
    if HAS_RESP:
        tl.store(FOUT + 11 * NBLK + b, r_max, lane)
        tl.store(KFOUT + 11 * NBLK + kidx, r_max, keep)


def stats_from(ibuf, fbuf, n, has_resp):
    i = ibuf[:, :n].unbind(0)
    f = fbuf[:, :n].unbind(0)
    return BlobStats(
        label_id=i[0],
        size=i[1],
        row_centroid=f[0],
        col_centroid=f[1],
        bbox_r0=i[2],
        bbox_r1=i[3],
        bbox_c0=i[4],
        bbox_c1=i[5],
        intensity_sum=f[2],
        intensity_sigma=f[3],
        intensity_max=f[4],
        z_max=f[5],
        log_bf_sum=f[6],
        posterior_mean=f[7],
        eccentricity=f[8],
        peakedness=f[9],
        background_sum=f[10],
        response_max=f[11] if has_resp else None,
    )


@torch.no_grad()
def extract(finder, binary, scores, frame_index, mask):
    from ..peakfinding.peaks.peakfinder import PeakResult

    H, W = binary.shape
    device = binary.device
    dtype = scores["excess"].dtype
    fg = binary.view(-1).nonzero()
    K = fg.shape[0]
    labels = torch.zeros(H, W, dtype=torch.long, device=device)

    def empty():
        return PeakResult(
            frame_index=frame_index,
            scores=scores,
            labels=labels,
            stats=empty_stats(device, dtype),
            keep=torch.zeros(0, dtype=torch.bool, device=device),
        )

    if K == 0:
        return empty()
    if K > KCAP or not 0 < finder.max_peaks <= BCAP or finder.size_min < 1:
        return None
    nodes = 256 if K <= 256 else 1024 if K <= 1024 else 4096
    lanes = min(nodes, BCAP)
    warps = 4 if nodes <= 256 else 8 if nodes <= 1024 else 16
    has_resp = finder.matched_filter and "mf_max" in scores
    iscratch = torch.empty(2 * nodes + 5, dtype=torch.int32, device=device)
    start, meta = iscratch[: nodes + 1], iscratch[nodes + 1 : nodes + 4]
    pix = iscratch[nodes + 4 :]
    nv = torch.empty((7, nodes), dtype=dtype, device=device)
    fbuf = torch.empty((2, 12 if has_resp else 11, lanes), dtype=dtype, device=device)
    ibuf = torch.empty((2, 6, lanes), dtype=torch.long, device=device)
    keep_buf = torch.empty(lanes, dtype=torch.uint8, device=device)
    label_nodes[(1,)](
        fg,
        K,
        labels,
        start,
        meta,
        pix,
        nv,
        scores["excess"],
        scores["z"],
        scores["log_bf"],
        scores["posterior"],
        scores["var_eff"],
        scores["mean_eff"],
        scores["mf_max"] if has_resp else scores["excess"],
        H=H,
        W=W,
        NBLK=nodes,
        LOGN=nodes.bit_length() - 1,
        CONN=1 if finder.connectivity == 1 else 2,
        ITERS=64,
        HAS_RESP=has_resp,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    foot = has_resp and finder.mf_threshold > 0
    scale = max(finder.mf_scales) if foot else 0.0
    blob_stats[(1,)](
        start,
        meta,
        pix,
        nv,
        fbuf[0],
        ibuf[0],
        keep_buf,
        fbuf[1],
        ibuf[1],
        MAX_SIZE,
        finder.size_min,
        finder.size_max,
        float(finder.size_max),
        finder.eccentricity_max,
        finder.peakedness_min,
        float(np.float32(1.0 / finder.mf_threshold)) if foot else 0.0,
        4.0 * 2.0 * math.pi * scale * scale,
        H=H,
        W=W,
        NODES=nodes,
        NBLK=lanes,
        HAS_RESP=has_resp,
        FOOT=foot,
        num_warps=min(warps, 8),
        enable_fp_fusion=False,
    )
    n_blobs, n_keep, oversize = meta.tolist()
    if oversize:
        return None
    if n_blobs > finder.max_peaks:
        return empty()
    result = PeakResult(
        frame_index=frame_index,
        scores=scores,
        labels=labels,
        stats=stats_from(ibuf[0], fbuf[0], n_blobs, has_resp),
        keep=keep_buf[:n_blobs].view(torch.bool),
        var=scores["var_eff"],
        valid_mask=mask,
        mean=scores["mean_eff"],
    )
    result.kept_stats = stats_from(ibuf[1], fbuf[1], n_keep, has_resp)
    return result
