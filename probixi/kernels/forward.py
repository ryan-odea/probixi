import torch
import triton
import triton.language as tl

BLOCK = 128


# CONST = [bc_row, bc_col, pix_A, clen_A, wavelength_A] (see indexer.forward._constants)
@triton.jit(do_not_specialize=["N", "P"])
def lift_kernel(POS, BASES, CONST, OUT, N, P, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    row = tl.load(POS + i * 2, valid, 0.0)
    col = tl.load(POS + i * 2 + 1, valid, 0.0)
    pix = tl.load(CONST + 2)
    clen = tl.load(CONST + 3)
    x = col - tl.load(CONST + 1)
    y = row - tl.load(CONST)
    for p in range(P):
        b = BASES + p * 10
        inside = (
            (row >= tl.load(b))
            & (row <= tl.load(b + 1))
            & (col >= tl.load(b + 2))
            & (col <= tl.load(b + 3))
        )
        fs_j = col - tl.load(b + 2)
        ss_i = row - tl.load(b)
        x = tl.where(
            inside,
            tl.load(b + 4) + fs_j * tl.load(b + 6) + ss_i * tl.load(b + 8),
            x,
        )
        y = tl.where(
            inside,
            tl.load(b + 5) + fs_j * tl.load(b + 7) + ss_i * tl.load(b + 9),
            y,
        )
    x_lab = x * pix
    y_lab = y * pix
    r = tl.sqrt_rn(x_lab * x_lab + y_lab * y_lab + clen * clen)
    inv_lambda = tl.div_rn(1.0, tl.load(CONST + 4))
    tl.store(OUT + i * 3, tl.div_rn(x_lab, r) * inv_lambda, valid)
    tl.store(OUT + i * 3 + 1, tl.div_rn(y_lab, r) * inv_lambda, valid)
    tl.store(OUT + i * 3 + 2, (tl.div_rn(clen, r) - 1.0) * inv_lambda, valid)


@triton.jit(do_not_specialize=["N", "P"])
def project_kernel(Q, BASES, CONST, OUT, N, P, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    pix = tl.load(CONST + 2)
    inv_lambda = tl.div_rn(1.0, tl.load(CONST + 4))
    sx = tl.div_rn(tl.load(Q + i * 3, valid, 0.0), inv_lambda)
    sy = tl.div_rn(tl.load(Q + i * 3 + 1, valid, 0.0), inv_lambda)
    sz = tl.div_rn(tl.load(Q + i * 3 + 2, valid, 0.0), inv_lambda) + 1.0
    scale = tl.div_rn(
        tl.load(CONST + 3),
        tl.maximum(sz, 1.0e-30, propagate_nan=tl.PropagateNan.ALL),
    )
    x_pix = tl.div_rn(sx * scale, pix)
    y_pix = tl.div_rn(sy * scale, pix)
    if P == 0:
        row = y_pix + tl.load(CONST)
        col = x_pix + tl.load(CONST + 1)
    else:
        row = tl.full((BLOCK,), float("nan"), tl.float32)
        col = tl.full((BLOCK,), float("nan"), tl.float32)
        found = i < 0
        for p in range(P):
            b = BASES + p * 10
            cx = tl.load(b + 4)
            cy = tl.load(b + 5)
            fsx = tl.load(b + 6)
            fsy = tl.load(b + 7)
            ssx = tl.load(b + 8)
            ssy = tl.load(b + 9)
            det = fsx * ssy - ssx * fsy
            rx = x_pix - cx
            ry = y_pix - cy
            fs_j = tl.div_rn(ssy * rx - ssx * ry, det)
            ss_i = tl.div_rn(-fsy * rx + fsx * ry, det)
            min_ss = tl.load(b)
            min_fs = tl.load(b + 2)
            on = (
                (sz > 0.0)
                & (fs_j >= 0)
                & (fs_j <= tl.load(b + 3) - min_fs)
                & (ss_i >= 0)
                & (ss_i <= tl.load(b + 1) - min_ss)
            )
            take = on & ~found
            row = tl.where(take, min_ss + ss_i, row)
            col = tl.where(take, min_fs + fs_j, col)
            found = found | on
    tl.store(OUT + i * 2, row, valid)
    tl.store(OUT + i * 2 + 1, col, valid)


def lift(pos, bases, consts):
    pos = pos.contiguous()
    out = torch.empty(len(pos), 3, device=pos.device, dtype=torch.float32)
    lift_kernel[(triton.cdiv(len(pos), BLOCK),)](
        pos,
        pos if bases is None else bases,
        consts,
        out,
        len(pos),
        0 if bases is None else len(bases),
        BLOCK=BLOCK,
        enable_fp_fusion=False,
    )
    return out


def project(q, bases, consts):
    q = q.contiguous()
    out = torch.empty(len(q), 2, device=q.device, dtype=torch.float32)
    project_kernel[(triton.cdiv(len(q), BLOCK),)](
        q,
        q if bases is None else bases,
        consts,
        out,
        len(q),
        0 if bases is None else len(bases),
        BLOCK=BLOCK,
        enable_fp_fusion=False,
    )
    return out
