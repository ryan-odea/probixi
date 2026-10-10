import math

import numpy as np
import torch

from probixi.indexer.lattice import B_to_cell
from probixi.indexer.refine import _assign_hkls, _cell_dofs
from probixi.io.cell import CellParams


def _rotation(x, y, z):
    # Rodrigues R(w) and the left Jacobian Jl(w), with dR/dw_j = [Jl e_j]x R
    s2 = x * x + y * y + z * z
    t2 = max(s2, 1e-24)
    t = math.sqrt(t2)
    sn = math.sin(t)
    a = sn / t
    b = (1.0 - math.cos(t)) / t2
    c = (t - sn) / (t2 * t)
    xy, xz, yz = x * y, x * z, y * z
    xx, yy, zz = x * x, y * y, z * z
    R = (
        1.0 - b * (yy + zz),
        b * xy - a * z,
        b * xz + a * y,
        b * xy + a * z,
        1.0 - b * (xx + zz),
        b * yz - a * x,
        b * xz - a * y,
        b * yz + a * x,
        1.0 - b * (xx + yy),
    )
    J = (
        1.0 - c * (yy + zz),
        c * xy - b * z,
        c * xz + b * y,
        c * xy + b * z,
        1.0 - c * (xx + zz),
        c * yz - b * x,
        c * xz - b * y,
        c * yz + b * x,
        1.0 - c * (xx + yy),
    )
    cross = []
    for j in range(3):
        v0, v1, v2 = J[j], J[3 + j], J[6 + j]
        cross += (0.0, -v2, v1, v2, 0.0, -v0, -v1, v0, 0.0)
    return R, cross


def _cell_basis(p, cols):
    # B = M^-T (lower triangular) and G_l = -dM_l^T so that dB_l = B G_l B
    a, b, c, al, be, ga = p
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    sa, sb, sg = math.sin(al), math.sin(be), math.sin(ga)
    qy = (ca - cb * cg) / sg
    cx = c * cb
    cy = c * qy
    cz2 = c * c - cx * cx - cy * cy
    live = cz2 >= 1e-12
    cz = math.sqrt(cz2) if live else 1e-6
    B = (
        1.0 / a,
        0.0,
        0.0,
        -cg / (a * sg),
        1.0 / (b * sg),
        0.0,
        (cg * cy - cx * sg) / (a * sg * cz),
        -cy / (b * sg * cz),
        1.0 / cz,
    )
    dcy_al = -c * sa / sg
    dcy_be = c * sb * cg / sg
    dcy_ga = c * (cb - ca * cg) / (sg * sg)
    dcx_be = -c * sb
    if live:
        dcz = (
            (c - cx * cb - cy * qy) / cz,
            -cy * dcy_al / cz,
            -(cx * dcx_be + cy * dcy_be) / cz,
            -cy * dcy_ga / cz,
        )
    else:
        dcz = (0.0, 0.0, 0.0, 0.0)
    dM = (
        ((0, 0, 1.0),),
        ((0, 1, cg), (1, 1, sg)),
        ((0, 2, cb), (1, 2, qy), (2, 2, dcz[0])),
        ((1, 2, dcy_al), (2, 2, dcz[1])),
        ((0, 2, dcx_be), (1, 2, dcy_be), (2, 2, dcz[2])),
        ((0, 1, -b * sg), (1, 1, b * cg), (1, 2, dcy_ga), (2, 2, dcz[3])),
    )
    G = []
    for ks in cols:
        g = [0.0] * 9
        for k in ks:
            for i, j, v in dM[k]:
                g[3 * j + i] -= v
        G += g
    return B, G


def _gram(ext):
    ext = ext.astype(np.float32)
    return ext @ ext.T


def _evaluator(U0, p0, cols, T, prior_width, hkl, q, khat, s_perp, s_par):
    # x -> (A, [f; J^T]) in float64 for f = [(1 - k k^T)(h A^T - q) / s_perp,
    # k.(h A^T - q) / s_par, T r / prior_width] and J = df/dx
    n, d = len(hkl), len(cols)
    W = np.empty((n, 4, 3))
    W[:, :3] = (np.eye(3) - khat[:, :, None] * khat[:, None, :]) / s_perp
    W[:, 3] = khat / s_par
    K = (W[:, :, :, None] * hkl[:, None, None, :]).reshape(4 * n, 9).T.copy()
    Wq = np.einsum("nrc,nc->nr", W, q).reshape(-1)
    Tw = T / prior_width[:, None]
    Jprior = np.zeros((3 + d, 6))
    Jprior[3:] = Tw.T

    def evaluate(x):
        R, cross = _rotation(*x[:3])
        p = list(p0)
        for j, ks in enumerate(cols):
            for k in ks:
                p[k] += x[3 + j]
        B, G = _cell_basis(p, cols)
        S = np.array([*R, *cross, *B, *G]).reshape(-1, 3)
        Bm = S[12:15]
        A = S[:3] @ (U0 @ Bm)
        dA = np.concatenate(
            [
                A.reshape(1, 9),
                (S[3:12] @ A).reshape(3, 9),
                (A @ (S[15:] @ Bm).reshape(d, 3, 3)).reshape(d, 9),
            ]
        )
        ext = np.empty((4 + d, 4 * n + 6))
        np.matmul(dA, K, out=ext[:, : 4 * n])
        ext[0, : 4 * n] -= Wq
        ext[0, 4 * n :] = Tw @ x[3:]
        ext[1:, 4 * n :] = Jprior
        return A, ext

    return evaluate


@np.errstate(all="ignore")
def refine_analytic(
    A,
    q,
    hkl,
    indexed,
    template,
    q_tolerance,
    iters=8,
    rounds=2,
    edge_tolerance=0.05,
    angle_tolerance_rad=math.radians(3.0),
    wavelength=None,
):
    dtype, out_device = A.dtype, A.device
    A_d, q_d, hkl_d, indexed = (t.detach().to("cpu") for t in (A, q, hkl, indexed))
    try:
        c0 = B_to_cell(A_d)
    except Exception:
        return None
    wd = torch.float32
    A_d, q_d, hkl_d = (t.to(wd) for t in (A_d, q_d, hkl_d))
    cell0 = CellParams(
        c0.a,
        c0.b,
        c0.c,
        c0.alpha,
        c0.beta,
        c0.gamma,
        lattice_type=template.lattice_type,
        unique_axis=template.unique_axis,
        centering=template.centering,
    )
    T, p0 = _cell_dofs(cell0, wd, torch.device("cpu"))
    cols = [[k for k, v in enumerate(col) if v] for col in T.T.tolist()]
    p0 = p0.tolist()
    T = T.double().numpy()
    m = 3 + len(cols)
    if int(indexed.sum()) * 3 <= m:
        return None
    A64 = A_d.double().numpy()
    q64 = q_d.double().numpy()
    try:
        U0 = A64 @ np.linalg.inv(np.array(_cell_basis(p0, cols)[0]).reshape(3, 3))
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return None
    prior_width = (
        np.array(
            [edge_tolerance * c0.a, edge_tolerance * c0.b, edge_tolerance * c0.c]
            + [angle_tolerance_rad] * 3
        )
        / 3.0
    )
    khat = np.zeros_like(q64)
    has_k = wavelength is not None and wavelength > 0
    if has_k:
        k_out = q64 + np.array([0.0, 0.0, 1.0 / wavelength])
        norm = np.maximum(np.linalg.norm(k_out, axis=-1, keepdims=True), 1e-12)
        khat = k_out / norm

    def split(Am, hkl_, sel):
        r = hkl_.double().numpy()[sel] @ Am.T - q64[sel]
        par = (r * khat[sel]).sum(-1)
        return r - par[:, None] * khat[sel], par

    perp, par = split(A64, hkl_d, indexed.numpy())
    s_par = max(math.sqrt(np.square(par).mean()), 1e-9)
    s_perp = max(
        math.sqrt(np.square(perp).sum(-1).mean() / (2.0 if has_k else 3.0)), 1e-9
    )
    if has_k:
        s_perp = max(s_perp, s_par / 100.0)

    def mean_square(Am, hkl_, sel):
        perp, par = split(Am, hkl_, sel.numpy())
        total = np.square(perp).sum() / s_perp**2 + np.square(par).sum() / s_par**2
        return float(total) / (4 * len(par))

    before = mean_square(A64, hkl_d, indexed)
    theta = np.zeros(m, dtype=np.float32)
    A_new = A_d
    for _ in range(rounds):
        if int(indexed.sum()) * 3 <= m:
            return None
        sel = indexed.numpy()
        evaluate = _evaluator(
            U0,
            p0,
            cols,
            T,
            prior_width,
            hkl_d.double().numpy()[sel],
            q64[sel],
            khat[sel],
            s_perp,
            s_par,
        )
        lam = 1e-3
        try:
            A_cur, ext = evaluate(theta.tolist())
        except (ArithmeticError, ValueError):
            return None
        Z = _gram(ext)
        loss = float(Z[0, 0])
        for _ in range(iters):
            Hm = Z[1:, 1:].copy()
            diag = np.einsum("ii->i", Hm)
            diag += lam * diag + np.float32(1e-18)
            try:
                step = np.linalg.solve(Hm, -Z[1:, 0])
            except Exception:
                return None
            cand = theta + step
            x = cand.tolist()
            if not all(map(math.isfinite, x)):
                lam *= 10.0
                continue
            try:
                A_c, ext_c = evaluate(x)
            except (ArithmeticError, ValueError):
                lam *= 10.0
                continue
            Zc = _gram(ext_c)
            lc = float(Zc[0, 0])
            if lc < loss:
                theta, A_cur, Z, loss, lam = cand, A_c, Zc, lc, lam / 3.0
            else:
                lam *= 10.0
        A32 = A_cur.astype(np.float32)
        if not np.isfinite(A32).all():
            return None
        A_new = torch.from_numpy(A32)
        hkl_d, indexed = _assign_hkls(A_new, q_d, q_tolerance)
    n = int(indexed.sum())
    if n == 0:
        return None
    improved = mean_square(A_new.double().numpy(), hkl_d, indexed) <= before
    sq = (hkl_d @ A_new.transpose(-1, -2) - q_d).square().sum(-1)
    rmsd = float(torch.sqrt(sq[indexed].mean()))
    return (
        A_new.to(dtype=dtype, device=out_device),
        hkl_d.long().to(out_device),
        indexed.to(out_device),
        rmsd,
        improved,
    )
