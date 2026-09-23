import torch
import triton
import triton.language as tl


@triton.jit
def correlate(
    OFF,
    KA,
    VA,
    GA,
    KB,
    VB,
    J,
    CC,
    NOVL,
    C: tl.constexpr,
    GP: tl.constexpr,
    INV_G: tl.constexpr,
    STEPS: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid // C
    j = tl.load(J + pid)
    a0 = tl.load(OFF + i)
    na = tl.load(OFF + i + 1) - a0
    b0 = tl.load(OFF + j)
    nb = tl.load(OFF + j + 1) - b0

    g_id = tl.arange(0, GP)
    sxy = tl.zeros((GP,), tl.float32)
    sx = tl.zeros((GP,), tl.float32)
    sy = tl.zeros((GP,), tl.float32)
    sx2 = tl.zeros((GP,), tl.float32)
    sy2 = tl.zeros((GP,), tl.float32)
    cnt = tl.zeros((GP,), tl.float32)

    for s in range(0, na, B):
        o = s + tl.arange(0, B)
        live = o < na
        key = tl.load(KA + a0 + o, live, 0x7FFFFFFF)
        av = tl.load(VA + a0 + o, live, 0.0)
        grp = tl.load(GA + a0 + o, live, -1)

        # lower_bound(key) in KB[b0 : b0+nb]; STEPS >= log2(max run length)
        lo = tl.zeros((B,), tl.int32)
        hi = nb + tl.zeros((B,), tl.int32)
        for _ in tl.static_range(STEPS):
            mid = (lo + hi) // 2
            probe = tl.load(KB + b0 + mid, mid < nb, 0x7FFFFFFF)
            less = probe < key
            lo = tl.where(less, mid + 1, lo)
            hi = tl.where(less, hi, mid)

        found = live & (lo < nb)
        found = found & (tl.load(KB + b0 + lo, found, 0x7FFFFFFF) == key)
        bv = tl.load(VB + b0 + lo, found, 0.0)
        av = tl.where(found, av, 0.0)

        sel = tl.where(found[None, :] & (grp[None, :] == g_id[:, None]), 1.0, 0.0)
        a = sel * av[None, :]
        b = sel * bv[None, :]
        sxy += tl.sum(a * bv[None, :], 1)
        sx += tl.sum(a, 1)
        sy += tl.sum(b, 1)
        sx2 += tl.sum(a * av[None, :], 1)
        sy2 += tl.sum(b * bv[None, :], 1)
        cnt += tl.sum(sel, 1)

    n = tl.maximum(cnt, 1.0)
    t1 = sx2 - sx * sx / n
    t2 = sy2 - sy * sy / n
    good = (cnt >= 2.0) & (t1 > 0.0) & (t2 > 0.0)
    cc = tl.where(good, (sxy - sx * sy / n) / tl.sqrt(t1 * t2), 0.0)
    tl.store(CC + pid, tl.sum(cc) * INV_G)
    tl.store(NOVL + pid, tl.sum(cnt).to(tl.int32))


def pair_cc(offsets, keys_a, vals_a, groups_a, keys_b, vals_b, partners, n_groups):
    n, c = partners.shape
    cc = torch.empty(n * c, dtype=torch.float32, device=keys_a.device)
    novl = torch.empty(n * c, dtype=torch.int32, device=keys_a.device)
    longest = int((offsets[1:] - offsets[:-1]).max())
    correlate[(n * c,)](
        offsets,
        keys_a,
        vals_a,
        groups_a,
        keys_b,
        vals_b,
        partners.reshape(-1),
        cc,
        novl,
        C=c,
        GP=max(2, triton.next_power_of_2(n_groups)),
        INV_G=1.0 / n_groups,
        STEPS=max(1, longest.bit_length()),
        B=128,
        num_warps=4,
    )
    return cc.view(n, c), novl.view(n, c)
