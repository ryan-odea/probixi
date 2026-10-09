import torch
import triton
import triton.language as tl

_FINISH = {
    "int8": lambda x: x.to(torch.int8),
    "uint8": lambda x: x.to(torch.uint8),
    "int16": lambda x: x.to(torch.int16),
    "uint16": lambda x: x & 0xFFFF,
    "int32": lambda x: x,
    "uint32": lambda x: x.to(torch.int64) & 0xFFFFFFFF,
}


@triton.jit
def width(D, c, mask):
    # Byte width of the escape token starting at the 0x80 byte at c
    b1 = tl.load(D + c + 1, mask, 1)
    b2 = tl.load(D + c + 2, mask, 0)
    b3 = tl.load(D + c + 3, mask, 0)
    b4 = tl.load(D + c + 4, mask, 0)
    b5 = tl.load(D + c + 5, mask, 0)
    b6 = tl.load(D + c + 6, mask, 0)
    wide = (b1 == 0) & (b2 == -128)
    huge = wide & (b3 == 0) & (b4 == 0) & (b5 == 0) & (b6 == -128)
    return 3 + 4 * wide.to(tl.int32) + 8 * huge.to(tl.int32)


@triton.jit
def walk(D, C, W, ERR, LENS, n, STRIDE, BLOCK: tl.constexpr):
    """Mark the escape tokens among the 0x80 candidates C in W (their width).

    A candidate 15 or more bytes behind its predecessor, or opening a frame, is
    surely a token. The others are resolved sequentially from there, one chain
    per lane: a candidate is a token when no earlier token reaches it.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inb = i < n
    c = tl.load(C + i, inb, 0)
    last = tl.load(C + i - 1, inb & (i > 0), 0)
    head = inb & ((i == 0) | (c - last >= 15) | (c // STRIDE != last // STRIDE))
    w = width(D, c, head)
    tl.store(W + c, w.to(tl.int8), head)
    frame = c // STRIDE
    limit = frame * STRIDE + tl.load(LENS + frame, head, 0)
    tl.store(ERR + frame, 1, head & (c + w > limit))
    end = c + w
    last = c
    cur = i
    active = head
    while tl.max(active.to(tl.int32), 0) > 0:
        j = cur + 1
        valid = active & (j < n)
        cj = tl.load(C + j, valid, 0)
        cont = valid & (cj - last < 15) & (cj // STRIDE == last // STRIDE)
        token = cont & (cj >= end)
        wj = width(D, cj, token)
        tl.store(W + cj, wj.to(tl.int8), token)
        frame = cj // STRIDE
        limit = frame * STRIDE + tl.load(LENS + frame, token, 0)
        tl.store(ERR + frame, 1, token & (cj + wj > limit))
        end = tl.where(token, cj + wj, end)
        last = tl.where(cont, cj, last)
        cur = tl.where(cont, j, cur)
        active = cont


@triton.jit
def keep(W, LENS, KEEP, STRIDE, total, BLOCK: tl.constexpr):
    """KEEP: byte inside its frame's data and not inside an escape token."""
    b = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inb = b < total
    frame = b // STRIDE
    inside = (b - frame * STRIDE) < tl.load(LENS + frame, inb, 0)
    hidden = b < 0
    for k in tl.static_range(1, 15):
        hidden |= tl.load(W + b - k, inb & (b >= k), 0) > k
    tl.store(KEEP + b, (inside & ~hidden).to(tl.int8), inb)


@triton.jit
def gather(D, W, KEEP, CS, FIRST, OUT, STRIDE, N, total, BLOCK: tl.constexpr):
    """OUT: the difference carried by each pixel's token, at its pixel index."""
    b = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inb = b < total
    kept = tl.load(KEEP + b, inb, 0) != 0
    frame = b // STRIDE
    pixel = tl.load(CS + b, kept, 1) - tl.load(FIRST + frame, kept, 0) - 1
    w = tl.load(W + b, kept, 0).to(tl.int32)
    plain = tl.load(D + b, kept, 0).to(tl.int32)
    start = tl.where(w == 3, 1, tl.where(w == 7, 3, 7))
    escape = kept & (w > 0)
    x0 = tl.load(D + b + start, escape, 0).to(tl.int32)
    x1 = tl.load(D + b + start + 1, escape, 0).to(tl.int32)
    x2 = tl.load(D + b + start + 2, escape & (w > 3), 0).to(tl.int32)
    x3 = tl.load(D + b + start + 3, escape & (w > 3), 0).to(tl.int32)
    short = (x0 & 255) | (x1 << 8)
    long = (x0 & 255) | ((x1 & 255) << 8) | ((x2 & 255) << 16) | (x3 << 24)
    value = tl.where(w == 0, plain, tl.where(w == 3, short, long))
    tl.store(OUT + frame * N + pixel, value, kept & (pixel < N))


def decode(stage, lens, count, dtype, names):
    """Decode byte-offset CBF data on the device.

    Parameters
    ----------
    stage : torch.Tensor
        ``(frames, stride)`` uint8 on the device; frame ``i`` holds its data in
        the first ``lens[i]`` bytes and zeros after, at least 16 of them.
    lens : torch.Tensor
        Data bytes of each frame, int32 on the device.
    count : int
        Pixels per frame.
    dtype : numpy.dtype
        Pixel type; the sum wraps at its width.
    names : sequence of str
        Frame names for error messages.

    Returns
    -------
    torch.Tensor
        ``(frames, count)`` with the pixel values (unsigned types widened).

    Raises
    ------
    ValueError
        If a frame ends inside a value or does not hold ``count`` pixels.
    """
    frames, stride = stage.shape
    total = frames * stride
    d = stage.view(-1).view(torch.int8)
    candidates = (d == -128).nonzero().squeeze(1)
    n = candidates.numel()
    width_ = torch.zeros(total, dtype=torch.int8, device=stage.device)
    errors = torch.zeros(frames, dtype=torch.int32, device=stage.device)
    if n:
        walk[(triton.cdiv(n, 128),)](
            d, candidates, width_, errors, lens, n, stride, BLOCK=128
        )
    kept = torch.empty(total, dtype=torch.int8, device=stage.device)
    keep[(triton.cdiv(total, 1024),)](width_, lens, kept, stride, total, BLOCK=1024)
    # flat scans are far faster than scans along the rows; wrapping sums stay exact
    position = kept.cumsum(0, dtype=torch.int32)
    ends = position[stride - 1 :: stride]
    first = torch.cat((ends.new_zeros(1), ends[:-1]))
    out = torch.empty((frames, count), dtype=torch.int32, device=stage.device)
    gather[(triton.cdiv(total, 1024),)](
        d, width_, kept, position, first, out, stride, count, total, BLOCK=1024
    )
    total_sum = out.view(-1).cumsum(0, dtype=torch.int32).view(frames, count)
    out = total_sum - torch.cat((total_sum.new_zeros(1), total_sum[:-1, -1])).unsqueeze(
        1
    )
    got, bad = torch.stack((ends - first, errors)).tolist()
    for name, pixels, error in zip(names, got, bad):
        if error:
            raise ValueError(f"{name}: CBF data end inside a value")
        if pixels != count:
            raise ValueError(
                f"{name}: CBF data decode to {pixels} pixels, expected {count}"
            )
    return _FINISH[dtype.name](out)
