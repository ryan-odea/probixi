import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def count_below(F, N, STRIDE, floor, OUT, INCLUSIVE: tl.constexpr, B: tl.constexpr):
    k = tl.program_id(0) * B + tl.arange(0, B)
    live = k < (N + STRIDE - 1) // STRIDE
    v = tl.load(F + k * STRIDE, live, 0.0)
    hit = v < floor
    if INCLUSIVE:
        hit = hit | (v == floor)
    tl.atomic_add(OUT, tl.sum((hit & live).to(tl.int32), 0))
    tl.atomic_add(OUT + 1, tl.sum(((v != v) & live).to(tl.int32), 0))


def below(frame, floor, stride):
    # median of the strided sample < floor, from the count of values below it
    n = (frame.numel() + stride - 1) // stride
    out = torch.zeros(2, dtype=torch.int32, device=frame.device)
    floor32 = float(np.float32(floor))
    count_below[(triton.cdiv(n, 1024),)](
        frame, frame.numel(), stride, floor32, out, INCLUSIVE=floor32 < floor, B=1024
    )
    count, nans = out.tolist()
    return nans == 0 and count >= (n + 1) // 2
