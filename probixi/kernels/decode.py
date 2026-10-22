import ctypes as ct
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution

import torch
import triton
import triton.language as tl


class Options(ct.Structure):
    # nvCOMP option structs are 64 bytes; only the backend field is set
    _fields_ = [("backend", ct.c_int), ("reserved", ct.c_char * 60)]


class Alignments(ct.Structure):
    _fields_ = [("input", ct.c_size_t), ("output", ct.c_size_t), ("temp", ct.c_size_t)]


@lru_cache(None)
def library():
    try:
        dist = distribution("nvidia-libnvcomp-cu12")
        path = dist.locate_file("nvidia/libnvcomp/lib64/libnvcomp.so.5")
        return ct.CDLL(str(path))
    except (PackageNotFoundError, OSError):
        return None


def check(status):
    if status:
        raise RuntimeError(f"nvCOMP failed with status {status}")


@triton.jit
def unshuffle(
    SRC,
    DST,
    META,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    BYTE_SHUFFLE: tl.constexpr,
):
    # META (frames, 2): bitshuffle block bytes (0 = none) and byte-shuffle flag
    frame = tl.program_id(1)
    j = tl.program_id(0) * B + tl.arange(0, B)
    block_bytes = (
        BLOCK_BYTES
        if BLOCK_BYTES is not None
        else tl.load(META + frame * 2).to(tl.int32)
    )
    byte_shuffle = (
        BYTE_SHUFFLE
        if BYTE_SHUFFLE is not None
        else tl.load(META + frame * 2 + 1).to(tl.int32)
    )
    # assemble each element as unsigned bits; bitcast once all its bytes are in
    value = tl.full((B,), 0, tl.uint64 if E == 8 else tl.uint32)
    if block_bytes > 0:
        ne = block_bytes // E
        block = j // ne
        within = j % ne
        count = tl.minimum(ne, N - block * ne) // 8 * 8
        base = frame * STRIDE + block * block_bytes
        for plane in tl.static_range(E * 8):
            v = tl.load(
                SRC + base + plane * (count // 8) + within // 8,
                (j < N) & (within < count),
                0,
            ).to(value.dtype)
            value |= (((v >> (within % 8).to(value.dtype)) & 1) << plane).to(
                value.dtype
            )
        # the final < 8 elements are stored verbatim, outside any bitshuffle block
        for byte in tl.static_range(E):
            v = tl.load(
                SRC + frame * STRIDE + j * E + byte, (j < N) & (within >= count), 0
            ).to(value.dtype)
            value |= v << (byte * 8)
    else:
        for byte in tl.static_range(E):
            offset = tl.where(byte_shuffle != 0, byte * N + j, j * E + byte)
            v = tl.load(SRC + frame * STRIDE + offset, j < N, 0).to(value.dtype)
            value |= v << (byte * 8)
    # DST has the storage dtype; the caller casts to the requested dtype
    typ = DST.dtype.element_ty
    unsigned = (
        tl.uint8
        if E == 1
        else tl.uint16 if E == 2 else tl.uint32 if E == 4 else tl.uint64
    )
    tl.store(DST + frame * N + j, value.to(unsigned).to(typ, bitcast=True), j < N)


@triton.jit
def adler_partials(SRC, OUT, SIZE: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr):
    block, frame = tl.program_id(0), tl.program_id(1)
    j = block * B + tl.arange(0, B)
    x = tl.load(SRC + frame * STRIDE + j, j < SIZE, 0).to(tl.int64)
    tl.store(OUT + (frame * tl.cdiv(SIZE, B) + block) * 2, tl.sum(x, 0))
    tl.store(
        OUT + (frame * tl.cdiv(SIZE, B) + block) * 2 + 1, tl.sum(x * (SIZE - j), 0)
    )


class Decoder:
    def __init__(self, codec, frame_bytes, device):
        self.lib = library()
        if self.lib is None:
            raise RuntimeError("nvCOMP 5.x CUDA library is unavailable")
        self.device, self.frame_bytes = device, frame_bytes
        self.stride = (frame_bytes + 255) // 256 * 256
        self.options = Options(2)  # CUDA SM backend, available on pre-Blackwell GPUs
        self.align = Alignments(1, 1, 1)
        self.codec = codec
        self.scratch = None
        self.temp = None
        if codec is not None:
            prefix = "nvcompBatched" + codec + "Decompress"
            align = getattr(self.lib, prefix + "GetRequiredAlignments")
            align.argtypes = [Options, ct.POINTER(Alignments)]
            check(align(self.options, ct.byref(self.align)))
            self.get_temp = getattr(self.lib, prefix + "GetTempSizeAsync")
            self.get_temp.argtypes = [
                ct.c_size_t,
                ct.c_size_t,
                Options,
                ct.POINTER(ct.c_size_t),
                ct.c_size_t,
            ]
            self.launch = getattr(self.lib, prefix + "Async")
            self.launch.argtypes = [ct.c_void_p] * 4 + [
                ct.c_size_t,
                ct.c_void_p,
                ct.c_size_t,
                ct.c_void_p,
                Options,
                ct.c_void_p,
                ct.c_void_p,
            ]

    def decode(self, packed, metadata, copies, shuffle, checksums, dtype, shape):
        # metadata rows (in_offset, in_bytes, out_offset, out_bytes) on the CPU
        batch = len(shuffle)
        compressed = packed.to(self.device, non_blocking=True)
        size = batch * self.stride
        if self.scratch is None or self.scratch.numel() < size:
            self.scratch = torch.empty(size, device=self.device, dtype=torch.uint8)
        if len(metadata):
            max_bytes = int(metadata[:, 3].max())
            temp = ct.c_size_t()
            check(
                self.get_temp(
                    len(metadata), max_bytes, self.options, ct.byref(temp), size
                )
            )
            if self.temp is None or self.temp.numel() < temp.value:
                self.temp = torch.empty(
                    max(1, temp.value), device=self.device, dtype=torch.uint8
                )
            m = metadata.to(self.device, non_blocking=True)
            inputs = (m[:, 0] + compressed.data_ptr()).contiguous()
            lengths = m[:, 1].contiguous()
            outputs = (m[:, 2] + self.scratch.data_ptr()).contiguous()
            capacities = m[:, 3].contiguous()
            actual = torch.empty_like(capacities)
            statuses = torch.empty(len(m), device=self.device, dtype=torch.int32)
            check(
                self.launch(
                    inputs.data_ptr(),
                    lengths.data_ptr(),
                    capacities.data_ptr(),
                    actual.data_ptr(),
                    len(m),
                    self.temp.data_ptr(),
                    temp.value,
                    outputs.data_ptr(),
                    self.options,
                    statuses.data_ptr(),
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
            )
            if not bool(((statuses == 0) & (actual == capacities)).all()):
                raise RuntimeError("nvCOMP decoded length or status mismatch")
        for source, target, length in copies:  # chunks stored without compression
            self.scratch[target : target + length].copy_(
                compressed[source : source + length]
            )
        if any(c is not None for c in checksums):
            pieces = torch.empty(
                (batch, triton.cdiv(self.frame_bytes, 4096), 2),
                dtype=torch.int64,
                device=self.device,
            )
            adler_partials[(pieces.shape[1], batch)](
                self.scratch, pieces, self.frame_bytes, self.stride, B=4096
            )
            sums = pieces.sum(1)
            # Adler-32: A = 1 + sum d_j, B = n + sum (n - j) d_j, both mod 65521
            got = (((sums[:, 1] + self.frame_bytes) % 65521) << 16) | (
                (sums[:, 0] + 1) % 65521
            )
            for expected, value in zip(checksums, got.tolist()):
                if expected is not None and expected != value:
                    raise ValueError("HDF5 zlib Adler32 checksum mismatch")
        original = torch.empty((batch, *shape), dtype=dtype, device=self.device)
        meta = torch.tensor(shuffle, dtype=torch.int64, device=self.device)
        n = self.frame_bytes // original.element_size()
        unshuffle[(triton.cdiv(n, 256), batch)](
            self.scratch,
            original,
            meta,
            n,
            self.stride,
            original.element_size(),
            B=256,
            # uniform framing compiles as constants; a mixed batch reads META per frame
            BLOCK_BYTES=(
                shuffle[0][0] if all(s == shuffle[0] for s in shuffle) else None
            ),
            BYTE_SHUFFLE=(
                shuffle[0][1] if all(s == shuffle[0] for s in shuffle) else None
            ),
        )
        return original
