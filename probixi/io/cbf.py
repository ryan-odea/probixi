from __future__ import annotations

import base64
import bz2
import hashlib
import os
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

PathLike = Union[str, Path]

_SECTION = b"--CIF-BINARY-FORMAT-SECTION--"
_START = b"\x0c\x1a\x04\xd5"  # CBF marker before the first binary data byte
_CHUNK = 1 << 14
_MAX_HEADER = 1 << 22

_FIELD = re.compile(rb"^[ \t]*([A-Za-z][\w-]*)[ \t]*:([^\r\n]*)", re.M)
_CONVERSION = re.compile(rb'conversions\s*=\s*"?([\w-]+)', re.I)
_NEXT_SECTION = re.compile(rb"--CIF-BINARY-FORMAT-SECTION--(?!--)")
_COMPRESSION = {"x-cbf_byte_offset": "byte_offset", "x-cbf_none": "none"}
_DTYPES = {
    "unsigned 8-bit integer": np.dtype(np.uint8),
    "signed 8-bit integer": np.dtype(np.int8),
    "unsigned 16-bit integer": np.dtype(np.uint16),
    "signed 16-bit integer": np.dtype(np.int16),
    "unsigned 32-bit integer": np.dtype(np.uint32),
    "signed 32-bit integer": np.dtype(np.int32),
}


class UnsupportedCbf(ValueError):
    # A valid CBF variant this reader does not decode
    ...


@dataclass(frozen=True)
class CbfHeader:
    shape: tuple[int, int]  # (ss, fs): second dimension, then fastest
    dtype: np.dtype
    compression: str  # "byte_offset" or "none"
    offset: int  # first data byte in the decompressed file
    size: int  # X-Binary-Size, bytes
    md5: Optional[str] = None  # base64 Content-MD5 of the data bytes


def is_cbf(path: PathLike) -> bool:
    return str(path).lower().endswith((".cbf", ".cbf.gz", ".cbf.bz2"))


def read_cbf_header(path: PathLike) -> CbfHeader:
    # Reads, and inflates, only up to the binary start marker
    path = Path(path)
    inflate = _decompressor(path)
    head = b""
    with path.open("rb") as fh:
        while _START not in head and len(head) < _MAX_HEADER:
            block = fh.read(_CHUNK)
            if not block:
                break
            head += _inflate(inflate, block, path) if inflate else block
        header = _parse(head, path)
        if (
            inflate is None
            and header.offset + header.size > os.fstat(fh.fileno()).st_size
        ):
            raise ValueError(f"{path}: truncated CBF file")
    return header


def read_cbf(path: PathLike, verify: bool = True) -> np.ndarray:
    """Decode the image of a CBF file.

    Parameters
    ----------
    path : str or Path
        A ``.cbf``, ``.cbf.gz`` or ``.cbf.bz2`` file with one binary section.
    verify : bool, default True
        Check ``Content-MD5`` when the file carries it.

    Returns
    -------
    numpy.ndarray
        Pixels with shape ``(ss, fs)`` and the file's integer type.

    Raises
    ------
    UnsupportedCbf
        For compression other than ``x-CBF_BYTE_OFFSET`` or ``x-CBF_NONE``,
        big-endian data, non-integer or 64-bit pixels, 3-D arrays, or several
        binary sections.
    ValueError
        If the file is not a CBF image, is truncated, or fails its checksum.
    """
    header, data = read_cbf_data(path, verify)
    return decode_cbf(header, data, path)


def read_cbf_data(path: PathLike, verify: bool = True) -> tuple[CbfHeader, memoryview]:
    # Checked, decompressed data bytes, still encoded; raises as read_cbf does
    path = Path(path)
    raw = path.read_bytes()
    inflate = _decompressor(path)
    if inflate is not None:
        raw = _inflate(inflate, raw, path)
        if not inflate.eof:
            raise ValueError(f"{path}: truncated compressed CBF file")
    header = _parse(raw, path)
    stop = header.offset + header.size
    if stop > len(raw):
        raise ValueError(f"{path}: truncated CBF file")
    if _NEXT_SECTION.search(raw, stop):
        raise UnsupportedCbf(f"{path}: files with several binary sections are not read")
    data = memoryview(raw)[header.offset : stop]
    if verify and header.md5 is not None:
        digest = base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest())
        if digest.decode() != header.md5:
            raise ValueError(f"{path}: Content-MD5 mismatch, the file is corrupt")
    return header, data


def decode_cbf(header: CbfHeader, data, path: PathLike) -> np.ndarray:
    # CPU decode; the GPU path in frames.py sends byte-offset data to kernels/cbf
    if header.compression == "none":
        pixels = np.frombuffer(data, header.dtype.newbyteorder("<")).astype(
            header.dtype
        )
    else:
        try:
            pixels = decode_byte_offset(
                data, header.shape[0] * header.shape[1], header.dtype
            )
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from None
    return pixels.reshape(header.shape)


def decode_byte_offset(data, count: int, dtype=np.int32) -> np.ndarray:
    # Little-endian deltas: int8 | 80 int16 | 80 0080 int32 | 80 0080 00000080 int64
    d = np.frombuffer(data, dtype=np.int8)
    candidates = np.flatnonzero(d == -128)
    if candidates.size:
        pos, width = _escapes(d, candidates)
        if pos[-1] + width[-1] > d.size:
            raise ValueError("CBF data end inside a value")
        value = _payload(d, pos, width)
        hidden = np.cumsum(width - 1)
        before = hidden - (width - 1)
        covered = np.repeat(pos + 1 - before, width - 1) + np.arange(hidden[-1])
        keep = np.ones(d.size, dtype=bool)
        keep[covered] = False
        x = d[keep].astype(np.int32)
        x[pos - before] = value
    else:
        x = d.astype(np.int32)
    if x.size != count:
        raise ValueError(f"CBF data decode to {x.size} pixels, expected {count}")
    torch.from_numpy(x).cumsum_(0)  # several times faster than np.cumsum
    dtype = np.dtype(dtype)
    return x.view(dtype) if dtype.itemsize == 4 else x.astype(dtype)


def _escapes(d: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Start and byte width of each escape; a 0x80 inside an earlier payload is none
    ahead = d[np.minimum(candidates[:, None] + np.arange(1, 7), d.size - 1)]
    wide = (ahead[:, 0] == 0) & (ahead[:, 1] == -128)
    huge = wide & ~ahead[:, 2:5].any(axis=1) & (ahead[:, 5] == -128)
    width = 3 + 4 * wide + 8 * huge
    end = candidates + width
    if (candidates[1:] < end[:-1]).any():
        n = candidates.size
        # Tokens chain to the first candidate past their end; pointer doubling walks it
        jump = np.append(np.searchsorted(candidates, end), n)
        token = np.zeros(n + 1, dtype=bool)
        token[0] = True
        for _ in range(n.bit_length()):
            token[jump[token]] = True
            jump = jump[jump]
        candidates, width = candidates[token[:n]], width[token[:n]]
    return candidates, width


def _payload(d: np.ndarray, pos: np.ndarray, width: np.ndarray) -> np.ndarray:
    # Value carried by each escape token
    raw = d.view(np.uint8)[np.minimum(pos[:, None] + np.arange(1, 15), d.size - 1)]

    def le(lo, hi, kind):
        return np.ascontiguousarray(raw[:, lo:hi]).view(kind)[:, 0]

    return np.where(
        width == 3,
        le(0, 2, "<i2"),
        np.where(width == 7, le(2, 6, "<i4"), le(6, 14, "<i8")),
    )


def _decompressor(path: Path):
    name = path.name.lower()
    if name.endswith(".gz"):
        return zlib.decompressobj(wbits=31)  # gzip container
    if name.endswith(".bz2"):
        return bz2.BZ2Decompressor()
    return None


def _inflate(inflate, block: bytes, path: Path) -> bytes:
    try:
        return inflate.decompress(block)
    except (zlib.error, EOFError, OSError) as exc:
        raise ValueError(f"{path}: cannot decompress CBF file ({exc})") from None


def _parse(head: bytes, path: Path) -> CbfHeader:
    start = head.find(_START)
    section = head.rfind(_SECTION, 0, start) if start >= 0 else -1
    if section < 0:
        raise ValueError(f"{path}: no CBF binary section found")
    mime = head[section:start]
    field = {
        k.lower().decode(): v.strip(b" \t\"'").decode("ascii", "replace")
        for k, v in _FIELD.findall(mime)
    }
    found = _CONVERSION.search(mime)
    conversion = found.group(1).decode() if found else ""
    if conversion.lower() not in _COMPRESSION:
        raise UnsupportedCbf(
            f"{path}: unsupported CBF compression {conversion!r}, "
            "only x-CBF_BYTE_OFFSET and x-CBF_NONE are read"
        )
    kind = field.get("x-binary-element-type", "").lower()
    if kind not in _DTYPES:
        raise UnsupportedCbf(f"{path}: unsupported CBF pixel type {kind!r}")
    if (
        field.get("x-binary-element-byte-order", "LITTLE_ENDIAN").upper()
        != "LITTLE_ENDIAN"
    ):
        raise UnsupportedCbf(f"{path}: big-endian CBF data are not read")
    try:
        size = int(field["x-binary-size"])
        count = int(field["x-binary-number-of-elements"])
        fast = int(field["x-binary-size-fastest-dimension"])
        slow = int(field["x-binary-size-second-dimension"])
        planes = int(field.get("x-binary-size-third-dimension", 1))
    except (KeyError, ValueError):
        raise ValueError(f"{path}: incomplete CBF binary-section header") from None
    if planes > 1:
        raise UnsupportedCbf(f"{path}: 3-D CBF arrays are not read")
    dtype = _DTYPES[kind]
    compression = _COMPRESSION[conversion.lower()]
    if not 0 < fast * slow == count or (
        compression == "none" and size != count * dtype.itemsize
    ):
        raise ValueError(f"{path}: inconsistent CBF binary-section header")
    return CbfHeader(
        shape=(slow, fast),
        dtype=dtype,
        compression=compression,
        offset=start + len(_START),
        size=size,
        md5=field.get("content-md5"),
    )
