from __future__ import annotations

import os
import queue
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import h5py
import hdf5plugin  # noqa: F401  (registers bitshuffle)
import numpy as np
import torch
from torch import Tensor

from .assemble import assemble_batch
from .cell import CellParams, read_crystfel_cell
from .geometry import Geometry, read_geometry, resolve_dynamic_fields
from .metadata import H5Info, Metadata, scan_h5

try:
    import bitshuffle

    _HAS_BSHUF = True
except Exception:  # pragma: no cover
    _HAS_BSHUF = False

PathLike = Union[str, Path]

_BSHUF_FILTER_ID = 32008
_BSHUF_LZ4 = 2


class DataLoader:
    """Resolve HDF5 files, geometry, and cell up front for frame loading.

    Parameters
    ----------
    list_file : str or Path
        Text file listing one HDF5 path per line (``#``/``;`` comments allowed).
    geometry_file : str or Path, optional
        CrystFEL ``.geom`` file to parse.
    cell_file : str or Path, optional
        CrystFEL ``.cell`` file to parse.

    Attributes
    ----------
    metadata : Metadata
        Resolved files, geometry, cell, and frame counts.
    """

    def __init__(
        self,
        list_file: PathLike,
        geometry_file: Optional[PathLike] = None,
        cell_file: Optional[PathLike] = None,
    ):
        self.list_file = Path(list_file)
        self.geometry_file = Path(geometry_file) if geometry_file else None
        self.cell_file = Path(cell_file) if cell_file else None
        self.metadata = self._build_metadata()

    @property
    def files(self) -> dict:
        return self.metadata.files

    def __len__(self) -> int:
        return self.metadata.n_frames

    def _read_list(self) -> list[str]:
        if not self.list_file.is_file():
            raise FileNotFoundError(f"List file not found: {self.list_file}")
        paths: list[str] = []
        with self.list_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(("#", ";")):
                    continue
                paths.append(line)
        return paths

    def _scan_files(
        self, geometry: Optional[Geometry] = None
    ) -> tuple[dict, Optional[tuple], int]:
        files: dict[str, H5Info] = {}
        frame_size: Optional[tuple] = None
        total_frames = 0
        for path in self._read_list():
            info = scan_h5(path, geometry)
            if frame_size is None:
                frame_size = info.frame_shape
            elif frame_size != info.frame_shape:
                raise ValueError(
                    f"Inconsistent frame size in {path}: "
                    f"{info.frame_shape} != {frame_size}"
                )
            total_frames += info.n_frames
            files[info.filename] = info
        return files, frame_size, total_frames

    def _parse_geometry(self) -> Optional[Geometry]:
        if self.geometry_file is None:
            return None
        return read_geometry(self.geometry_file)

    def _parse_cell(self) -> Optional[CellParams]:
        if self.cell_file is None:
            return None
        return read_crystfel_cell(self.cell_file)

    def _build_metadata(self) -> Metadata:
        geometry = self._parse_geometry()
        files, frame_size, total_frames = self._scan_files(geometry)
        if geometry is not None and files:
            data_file = next(iter(files.values())).filename
            resolve_dynamic_fields(geometry, data_file)
        return Metadata(
            files=files,
            geometry=geometry,
            cell=self._parse_cell(),
            frame_size=frame_size,
            n_files=len(files),
            n_frames=total_frames,
        )


# Prefetch - hides io behind (hopefully) useful work
_PREFETCH_SENTINEL = object()


def _bshuf_lz4_decoder(dset: h5py.Dataset, frame_shape):
    if not _HAS_BSHUF:
        return None
    if dset.chunks != (1,) + tuple(int(x) for x in frame_shape):
        return None
    plist = dset.id.get_create_plist()
    if plist.get_nfilters() != 1:
        return None
    fid, _flags, cd, _name = plist.get_filter(0)
    if fid != _BSHUF_FILTER_ID or len(cd) < 5 or cd[4] != _BSHUF_LZ4:
        return None
    dt = dset.dtype
    if dt.byteorder not in ("=", "<", "|"):
        return None
    itemsize = dt.itemsize
    shape = tuple(int(x) for x in frame_shape)

    def decode(raw) -> np.ndarray:
        blk = struct.unpack(">i", raw[8:12])[0]
        comp = np.frombuffer(raw[12:], dtype=np.uint8)
        return bitshuffle.decompress_lz4(comp, shape, dt, blk // itemsize if blk else 0)

    return decode


def _iter_file_frames(info, f_lo, f_hi, pool, window, stop, fast_state):
    with h5py.File(info.filename, "r") as f:
        decoder = None
        dset = None
        if info.placements is None:
            node = f[info.dataset]
            if not isinstance(node, h5py.Dataset):
                raise TypeError(
                    f"{info.dataset!r} in {info.filename} is not an HDF5 dataset"
                )
            dset = node
            decoder = _bshuf_lz4_decoder(dset, info.frame_shape)
            if decoder is not None:
                if not fast_state["checked"]:
                    raw0 = dset.id.read_direct_chunk((f_lo, 0, 0))[1]
                    fast_state["ok"] = bool(
                        np.array_equal(decoder(raw0), np.asarray(dset[f_lo]))
                    )
                    fast_state["checked"] = True
                if not fast_state["ok"]:
                    decoder = None
        i = f_lo
        while i < f_hi and not stop.is_set():
            end = min(i + window, f_hi)
            if decoder is not None:
                raws = [dset.id.read_direct_chunk((j, 0, 0))[1] for j in range(i, end)]
                for arr in pool.map(decoder, raws):
                    yield arr
            elif dset is not None:
                block = np.asarray(dset[i:end])
                for k in range(block.shape[0]):
                    yield block[k]
            else:
                block = assemble_batch(f, i, end, info.placements, info.frame_shape)
                for k in range(block.shape[0]):
                    yield block[k]
            i = end


def _prefetch_worker(
    entries: dict,
    chosen: list,
    lo: int,
    hi: int,
    q: queue.Queue,
    batch_size: int,
    stop: threading.Event,
    pool: ThreadPoolExecutor,
    window: int,
) -> None:
    def _put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _emit(buf: list) -> Tensor:
        arr = np.stack(buf)
        try:
            return torch.from_numpy(arr)
        except TypeError:
            return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    fast_state = {"checked": False, "ok": True}
    try:
        offset = 0
        buf: list[np.ndarray] = []
        for fname in chosen:
            if stop.is_set():
                break
            info = entries[fname]
            n = int(info.n_frames)
            f_lo = max(0, lo - offset)
            f_hi = min(n, hi - offset)
            if f_lo < f_hi:
                for arr in _iter_file_frames(
                    info, f_lo, f_hi, pool, window, stop, fast_state
                ):
                    buf.append(arr)
                    if len(buf) >= batch_size:
                        if not _put(_emit(buf[:batch_size])):
                            return
                        buf = buf[batch_size:]
            offset += n
            if offset >= hi:
                break
        if buf and not stop.is_set():
            _put(_emit(buf))
    finally:
        _put(_PREFETCH_SENTINEL)


def iter_frames(
    loader: DataLoader,
    *,
    files: Optional[Sequence[str]] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
    prefetch: int = 2,
    decode_workers: int = 8,
) -> Iterator[Tensor]:
    """Stream frames from a loader as tensors, prefetching reads off-thread.

    A background worker reads HDF5 slices into a bounded queue (up to ``prefetch``
    batches ahead) so disk I/O overlaps compute. Bitshuffle-LZ4 chunked datasets
    are read raw and decoded across a ``decode_workers`` thread pool.

    Parameters
    ----------
    loader : DataLoader
        Source loader whose ``metadata.files`` are read.
    files : sequence of str, optional
        Subset of filenames to stream; defaults to all files in the loader.
    start, stop : int, optional
        Global frame index range ``[start, stop)``; defaults to the full run.
    device : torch.device, optional
        Target device; frames are cast to ``dtype`` after transfer.
    dtype : torch.dtype, default torch.float32
        Output tensor dtype.
    batch_size : int, default 1
        Frames per yielded tensor.
    prefetch : int, default 2
        Max batches the worker reads ahead.
    decode_workers : int, default 8
        Threads used to decode bitshuffle-LZ4 chunks in parallel.

    Yields
    ------
    torch.Tensor
        A single frame ``(H, W)`` when ``batch_size == 1``, else a stacked
        batch ``(B, H, W)``.
    """
    metadata = loader.metadata
    entries = metadata.files
    chosen = list(entries.keys()) if files is None else list(files)
    lo = int(start) if start is not None else 0
    hi = int(stop) if stop is not None else metadata.n_frames

    nworkers = max(1, min(int(decode_workers), os.cpu_count() or 4))
    window = max(int(batch_size), nworkers)
    pool = ThreadPoolExecutor(max_workers=nworkers)

    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_prefetch_worker,
        args=(entries, chosen, lo, hi, q, batch_size, stop_event, pool, window),
        daemon=True,
    )
    worker.start()

    try:
        while True:
            item = q.get()
            if item is _PREFETCH_SENTINEL:
                break
            t = item
            if device is not None:
                t = t.to(device, non_blocking=True)
            t = t.to(dtype)
            yield t[0] if batch_size == 1 else t
    finally:
        stop_event.set()
        while worker.is_alive():
            try:
                if q.get(timeout=0.1) is _PREFETCH_SENTINEL:
                    break
            except queue.Empty:
                continue
        worker.join()
        pool.shutdown(wait=False)
