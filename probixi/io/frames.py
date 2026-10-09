from __future__ import annotations

import os
import queue
import struct
import threading
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import h5py
import hdf5plugin  # noqa: F401  (registers bitshuffle)
import numpy as np
import torch
from torch import Tensor

from .assemble import assemble_batch
from .cbf import UnsupportedCbf, decode_cbf, is_cbf, read_cbf, read_cbf_data
from .cell import CellParams, read_crystfel_cell
from .geometry import Geometry, read_geometry, resolve_dynamic_fields
from .metadata import CbfInfo, H5Info, Metadata, scan_cbf, scan_h5

try:
    import bitshuffle

    _HAS_BSHUF = True
except Exception:  # pragma: no cover
    bitshuffle = None
    _HAS_BSHUF = False

PathLike = Union[str, Path]

_BSHUF_FILTER_ID = 32008
_BSHUF_LZ4 = 2


class DataLoader:
    """Resolve HDF5 or CBF files, geometry, and cell up front for frame loading.

    Parameters
    ----------
    list_file : str or Path
        Text file listing HDF5 or CBF paths or ``path //event`` entries in
        processing order (``#``/``;`` comments allowed). Consecutive events share
        batched IO. A CBF file (``.cbf``, ``.cbf.gz``, ``.cbf.bz2``) holds one
        image, so ``//0`` is its only event; unreadable files are skipped with a
        warning, but a CBF variant the reader does not decode raises
        :class:`~probixi.io.cbf.UnsupportedCbf`.
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
        files: dict[str, Union[H5Info, CbfInfo]] = {}
        frame_size: Optional[tuple] = None
        total_frames = 0
        skipped = 0
        entries = self._read_list()
        cache = _scan_cbf_files([entry.rsplit(" //", 1)[0] for entry in entries])
        previous = None
        for entry in entries:
            parts = entry.rsplit(" //", 1)
            path = parts[0]
            event = int(parts[1]) if len(parts) == 2 and parts[1] else None
            try:
                if path not in cache:
                    cache[path] = scan_h5(path, geometry)
                info = cache[path]
                if isinstance(info, Exception):
                    raise info
            except UnsupportedCbf:
                raise
            except Exception as exc:
                warnings.warn(f"skipping unreadable file {path!r}: {exc}")
                skipped += 1
                continue
            if frame_size is None:
                frame_size = info.frame_shape
            elif frame_size != info.frame_shape:
                warnings.warn(
                    f"skipping {path!r}: frame size {info.frame_shape} "
                    f"!= expected {frame_size}"
                )
                skipped += 1
                continue
            if event is not None:
                if not 0 <= event < info.n_frames:
                    raise ValueError(
                        f"event {event} outside {path} ({info.n_frames} frames)"
                    )
                info = replace(
                    info, event_start=event, source_n_frames=info.n_frames, n_frames=1
                )
            total_frames += info.n_frames
            if (
                event is not None
                and previous is not None
                and previous.filename == info.filename
                and previous.event_start + previous.n_frames == event
            ):
                previous.n_frames += 1
            else:
                key = entry if entry not in files else f"{entry}#{len(files)}"
                files[key] = replace(info)
                previous = files[key]
        if skipped:
            warnings.warn(
                f"skipped {skipped} unreadable file(s); "
                f"{len(files)} readable file(s) remain"
            )
        if not files:
            raise ValueError(
                f"no readable files in {self.list_file} ({skipped} skipped)"
            )
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
            n_files=len({info.filename for info in files.values()}),
            n_frames=total_frames,
        )


def _scan_cbf_files(paths: Sequence[str]) -> dict:
    # Header reads of thousands of files wait on the filesystem, so they overlap.
    # A failed scan is kept as the exception for the caller to report per entry.
    paths = [path for path in dict.fromkeys(paths) if is_cbf(path)]

    def scan(path):
        try:
            return scan_cbf(path)
        except Exception as exc:  # noqa: BLE001
            return exc

    with ThreadPoolExecutor(max_workers=64) as pool:
        return dict(zip(paths, pool.map(scan, paths)))


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
        assert bitshuffle is not None
        blk = struct.unpack(">i", raw[8:12])[0]
        comp = np.frombuffer(raw[12:], dtype=np.uint8)
        return bitshuffle.decompress_lz4(comp, shape, dt, blk // itemsize if blk else 0)

    return decode


def _iter_file_frames(info, f_lo, f_hi, pool, window, stop, fast_state):
    f_lo += info.event_start
    f_hi += info.event_start
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
                assert dset is not None
                raws = [dset.id.read_direct_chunk((j, 0, 0))[1] for j in range(i, end)]
                for arr in pool.map(decoder, raws):
                    yield arr
            elif dset is not None:
                block = np.asarray(dset[i:end])
                for k in range(block.shape[0]):
                    yield block[k]
            else:
                assert info.placements is not None
                block = assemble_batch(f, i, end, info.placements, info.frame_shape)
                for k in range(block.shape[0]):
                    yield block[k]
            i = end


def _read_cbf_raw(path):
    # Checked data bytes of a byte-offset file for the device; other files decoded
    header, data = read_cbf_data(path)
    if header.compression == "byte_offset":
        return header, data, path
    return decode_cbf(header, data, path)


def _stage(buf, pin):
    # Pack the data bytes of the CBF entries of buf into one pinned buffer
    raws = [e for e in buf if isinstance(e, tuple)]
    if not raws:
        return buf, None, None
    stride = (max(len(e[1]) for e in raws) + 31) // 16 * 16
    with torch.cuda.device(pin):
        stage = torch.empty((len(raws), stride), dtype=torch.uint8, pin_memory=True)
        lens = torch.tensor([len(e[1]) for e in raws], dtype=torch.int32).pin_memory()
    for row, (_, data, _) in zip(stage.numpy(), raws):
        row[: len(data)] = np.frombuffer(data, dtype=np.uint8)
        row[len(data) :] = 0
    return buf, stage, lens


def _device_frames(item, kernel, device, dtype):
    # Frames of one staged item on the device, in order
    buf, stage, lens = item
    if stage is not None:
        stage = stage.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
    frames, row, i = [], 0, 0
    while i < len(buf):
        if not isinstance(buf[i], tuple):
            try:
                frame = torch.from_numpy(buf[i])
            except TypeError:
                frame = torch.from_numpy(np.ascontiguousarray(buf[i], np.float32))
            frames.append(frame.to(device).to(dtype))
            i += 1
            continue
        header = buf[i][0]
        j = i
        while (
            j < len(buf)
            and isinstance(buf[j], tuple)
            and buf[j][0].shape == header.shape
            and buf[j][0].dtype == header.dtype
        ):
            j += 1
        m = j - i
        names = [str(e[2]) for e in buf[i:j]]
        count = header.shape[0] * header.shape[1]
        out = kernel.decode(
            stage[row : row + m], lens[row : row + m], count, header.dtype, names
        )
        frames.extend(out.view(m, *header.shape).to(dtype).unbind(0))
        row, i = row + m, j
    return frames


def _iter_cbf(entries, chosen, lo, hi, pool, window, read=read_cbf):
    # Decoded frames of the selected CBF entries in order, ``window`` files ahead
    paths, offset = [], 0
    for name in chosen:
        info = entries[name]
        if offset >= hi:
            break
        if isinstance(info, CbfInfo) and offset >= lo:
            paths.append(info.filename)
        offset += info.n_frames
    paths = iter(paths)
    pending = deque(pool.submit(read, path) for path in islice(paths, window))
    while pending:
        yield pending.popleft().result()
        pending.extend(pool.submit(read, path) for path in islice(paths, 1))


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
    pin: Optional[int] = None,
    on_device: bool = False,
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
        if on_device:
            return _stage(buf, pin)
        if pin is not None:
            # pinned staging keeps the host-to-device copy asynchronous
            try:
                dtype = torch.from_numpy(buf[0][:0]).dtype
                with torch.cuda.device(pin):
                    out = torch.empty(
                        (len(buf), *buf[0].shape), dtype=dtype, pin_memory=True
                    )
                np.stack(buf, out=out.numpy())
                return out
            except TypeError:
                pass
        arr = np.stack(buf)
        try:
            return torch.from_numpy(arr)
        except TypeError:
            return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    fast_state = {"checked": False, "ok": True}
    cbf_frames = _iter_cbf(
        entries, chosen, lo, hi, pool, window, _read_cbf_raw if on_device else read_cbf
    )
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
                frames = (
                    [next(cbf_frames)]
                    if isinstance(info, CbfInfo)
                    else _iter_file_frames(
                        info, f_lo, f_hi, pool, window, stop, fast_state
                    )
                )
                for arr in frames:
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
    except Exception as exc:  # noqa: BLE001 -- forward worker failures to the consumer
        _put(exc)
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

    A background worker reads into a bounded queue to overlap IO and compute.
    When CUDA, Triton and nvCOMP are available, supported full-frame chunks are
    transferred compressed and decoded on device, including simple whole-frame
    VDS mappings. Other formats retain HDF5 decoding. GPU input uses internal
    batches of eight without changing the yielded batch shape. The reference
    path optionally decodes bitshuffle/LZ4 across ``decode_workers`` threads.
    CBF files are read several files ahead across twice ``decode_workers``
    threads; byte-offset data are decoded on the device, other CBF data and
    CPU devices on the CPU, then moved to ``device`` from pinned memory. A list
    that selects any CBF file takes this path for all of its files.
    Use ``probixi.kernels.use_engine`` to force a backend for comparisons.

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
        Threads used to decode bitshuffle-LZ4 chunks or CBF files in parallel.

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

    from ..kernels import Engine, _engine, select

    device = torch.device(device) if device is not None else None
    cuda = device is not None and device.type == "cuda"
    has_cbf = any(isinstance(entries[name], CbfInfo) for name in chosen)
    cbf_kernel = None
    if has_cbf:
        cbf_kernel = select("cbf", cuda)
        kernel = None
    else:
        kernel = select("decode", cuda)
    if kernel is not None:
        if kernel.library() is not None:
            from ._compressed import iter_gpu_frames

            with torch.cuda.device(device):
                yield from iter_gpu_frames(
                    loader,
                    chosen,
                    lo,
                    hi,
                    device,
                    dtype,
                    batch_size,
                    prefetch,
                    kernel,
                    _engine.get() is Engine.ACCELERATED,
                )
            return
        if _engine.get() is Engine.ACCELERATED:
            raise RuntimeError("nvCOMP 5.x CUDA library is unavailable")

    nworkers = max(1, min(int(decode_workers), os.cpu_count() or 4))
    if has_cbf:
        nworkers *= 2  # file reads wait on the filesystem: more in flight than cores
    window = max(int(batch_size), nworkers)
    pool = ThreadPoolExecutor(max_workers=nworkers)

    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
    stop_event = threading.Event()
    pin = None
    if has_cbf and cuda:
        pin = device.index if device.index is not None else torch.cuda.current_device()
    worker = threading.Thread(
        target=_prefetch_worker,
        args=(
            entries,
            chosen,
            lo,
            hi,
            q,
            8 if cbf_kernel else batch_size,
            stop_event,
            pool,
            window,
            pin,
            cbf_kernel is not None,
        ),
        daemon=True,
    )
    worker.start()

    pending: list[Tensor] = []
    try:
        while True:
            item = q.get()
            if item is _PREFETCH_SENTINEL:
                if pending:
                    yield torch.stack(pending)
                break
            if isinstance(item, Exception):
                raise item
            if cbf_kernel is not None:
                with torch.cuda.device(device):
                    frames = _device_frames(item, cbf_kernel, device, dtype)
                for frame in frames:
                    if batch_size == 1:
                        yield frame
                    else:
                        pending.append(frame)
                        if len(pending) == batch_size:
                            yield torch.stack(pending)
                            pending.clear()
                continue
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
