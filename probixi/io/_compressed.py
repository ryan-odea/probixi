import queue
import struct
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from .assemble import assemble_batch


class Unsupported(ValueError):
    """A valid format that this accelerator cannot decode."""


@dataclass(frozen=True)
class Layout:
    dtype: np.dtype
    shape: tuple
    filters: tuple
    codec: str | None

    @classmethod
    def read(cls, d):
        if d.ndim != 3 or d.chunks != (1, *d.shape[1:]):
            raise Unsupported("requires one complete frame per chunk")
        dt = d.dtype
        if (
            dt.kind not in "uif"
            or dt.itemsize not in (1, 2, 4, 8)
            or dt.byteorder == ">"
        ):
            raise Unsupported("unsupported pixel datatype")
        p = d.id.get_create_plist()
        filters = tuple(
            (p.get_filter(i)[0], p.get_filter(i)[2]) for i in range(p.get_nfilters())
        )
        ids = [fid for fid, _ in filters]
        if not ids:
            raise Unsupported("uncompressed input uses the ordinary transfer path")
        if ids[0] == 2:
            ids = ids[1:]
        if len(ids) != 1 or ids[0] not in (1, 32004, 32008, 32015):
            raise Unsupported("unsupported HDF5 filter pipeline")
        fid, cd = filters[-1]
        if fid == 32008:
            if len(filters) != 1 or len(cd) < 3 or cd[2] != dt.itemsize:
                raise Unsupported("unsupported bitshuffle parameters")
            codec = {0: None, 2: "LZ4", 3: "Zstd"}.get(
                cd[4] if len(cd) > 4 else 0, "unknown"
            )
            if codec == "unknown":
                raise Unsupported("unsupported bitshuffle compressor")
        else:
            codec = {1: "Deflate", 32004: "LZ4", 32015: "Zstd"}[fid]
        size = int(np.prod(d.shape[1:])) * dt.itemsize
        if size >= 2**31:
            raise Unsupported("chunk exceeds supported decoder size")
        return cls(dt, tuple(d.shape[1:]), filters, codec)

    @property
    def frame_bytes(self):
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def blocks(self, raw, mask):
        """Return compressed blocks (array or list), verbatim copies, shuffle settings
        and checksum."""
        size = self.frame_bytes
        active = [
            (fid, cd) for i, (fid, cd) in enumerate(self.filters) if not mask & (1 << i)
        ]
        if mask >> len(self.filters):
            raise ValueError("invalid HDF5 skipped-filter mask")
        byte_shuffle = bool(active and active[0][0] == 2)
        if byte_shuffle:
            active = active[1:]
        if not active:
            if len(raw) != size:
                raise ValueError("raw HDF5 chunk size mismatch")
            return [], [(0, 0, size)], (0, byte_shuffle), None
        fid, cd = active[0]
        if fid == 1:
            if (
                len(raw) < 6
                or raw[0] & 15 != 8
                or raw[0] >> 4 > 7
                or (raw[0] * 256 + raw[1]) % 31
            ):
                raise ValueError("invalid HDF5 zlib framing")
            if raw[1] & 32:
                raise Unsupported("zlib preset dictionary")
            return (
                [(2, len(raw) - 6, 0, size)],
                [],
                (0, byte_shuffle),
                struct.unpack(">I", raw[-4:])[0],
            )
        if fid == 32015:
            self._zstd(raw)
            return [(0, len(raw), 0, size)], [], (0, byte_shuffle), None
        if fid == 32008 and self.codec is None:
            block = (cd[3] if len(cd) > 3 else 0) or max(
                128, (8192 // self.dtype.itemsize) // 8 * 8
            )
            if len(raw) != size or block % 8:
                raise ValueError("invalid uncompressed bitshuffle block")
            return [], [(0, 0, size)], (block * self.dtype.itemsize, 0), None
        if len(raw) < 12:
            raise ValueError("truncated HDF5 compressed header")
        n, block = struct.unpack_from(">QI", raw)
        if n != size or block == 0:
            raise ValueError("invalid HDF5 compressed size/block header")
        is_bit = fid == 32008
        if is_bit and block % (8 * self.dtype.itemsize):
            raise ValueError("invalid bitshuffle block size")
        shuffled = size - size % (8 * self.dtype.itemsize) if is_bit else size
        unpack = struct.Struct(">I").unpack_from
        zstd = self._zstd if self.codec == "Zstd" else None
        offset, headers, end = 12, [], len(raw)
        try:
            for _ in range((shuffled + block - 1) // block):
                headers.append(offset)
                length = unpack(raw, offset)[0]
                offset += 4 + length
                if length == 0 or offset > end:
                    raise ValueError("truncated compressed block")
                if zstd:
                    zstd(raw[offset - length : offset])
        except struct.error:
            raise ValueError("truncated compressed block header") from None
        headers = np.array(headers, dtype=np.int64)
        dest = np.arange(len(headers), dtype=np.int64) * block
        blocks = np.stack(
            [
                headers + 4,
                np.diff(headers, append=offset) - 4,
                dest,
                np.minimum(block, shuffled - dest),
            ],
            axis=1,
        )
        copies = []
        if not is_bit:
            plain = blocks[:, 1] == blocks[:, 3]
            copies = [(s, d, n) for s, n, d, _ in blocks[plain].tolist()]
            blocks = blocks[~plain]
        if shuffled < size:
            copies.append((offset, shuffled, size - shuffled))
            offset += size - shuffled
        if offset != len(raw):
            raise ValueError("unexpected compressed chunk tail")
        return blocks, copies, (block if is_bit else 0, byte_shuffle), None

    @staticmethod
    def _zstd(raw):
        if len(raw) < 5 or raw[:4] != b"\x28\xb5\x2f\xfd":
            raise ValueError("invalid Zstd frame")
        if raw[4] & 7:
            # Dictionary IDs and optional frame checksums need additional handling.
            raise Unsupported("Zstd dictionary/checksummed frame")


def pack(chunks, parsed, decoder):
    """Pack stored bytes, aligning compressed blocks as required by each codec."""
    records, copies, shuffle, checksums, parts = [], [], [], [], []
    position = 0
    for frame, ((_, raw), (blocks, plain, shuf, checksum)) in enumerate(
        zip(chunks, parsed)
    ):
        base = frame * decoder.stride
        # One host copy per frame when arbitrary compressed-pointer alignment is allowed.
        start = position
        parts.append((start, raw))
        position += len(raw)
        copies.extend((start + s, base + d, n) for s, d, n in plain)
        blocks = np.asarray(blocks, dtype=np.int64).reshape(-1, 4)
        rows = blocks + (start, 0, base, 0)
        if (rows[:, 2] % decoder.align.output).any():
            raise Unsupported("unaligned decoded block")
        for k in np.flatnonzero(rows[:, 0] % decoder.align.input):
            source, length = blocks[k, :2].tolist()
            ptr = (
                (position + decoder.align.input - 1)
                // decoder.align.input
                * decoder.align.input
            )
            parts.append((ptr, memoryview(raw)[source : source + length]))
            position = ptr + length
            rows[k, 0] = ptr
        records.append(rows)
        shuffle.append(shuf)
        checksums.append(checksum)
    host = torch.empty(position, dtype=torch.uint8, pin_memory=True)
    view = host.numpy()
    for offset, data in parts:
        view[offset : offset + len(data)] = np.frombuffer(data, dtype=np.uint8)
    metadata = torch.from_numpy(np.concatenate(records)).pin_memory()
    return host, metadata, copies, shuffle, checksums


def _frame_sources(d, lo, hi, open_dataset):
    if not d.is_virtual:
        return [(d, i) for i in range(lo, hi)]
    shape = tuple(d.shape[1:])
    result = {}
    for mapping in d.virtual_sources():
        filename = mapping.file_name
        if filename == ".":
            filename = d.file.filename
        elif not Path(filename).is_absolute():
            filename = str(Path(d.file.filename).parent / filename)
        try:
            source = open_dataset(filename, mapping.dset_name)
        except (OSError, KeyError):
            raise Unsupported("VDS source missing; use HDF5 fill handling") from None
        if (
            source.is_virtual
            or source.ndim != 3
            or tuple(source.shape[1:]) != shape
            or source.dtype != d.dtype
        ):
            raise Unsupported("complex VDS source")
        selections = []
        for space, full_shape in [
            (mapping.vspace, d.shape),
            (mapping.src_space, source.shape),
        ]:
            if space.get_select_type() == h5py.h5s.SEL_ALL:
                blocks = [(np.zeros(3, dtype=int), np.array(full_shape) - 1)]
            elif space.get_select_type() == h5py.h5s.SEL_HYPERSLABS:
                blocks = space.get_select_hyper_blocklist()
            else:
                raise Unsupported("non-hyperslab VDS mapping")
            frames = []
            for first, last in blocks:
                if tuple(first[1:]) != (0, 0) or tuple(last[1:] + 1) != shape:
                    raise Unsupported("partial-frame VDS mapping")
                frames.extend(range(int(first[0]), int(last[0]) + 1))
            selections.append(frames)
        if len(selections[0]) != len(selections[1]):
            raise Unsupported("mismatched VDS selections")
        for virtual, physical in zip(*selections):
            if lo <= virtual < hi:
                if virtual in result:
                    raise Unsupported("overlapping VDS mappings")
                result[virtual] = (source, physical)
    if len(result) != hi - lo:
        raise Unsupported("VDS fill regions")
    return [result[i] for i in range(lo, hi)]


def _stage(d, start, end):
    """Read a batch into pinned memory so its device copy can be asynchronous."""
    try:
        dtype = torch.from_numpy(np.empty(0, dtype=d.dtype)).dtype
    except (TypeError, ValueError):
        return np.asarray(d[start:end])
    staged = torch.zeros((end - start, *d.shape[1:]), dtype=dtype, pin_memory=True)
    d.read_direct(staged.numpy(), np.s_[start:end])
    return staged


def _read_batches(entries, chosen, lo, hi, strict, decoder_for):
    """CPU producer owns all HDF5 handles and pinned staging; it launches no kernels."""
    with ExitStack() as stack:
        opened = {}

        def open_dataset(filename, name):
            if filename not in opened:
                opened[filename] = stack.enter_context(h5py.File(filename, "r"))
            return opened[filename][name]

        offset = 0
        for filename in chosen:
            info = entries[filename]
            first, last = max(0, lo - offset), min(info.n_frames, hi - offset)
            offset += info.n_frames
            first += info.event_start
            last += info.event_start
            if first >= last:
                continue
            d = (
                open_dataset(info.filename, info.dataset)
                if info.placements is None
                else None
            )
            try:
                if d is None:
                    raise Unsupported("detector assembly")
                sources = _frame_sources(d, first, last, open_dataset)
                layouts = {
                    s.name + "@" + s.file.filename: Layout.read(s) for s, _ in sources
                }
            except Unsupported as exc:
                if strict:
                    raise RuntimeError(str(exc)) from exc
                sources = None
            for start in range(first, last, 8):
                end = min(start + 8, last)
                if sources is not None:
                    selected = sources[start - first : end - first]
                    layout = layouts[
                        selected[0][0].name + "@" + selected[0][0].file.filename
                    ]
                    if all(
                        layouts[s.name + "@" + s.file.filename] == layout
                        for s, _ in selected
                    ):
                        raws, parsed = [], []
                        try:
                            for source, i in selected:
                                if (
                                    source.id.get_chunk_info_by_coord((i, 0, 0)).size
                                    == 0
                                ):
                                    raise Unsupported("unallocated fill chunk")
                                raw = source.id.read_direct_chunk((i, 0, 0))
                                parsed.append(layout.blocks(raw[1], raw[0]))
                                raws.append(raw)
                        except Unsupported as exc:
                            if strict:
                                raise RuntimeError(str(exc)) from exc
                        else:
                            try:
                                payload = pack(raws, parsed, decoder_for(layout))
                            except Unsupported:
                                if strict:
                                    raise
                            else:
                                yield layout, payload
                                continue
                    elif strict:
                        raise RuntimeError("mixed compression inside a decoding batch")
                if d is not None:
                    yield None, _stage(d, start, end)
                else:
                    f = opened.get(info.filename)
                    if f is None:
                        f = stack.enter_context(h5py.File(info.filename, "r"))
                        opened[info.filename] = f
                    yield None, assemble_batch(
                        f, start, end, info.placements, info.frame_shape
                    )


def iter_gpu_frames(
    loader, chosen, lo, hi, device, dtype, batch_size, prefetch, kernel, strict
):
    """Preserve logical batching/order while transporting compressed batches."""
    q, stop = queue.Queue(max(1, prefetch)), threading.Event()
    sentinel = object()
    decoders = {}
    index = torch.cuda.current_device()

    def decoder_for(layout):
        key = (layout.codec, layout.frame_bytes)
        if key not in decoders:
            decoders[key] = kernel.Decoder(layout.codec, layout.frame_bytes, device)
        return decoders[key]

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def produce():
        try:
            torch.cuda.set_device(index)  # new threads default to device 0
            for item in _read_batches(
                loader.metadata.files, chosen, lo, hi, strict, decoder_for
            ):
                if stop.is_set():
                    break
                put(item)
        except Exception as exc:  # noqa: BLE001 -- propagate worker failures
            put(exc)
        finally:
            put(sentinel)

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    pending, copied = [], None
    try:
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            layout, data = item
            if layout is None:
                if not isinstance(data, torch.Tensor):
                    if not data.dtype.isnative:
                        data = data.astype(data.dtype.newbyteorder("="))
                    data = torch.from_numpy(data)
                frames = data.to(
                    device=device, dtype=dtype, non_blocking=data.dtype == dtype
                )
                if copied is not None:  # bound the pinned buffers held by copies
                    copied.synchronize()
                copied = torch.cuda.Event()
                copied.record()
            else:
                original_dtype = torch.from_numpy(np.empty(0, dtype=layout.dtype)).dtype
                decoder = decoder_for(layout)
                frames = decoder.decode(*data, original_dtype, layout.shape).to(dtype)
            for frame in frames.unbind(0):
                if batch_size == 1:
                    yield frame
                else:
                    pending.append(frame)
                    if len(pending) == batch_size:
                        yield torch.stack(pending)
                        pending.clear()
        if pending:
            yield torch.stack(pending)
    finally:
        stop.set()
        worker.join()
