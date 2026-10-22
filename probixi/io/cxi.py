from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from .metadata import CbfInfo
from .writer import _build_frame_ranges

PathLike = str | Path

PEAK_LIST_PATH = "/entry_1/result_1"  # Cheetah's CXI peak-list group


class PeakOffloader:
    """Write peak-search results as CXI files for ``indexamajig --peaks=cxi``.

    One ``.cxi`` is written per source HDF5 file. It external-links the raw image
    stack and stores the peaks under ``/entry_1/result_1``, indexed by event. The
    directory also receives ``peaks.lst`` and ``<geometry>_cxi.geom``, the input
    geometry with ``peak_list`` and ``peak_list_type`` appended. Nothing is written
    until the context exits.

    Parameters
    ----------
    path : str or Path
        Output directory, created on entry.
    geometry_file : str or Path, optional
        Input geometry file. Without it no ``_cxi.geom`` is written.
    files : dict, optional
        Loader file map, list entry to :class:`~probixi.io.metadata.H5Info`, which
        resolves each frame to its source file and event.

    Raises
    ------
    ValueError
        If ``files`` holds CBF or per-event input, which has no image stack to
        link. Write a DuckDB output for such input.

    Notes
    -----
    Use as a context manager. The instance is callable, so it can be passed to
    ``PeakStream.for_each``::

        with PeakOffloader(out_dir, geometry_file=gf, files=meta.files) as off:
            for result in pipeline.peak_stream(frames):
                off.write(result)
    """

    def __init__(
        self,
        path: PathLike,
        *,
        geometry_file: Optional[PathLike] = None,
        files: Optional[dict] = None,
    ):
        if any(
            isinstance(info, CbfInfo) or getattr(info, "events", None)
            for info in (files or {}).values()
        ):
            raise ValueError(
                "the CXI peak export links the HDF5 image stack and cannot "
                "reference CBF or per-event input; write a .duckdb output instead"
            )
        self.out_dir = Path(path)
        self._geometry_file = Path(geometry_file) if geometry_file else None
        self._ranges = _build_frame_ranges(files)
        self._event_starts = [
            getattr(info, "event_start", 0) for info in (files or {}).values()
        ]
        self._info = {}
        for fname, info in (files or {}).items():
            self._info[str(getattr(info, "filename", fname))] = info
        self._buf: dict[str, dict[int, tuple[list, list, list]]] = {}
        self._active = False

    def __enter__(self) -> "PeakOffloader":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._active = True
        return self

    def __exit__(self, *exc) -> None:
        self.flush()
        self._active = False

    def __call__(self, result) -> None:
        self.write(result)

    def _locate(self, frame_index: Optional[int]) -> tuple[str, int]:
        if frame_index is None:
            return "unknown", 0
        for i, (start, stop, fname) in enumerate(self._ranges):
            if start <= frame_index < stop:
                return fname, frame_index - start + self._event_starts[i]
        return "unknown", int(frame_index)

    def write(self, result) -> None:
        """Buffer the kept peaks of one frame; frames without peaks add nothing."""
        if not self._active:
            raise RuntimeError("PeakOffloader must be used as a context manager")
        stats = result.kept_stats
        rows, cols, intensities = (
            torch.stack([stats.row_centroid, stats.col_centroid, stats.intensity_sum])
            .detach()
            .cpu()
            .tolist()
        )
        if not rows:
            return
        fname, event = self._locate(result.frame_index)
        self._buf.setdefault(fname, {})[event] = (cols, rows, intensities)  # (fs, ss)

    def flush(self) -> None:
        """Write the buffered peaks, ``peaks.lst`` and the geometry; runs on exit."""
        lst_lines: list[str] = []
        for fname in sorted(self._buf):
            info = self._info.get(fname)
            if info is None:
                continue
            cxi_path = self.out_dir / (Path(fname).stem + ".cxi")
            self._write_cxi(cxi_path, fname, info, self._buf[fname])
            lst_lines.append(str(cxi_path.resolve()))

        if lst_lines:
            (self.out_dir / "peaks.lst").write_text("\n".join(lst_lines) + "\n")
            self._write_geometry()
        self._buf.clear()

    def _write_cxi(self, cxi_path, raw_file, info, events) -> None:
        n_frames = int(
            getattr(info, "source_n_frames", None) or getattr(info, "n_frames", 0)
        ) or (max(events) + 1)
        dataset = str(getattr(info, "dataset", "/entry/data/data"))
        max_peaks = max((len(fs) for fs, _, _ in events.values()), default=1)
        max_peaks = max(max_peaks, 1)

        npeaks = np.zeros(n_frames, dtype=np.int32)
        xpos = np.zeros((n_frames, max_peaks), dtype=np.float32)
        ypos = np.zeros((n_frames, max_peaks), dtype=np.float32)
        totint = np.zeros((n_frames, max_peaks), dtype=np.float32)

        for event, (fs, ss, inten) in events.items():
            if not (0 <= event < n_frames):
                continue
            k = len(fs)
            npeaks[event] = k
            xpos[event, :k] = np.asarray(fs, dtype=np.float32)
            ypos[event, :k] = np.asarray(ss, dtype=np.float32)
            totint[event, :k] = np.asarray(inten, dtype=np.float32)

        with h5py.File(cxi_path, "w") as f:
            f[dataset] = h5py.ExternalLink(str(Path(raw_file).resolve()), dataset)
            f.create_dataset(f"{PEAK_LIST_PATH}/nPeaks", data=npeaks)
            f.create_dataset(f"{PEAK_LIST_PATH}/peakXPosRaw", data=xpos)
            f.create_dataset(f"{PEAK_LIST_PATH}/peakYPosRaw", data=ypos)
            f.create_dataset(f"{PEAK_LIST_PATH}/peakTotalIntensity", data=totint)

    def _write_geometry(self) -> None:
        if self._geometry_file is None:
            return
        text = self._geometry_file.read_text()
        out = self.out_dir / (self._geometry_file.stem + "_cxi.geom")
        lines = [text.rstrip("\n")]
        if "peak_list " not in text and "peak_list=" not in text:
            lines.append(f"peak_list = {PEAK_LIST_PATH}")
        if "peak_list_type" not in text:
            lines.append("peak_list_type = cxi")
        out.write_text("\n".join(lines) + "\n")
