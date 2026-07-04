from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .writer import _build_frame_ranges

PathLike = str | Path

PEAK_LIST_PATH = "/entry_1/result_1"


class PeakOffloader:
    """Write peak-search results as CrystFEL-readable CXI files.

    Exports peaks in the CXI layout so that ``indexamajig --peaks=cxi`` can
    re-index probixi's peaks with its own engine. One ``.cxi`` is written per source HDF5 file; each external-links
    the raw image stack and stores the peak arrays under ``/entry_1/result_1`` indexed by event.

    Alongside the ``.cxi`` files it writes ``peaks.lst`` and a companion
    geometry file (the input geometry plus ``peak_list``/``peak_list_type``),
    so the output directory is drop-in for indexamajig.

    Use as a context manager; the instance is the per-frame writer expected by
    ``PeakStream.for_each`` (``__call__`` forwards to :meth:`write`)::

        with PeakOffloader(out_dir, geometry=geom, geometry_file=gf,
                           files=meta.files) as off:
            for result in pipeline.peak_stream(frames):
                off.write(result)

    Parameters
    ----------
    path : str or Path
        Output directory (created if absent).
    geometry_file : str or Path, optional
        Input geometry file; copied verbatim with peak-list keys appended.
    files : dict
        Loader file map (``filename`` -> :class:`~probixi.io.metadata.H5Info`),
        used to resolve frames to their source file/event and to external-link
        the raw image stack.
    """

    def __init__(
        self,
        path: PathLike,
        *,
        geometry_file: Optional[PathLike] = None,
        files: Optional[dict] = None,
    ):
        self.out_dir = Path(path)
        self._geometry_file = Path(geometry_file) if geometry_file else None
        self._ranges = _build_frame_ranges(files)
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
        for start, stop, fname in self._ranges:
            if start <= frame_index < stop:
                return fname, frame_index - start
        return "unknown", int(frame_index)

    def write(self, result) -> None:
        if not self._active:
            raise RuntimeError("PeakOffloader must be used as a context manager")
        stats = result.kept_stats
        rows = stats.row_centroid.detach().cpu().tolist()
        cols = stats.col_centroid.detach().cpu().tolist()
        intensities = stats.intensity_sum.detach().cpu().tolist()
        if not rows:
            return
        fname, event = self._locate(result.frame_index)
        self._buf.setdefault(fname, {})[event] = (cols, rows, intensities)

    def flush(self) -> None:
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
        n_frames = int(getattr(info, "n_frames", 0)) or (max(events) + 1)
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
