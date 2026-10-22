from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import h5py
import hdf5plugin  # noqa: F401  (registers bitshuffle)

from .assemble import PanelPlacement, assembled_frame_shape, build_placements
from .cbf import read_cbf_header
from .cell import CellParams
from .geometry import Geometry, match_events

PathLike = Union[str, Path]


@dataclass
class H5Info:
    """Frame stack of one HDF5 file, or a contiguous selection of its events.

    Parameters
    ----------
    filename : str
        HDF5 file.
    dataset : str
        Frame dataset; holds ``%`` for a per-event path such as ``/%/data``.
    n_frames : int
        Frames this entry selects.
    frame_shape : tuple[int, ...]
        Assembled frame shape ``(ss, fs)`` in data space.
    raw_shape : tuple[int, ...]
        Shape of the stored array before assembly.
    placements : list[PanelPlacement], optional
        Panel placements when frames are assembled from the stored array;
        ``None`` for a plain ``(N, ss, fs)`` stack, which is read by slicing.
    event_start : int, default 0
        First selected source event.
    source_n_frames : int, optional
        Length of the source stack when the entry selects some of its events.
    events : list[str], optional
        Names that ``%`` stands for, one frame each.
    photon_energy_eV : list[float], optional
        Per-event photon energies in eV from a ``photon_energy = /%/...`` path.
    """

    filename: str
    dataset: str
    n_frames: int
    frame_shape: tuple[int, ...]
    raw_shape: tuple[int, ...] = ()
    placements: Optional[list[PanelPlacement]] = None
    event_start: int = 0
    source_n_frames: Optional[int] = None
    events: Optional[list[str]] = None
    photon_energy_eV: Optional[list[float]] = None


@dataclass
class CbfInfo:
    """Single image of one CBF file.

    Parameters
    ----------
    filename : str
        ``.cbf``, ``.cbf.gz`` or ``.cbf.bz2`` file.
    frame_shape : tuple[int, ...]
        Image shape ``(ss, fs)``.
    dtype : str
        Decoded pixel type, for example ``"int32"``.
    n_frames : int, default 1
        Always 1.
    event_start : int, default 0
        Always 0; ``//0`` is the only event.
    source_n_frames : int, optional
        Set when the list entry names ``//0``. ``None`` for an entry without an
        event, which the stream writes as the empty event ``//``, as CrystFEL does.
    """

    filename: str
    frame_shape: tuple[int, ...]
    dtype: str
    n_frames: int = 1
    event_start: int = 0
    source_n_frames: Optional[int] = None

    @property
    def eventless(self) -> bool:
        return self.source_n_frames is None


@dataclass
class Metadata:
    """Files, geometry and cell that :class:`DataLoader` resolved for a run.

    Parameters
    ----------
    files : dict
        List entry to :class:`H5Info` or :class:`CbfInfo`, in processing order.
    geometry : Geometry, optional
        Parsed geometry; ``None`` without a geometry file.
    cell : CellParams, optional
        Target cell; ``None`` without a cell file.
    frame_size : tuple, optional
        Frame shape ``(ss, fs)``.
    n_files : int
        Distinct source files.
    n_frames : int
        Frames over all entries.
    entry_point : int, optional
        Starting frame offset; :class:`DataLoader` does not set it.
    """

    files: dict = field(default_factory=dict)
    geometry: Optional[Geometry] = None
    cell: Optional[CellParams] = None
    frame_size: Optional[tuple] = None
    n_files: int = 0
    n_frames: int = 0
    entry_point: Optional[int] = None

    @property
    def distance(self) -> Optional[float]:
        return self.geometry.distance if self.geometry else None


def scan_cbf(path: PathLike) -> CbfInfo:
    """Describe the image of one CBF file from its header alone."""
    header = read_cbf_header(path)
    return CbfInfo(
        filename=str(Path(path)),
        frame_shape=header.shape,
        dtype=header.dtype.name,
    )


def scan_h5(path: PathLike, geometry: Optional[Geometry] = None) -> H5Info:
    """Locate the frames of one HDF5 file by the geometry's ``data`` and ``dimN``.

    Without a ``data`` path the first 3-D dataset in the file is the frame stack.
    """
    path = Path(path)
    layout = geometry.data_layout if geometry is not None else None
    if layout is None or layout.data_path is None:
        with h5py.File(path, "r") as f:
            dataset_path, shape = _find_frame_dataset(f)
        return H5Info(
            filename=str(path),
            dataset=dataset_path,
            n_frames=int(shape[0]),
            frame_shape=tuple(shape[1:]),
            raw_shape=tuple(shape),
            placements=None,
        )

    assert geometry is not None  # layout is non-None, so geometry was too
    if "%" in str(layout.data_path):
        return _scan_events(path, str(layout.data_path), geometry)
    with h5py.File(path, "r") as f:
        node = f.get(layout.data_path)
        if not isinstance(node, h5py.Dataset):
            raise ValueError(
                f"data path {layout.data_path!r} is not a dataset in {path}"
            )
        raw_shape = tuple(int(s) for s in node.shape)

    needs_assembly = (
        bool(layout.fixed)
        or bool(geometry.panel_layouts)
        or _explicit_nonstandard(layout)
    )
    if not needs_assembly and len(raw_shape) == 3:
        return H5Info(
            filename=str(path),
            dataset=str(layout.data_path),
            n_frames=int(raw_shape[0]),
            frame_shape=(int(raw_shape[1]), int(raw_shape[2])),
            raw_shape=raw_shape,
            placements=None,
        )

    placements = build_placements(geometry)
    if not placements:
        if len(raw_shape) == 3:
            return H5Info(
                filename=str(path),
                dataset=str(layout.data_path),
                n_frames=int(raw_shape[0]),
                frame_shape=(int(raw_shape[1]), int(raw_shape[2])),
                raw_shape=raw_shape,
                placements=None,
            )
        raise ValueError(f"could not derive panel layout for {path}")
    n_frames = int(raw_shape[layout.event_axis]) if layout.event_axis is not None else 1
    frame_shape = assembled_frame_shape(placements)
    _validate_placements(placements, layout.data_path, raw_shape)
    return H5Info(
        filename=str(path),
        dataset=str(layout.data_path),
        n_frames=n_frames,
        frame_shape=frame_shape,
        raw_shape=raw_shape,
        placements=placements,
    )


def _scan_events(path: Path, pattern: str, geometry: Geometry) -> H5Info:
    energy = geometry.parameters.get("photon_energy")
    energies = None
    with h5py.File(path, "r") as f:
        events = match_events(f, pattern)
        if not events:
            raise ValueError(f"no datasets match {pattern!r} in {path}")
        shape = tuple(int(s) for s in f[pattern.replace("%", events[0])].shape)
        if isinstance(energy, str) and "%" in energy:
            energies = [float(f[energy.replace("%", n)][()]) for n in events]
    if len(shape) != 2:
        raise ValueError(f"per-event dataset {pattern!r} is {len(shape)}D, not 2D")
    return H5Info(
        filename=str(path),
        dataset=pattern,
        n_frames=len(events),
        frame_shape=shape,
        raw_shape=shape,
        events=events,
        photon_energy_eV=energies,
    )


def _explicit_nonstandard(layout) -> bool:
    if layout.event_axis is None and layout.ss_axis is None and layout.fs_axis is None:
        return False
    return not (
        len(layout.dims) == 3
        and layout.event_axis == 0
        and layout.ss_axis == 1
        and layout.fs_axis == 2
    )


def _validate_placements(
    placements: list[PanelPlacement], scanned_path: str, raw_shape: tuple[int, ...]
) -> None:
    # Fail at load, not later inside the prefetch worker
    nd = len(raw_shape)
    for p in placements:
        if p.data_path != scanned_path:
            continue
        if p.ndim != nd:
            raise ValueError(
                f"panel {p.name!r}: layout dims {p.ndim} != dataset ndim {nd}"
            )
        for ax, sel in p.fixed.items():
            if not 0 <= sel < raw_shape[ax]:
                raise ValueError(
                    f"panel {p.name!r}: fixed index {sel} out of range on axis {ax} "
                    f"(size {raw_shape[ax]})"
                )
        exp_ss = p.max_ss - p.min_ss + 1
        exp_fs = p.max_fs - p.min_fs + 1
        assert p.ss_axis is not None and p.fs_axis is not None
        got_ss, got_fs = raw_shape[p.ss_axis], raw_shape[p.fs_axis]
        if got_ss != exp_ss or got_fs != exp_fs:
            raise ValueError(
                f"panel {p.name!r}: dataset slab ({got_ss}, {got_fs}) != panel "
                f"region ({exp_ss}, {exp_fs})"
            )


def _find_frame_dataset(h5file: h5py.File) -> tuple[str, tuple]:
    # The first 3-D dataset in visit order is the frame stack
    found: dict = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            found["path"] = name
            found["shape"] = tuple(obj.shape)
            return True
        return None

    h5file.visititems(visitor)
    if "path" not in found:
        raise ValueError(f"No 3D dataset found in {h5file.filename!r}")
    return found["path"], found["shape"]
