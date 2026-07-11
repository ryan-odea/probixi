from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import h5py
import hdf5plugin  # noqa: F401  (registers bitshuffle)

from .assemble import PanelPlacement, assembled_frame_shape, build_placements
from .cell import CellParams
from .geometry import Geometry

PathLike = Union[str, Path]


@dataclass
class H5Info:
    """Locator and shape for the frame stack inside one HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    dataset : str
        Internal path to the frame dataset within the file.
    n_frames : int
        Number of frames (size of the event axis).
    frame_shape : tuple[int, ...]
        Shape of a single assembled frame ``(ss, fs)`` in data space.
    raw_shape : tuple[int, ...]
        Full shape of the source HDF5 array (before assembly).
    placements : list[PanelPlacement], optional
        Panel placement records for the assembly path; ``None`` selects the fast
        contiguous-slice path for a plain ``(N, ss, fs)`` stack.
    """

    filename: str
    dataset: str
    n_frames: int
    frame_shape: tuple[int, ...]
    raw_shape: tuple[int, ...] = ()
    placements: Optional[list[PanelPlacement]] = None


@dataclass
class Metadata:
    """Everything the loader resolved up front about a run.

    Parameters
    ----------
    files : dict
        Map from filename to its :class:`H5Info` (dataset path, counts, shape).
    geometry : Geometry, optional
        Parsed detector geometry, or None if no geometry file was given.
    cell : CellParams, optional
        Parsed target unit cell, or None if no cell file was given.
    frame_size : tuple, optional
        Detector frame shape ``(rows, cols)``, or None if no files.
    n_files : int
        Number of source files.
    n_frames : int
        Total frame count summed across all files.
    entry_point : int, optional
        Optional starting frame offset for the run.
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


def scan_h5(path: PathLike, geometry: Optional[Geometry] = None) -> H5Info:
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
        raise ValueError(
            f"per-event data paths ('%' in {layout.data_path!r}) are not supported"
        )
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
    # Fail fast at load so assembly never trips inside the prefetch worker.
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
    # First dataset encountered in the tree is the frame stack.
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
