from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import h5py
import hdf5plugin  # noqa: F401  (registers bitshuffle)

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
        Internal path to the 3-D frame dataset within the file.
    n_frames : int
        Number of frames (size of the leading dataset axis).
    frame_shape : tuple[int, ...]
        Shape of a single frame (the trailing axes).
    """

    filename: str
    dataset: str
    n_frames: int
    frame_shape: tuple[int, ...]


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


def scan_h5(path: PathLike) -> H5Info:
    # Locate frame stack and read its shape.
    path = Path(path)
    with h5py.File(path, "r") as f:
        dataset_path, shape = _find_frame_dataset(f)
    return H5Info(
        filename=str(path),
        dataset=dataset_path,
        n_frames=int(shape[0]),
        frame_shape=tuple(shape[1:]),
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
