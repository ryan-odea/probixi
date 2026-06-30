from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import h5py
import hdf5plugin  # noqa: F401  (registers bitshuffle)
import numpy as np

from .geometry import Geometry, parse_axis_vector


@dataclass
class PanelPlacement:
    name: str
    min_ss: int
    max_ss: int
    min_fs: int
    max_fs: int
    data_path: Optional[str]
    ndim: int
    event_axis: Optional[int]
    ss_axis: Optional[int]
    fs_axis: Optional[int]
    fixed: dict


def build_placements(geometry: Geometry) -> Optional[list[PanelPlacement]]:
    base = geometry.data_layout
    panels = geometry.panels or {}
    if not panels:
        return None
    placements: list[PanelPlacement] = []
    for name, panel in panels.items():
        layout = geometry.panel_layouts.get(name) or base
        if layout is None or layout.ss_axis is None or layout.fs_axis is None:
            return None
        data_path = layout.data_path or (base.data_path if base else None)
        if data_path is None:
            return None
        try:
            placements.append(
                PanelPlacement(
                    name=str(name),
                    min_ss=int(panel["min_ss"]),
                    max_ss=int(panel["max_ss"]),
                    min_fs=int(panel["min_fs"]),
                    max_fs=int(panel["max_fs"]),
                    data_path=data_path,
                    ndim=len(layout.dims),
                    event_axis=layout.event_axis,
                    ss_axis=layout.ss_axis,
                    fs_axis=layout.fs_axis,
                    fixed=dict(layout.fixed),
                )
            )
        except (KeyError, TypeError, ValueError):
            return None
    return placements


def assembled_frame_shape(placements: list[PanelPlacement]) -> tuple[int, int]:
    """Data-space ``(H, W)`` spanning every panel rectangle."""
    h = max(p.max_ss for p in placements) + 1
    w = max(p.max_fs for p in placements) + 1
    return (int(h), int(w))


def _panel_index(p: PanelPlacement, arr_ndim: int, event_i: int) -> tuple:
    has_event = p.event_axis is not None and arr_ndim == p.ndim
    idx: list = []
    for ax in range(p.ndim):
        if ax == p.event_axis:
            if has_event:
                idx.append(int(event_i))
            continue
        if ax in p.fixed:
            idx.append(int(p.fixed[ax]))
        elif ax in (p.ss_axis, p.fs_axis):
            idx.append(slice(None))
        else:
            idx.append(0)
    return tuple(idx)


def _slab_to_ss_fs(slab: np.ndarray, p: PanelPlacement) -> np.ndarray:
    if p.ss_axis is not None and p.fs_axis is not None and p.ss_axis > p.fs_axis:
        return slab.T
    return slab


def assemble_batch(
    f: h5py.File,
    lo: int,
    hi: int,
    placements: list[PanelPlacement],
    frame_shape: tuple[int, int],
) -> np.ndarray:
    dsets: dict[str, h5py.Dataset] = {}
    for p in placements:
        if p.data_path not in dsets:
            node = f[p.data_path]
            if not isinstance(node, h5py.Dataset):
                raise TypeError(f"{p.data_path!r} in {f.filename} is not a dataset")
            dsets[p.data_path] = node
    out_dtype = next(iter(dsets.values())).dtype
    h, w = int(frame_shape[0]), int(frame_shape[1])
    n = hi - lo
    batch = np.zeros((n, h, w), dtype=out_dtype)
    for j in range(n):
        for p in placements:
            dset = dsets[p.data_path]
            slab = _slab_to_ss_fs(
                np.asarray(dset[_panel_index(p, dset.ndim, lo + j)]), p
            )
            batch[j, p.min_ss : p.max_ss + 1, p.min_fs : p.max_fs + 1] = slab
    return batch


def _read_static_mask_slab(dset: h5py.Dataset, geometry: Geometry) -> np.ndarray:
    layout = geometry.data_layout
    if (
        layout is not None
        and layout.event_axis is not None
        and dset.ndim == len(layout.dims)
    ):
        idx: list = [slice(None)] * dset.ndim
        idx[layout.event_axis] = 0
        return np.asarray(dset[tuple(idx)])
    return np.asarray(dset[()])


def _apply_mask_bits(raw: np.ndarray, mask_good: int, mask_bad: int) -> np.ndarray:
    m = np.asarray(raw)
    if not np.issubdtype(m.dtype, np.unsignedinteger):
        m = m.astype(np.uint64)
    g = np.uint64(int(mask_good) & 0xFFFFFFFFFFFFFFFF)
    b = np.uint64(int(mask_bad) & 0xFFFFFFFFFFFFFFFF)
    return ((m & g) == g) & ((m & b) == np.uint64(0))


def read_mask(
    geometry: Geometry,
    data_filename: str,
    frame_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    spec = geometry.mask_spec
    if spec is None or spec.mask_path is None:
        return None
    fname = spec.mask_file or data_filename
    try:
        with h5py.File(fname, "r") as f:
            node = f.get(spec.mask_path)
            if not isinstance(node, h5py.Dataset):
                warnings.warn(
                    f"mask dataset {spec.mask_path!r} not in {fname!r}; skipping mask"
                )
                return None
            raw = _read_static_mask_slab(node, geometry)
    except OSError:
        warnings.warn(f"could not open mask file {fname!r}; skipping mask")
        return None

    good = _apply_mask_bits(raw, spec.mask_good, spec.mask_bad)
    fshape = (int(frame_shape[0]), int(frame_shape[1]))
    if good.shape == fshape:
        return good

    placements = build_placements(geometry)
    if placements is not None:
        try:
            assembled = np.ones(fshape, dtype=bool)
            for p in placements:
                slab = _slab_to_ss_fs(good[_panel_index(p, good.ndim, 0)], p)
                assembled[p.min_ss : p.max_ss + 1, p.min_fs : p.max_fs + 1] = slab
            return assembled
        except (IndexError, ValueError):
            pass
    warnings.warn(f"mask shape {good.shape} != frame {fshape}; skipping mask")
    return None


def _physical_bases(geometry: Geometry) -> Optional[list[tuple]]:
    # (min_ss, max_ss, min_fs, max_fs, corner_x, corner_y, fs_x, fs_y, ss_x, ss_y)
    # per panel; None unless >=2 panels all carry a usable corner + fs/ss basis.
    panels = geometry.panels or {}
    if len(panels) < 2:
        return None
    bases: list[tuple] = []
    for p in panels.values():
        fs = parse_axis_vector(p.get("fs"))
        ss = parse_axis_vector(p.get("ss"))
        if fs is None or ss is None:
            return None
        try:
            bases.append(
                (
                    int(p["min_ss"]),
                    int(p["max_ss"]),
                    int(p["min_fs"]),
                    int(p["max_fs"]),
                    float(p["corner_x"]),
                    float(p["corner_y"]),
                    fs[0],
                    fs[1],
                    ss[0],
                    ss[1],
                )
            )
        except (KeyError, TypeError, ValueError):
            return None
    return bases


def build_physical_assembler(geometry: Optional[Geometry]):
    """Build a raw-image -> physically-assembled-detector-image mapping.

    Tiled/rotated detectors (e.g. a CSPAD whose ASICs are stored in raw readout
    order) place each panel by its ``corner_x``/``corner_y`` + ``fs``/``ss``
    basis. This scatters each raw panel slab to its true physical position
    (nearest pixel), so a continuous powder ring renders as a continuous ring
    rather than fragments across panel boundaries.

    Returns
    -------
    tuple or None
        ``(assemble, beam_yx, out_shape)`` where ``assemble`` maps a raw
        ``(H, W)`` array to a physical ``(H', W')`` array (gaps ``NaN``) and
        ``beam_yx`` is the beam position in that output. ``None`` when the
        geometry is single-panel or lacks usable per-panel bases -- the raw image
        is already the physical image, so no assembly is needed.
    """
    if geometry is None:
        return None
    bases = _physical_bases(geometry)
    if bases is None:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for min_ss, max_ss, min_fs, max_fs, cx, cy, fsx, fsy, ssx, ssy in bases:
        fl, sl = max_fs - min_fs, max_ss - min_ss
        for fj in (0, fl):
            for si in (0, sl):
                xs.append(cx + fj * fsx + si * ssx)
                ys.append(cy + fj * fsy + si * ssy)
    x_min, x_max = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    y_min, y_max = int(np.floor(min(ys))), int(np.ceil(max(ys)))
    width, height = x_max - x_min + 1, y_max - y_min + 1
    # beam is at physical (0, 0); +y points up (origin="upper" -> small row index)
    beam_yx = (float(y_max), float(-x_min))

    def assemble(raw: np.ndarray) -> np.ndarray:
        img = np.asarray(raw, dtype=float)
        out = np.full((height, width), np.nan, dtype=float)
        for min_ss, max_ss, min_fs, max_fs, cx, cy, fsx, fsy, ssx, ssy in bases:
            ss_i = np.arange(max_ss - min_ss + 1)[:, None]
            fs_j = np.arange(max_fs - min_fs + 1)[None, :]
            X = np.rint(cx + fs_j * fsx + ss_i * ssx - x_min).astype(int)
            Y = np.rint(y_max - (cy + fs_j * fsy + ss_i * ssy)).astype(int)
            np.clip(X, 0, width - 1, out=X)
            np.clip(Y, 0, height - 1, out=Y)
            out[Y, X] = img[min_ss : max_ss + 1, min_fs : max_fs + 1]
        return out

    return assemble, beam_yx, (height, width)
