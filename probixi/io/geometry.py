from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import h5py
import hdf5plugin  # noqa: F401  (registers bitshuffle)
import numpy as np

PathLike = Union[str, Path]

# CrystFEL fs/ss axis spec, e.g. "+0.003976x +0.999992y" (exponents allowed).
_AXIS_RE = re.compile(r"([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*([xy])")


def parse_axis_vector(spec: object) -> Optional[tuple[float, float]]:
    """Parse a CrystFEL ``fs``/``ss`` axis spec into an ``(x, y)`` tuple.

    ``"+0.003976x +0.999992y"`` -> ``(0.003976, 0.999992)``. Returns ``None`` for
    anything without an ``x`` or ``y`` component (so callers can fall back).
    """
    if not isinstance(spec, str):
        return None
    comps = {axis: float(val) for val, axis in _AXIS_RE.findall(spec)}
    if "x" not in comps and "y" not in comps:
        return None
    return (comps.get("x", 0.0), comps.get("y", 0.0))


EV_ANGSTROM = 12398.419843320026
_CLEN_MM_THRESHOLD_M = 2.0

DETECTOR_KEYS = ("detector", "detector_type", "type")
PANEL_REQUIRED = {"min_fs", "max_fs", "min_ss", "max_ss", "corner_x", "corner_y"}
MASK_REQUIRED = {"min_fs", "max_fs", "min_ss", "max_ss"}
_MASK_KEYS = ("mask", "mask_file", "mask_good", "mask_bad")


@dataclass
class DataLayout:
    data_path: Optional[str]
    dims: list
    event_axis: Optional[int] = None
    ss_axis: Optional[int] = None
    fs_axis: Optional[int] = None
    fixed: dict = field(default_factory=dict)


@dataclass
class MaskSpec:
    """CrystFEL bad-pixel mask reference (``mask``/``mask_file`` + bit flags).

    A pixel is GOOD iff ``(value & mask_good) == mask_good`` and
    ``(value & mask_bad) == 0`` -- all ``mask_good`` bits set and no ``mask_bad``
    bit set.

    Parameters
    ----------
    mask_file : str, optional
        External HDF5 file holding the mask; None means the data file itself.
    mask_path : str, optional
        Internal HDF5 path to the mask array.
    mask_good : int
        Bits that must all be set for a pixel to be good.
    mask_bad : int
        Bits whose presence marks a pixel bad.
    """

    mask_file: Optional[str]
    mask_path: Optional[str]
    mask_good: int = 0
    mask_bad: int = 0


@dataclass
class BadRegion:
    name: str
    min_fs: int
    max_fs: int
    min_ss: int
    max_ss: int


@dataclass
class Geometry:
    """Parsed CrystFEL ``.geom`` file.

    Parameters
    ----------
    parameters : dict
        Top-level scalar/string parameters (e.g. ``clen``, ``res``).
    panels : dict[str, dict]
        Per-panel definitions, keyed by panel name.
    bad_regions : list[BadRegion]
        Masked-out detector regions.
    detector_type : str, optional
        Detector type, if declared.
    distance : float, optional
        Detector/Camera length ``clen`` in meters, or None if given as an HDF5 path.
    beam_center : tuple[float, float], optional
        Beam center ``(row, col)``, only when a single panel is defined.
    wavelength : float, optional
        Wavelength in angstroms.
    pixel_size : float, optional
        Pixel size in meters.
    """

    parameters: dict
    panels: dict[str, dict]
    bad_regions: list[BadRegion] = field(default_factory=list)
    detector_type: Optional[str] = None
    distance: Optional[float] = None
    beam_center: Optional[tuple[float, float]] = None
    wavelength: Optional[float] = None
    pixel_size: Optional[float] = None
    data_layout: Optional["DataLayout"] = None
    mask_spec: Optional["MaskSpec"] = None
    panel_layouts: dict[str, "DataLayout"] = field(default_factory=dict)
    panel_masks: dict[str, "MaskSpec"] = field(default_factory=dict)

    def to_dict(self) -> dict:
        missing = [
            name
            for name, value in (
                ("beam_center", self.beam_center),
                ("clen", self.distance),
                ("pixel_size", self.pixel_size),
                ("wavelength", self.wavelength),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"Geometry missing required fields: {missing}")
        return {
            "beam_center": self.beam_center,
            "clen": self.distance,
            "pixel_size": self.pixel_size,
            "wavelength": self.wavelength,
            "panels": self.panels,
        }


def read_geometry(path: PathLike) -> Geometry:
    """Read a CrystFEL ``.geom`` file into a :class:`Geometry`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.geom`` file.

    Returns
    -------
    Geometry
        Parsed geometry with panels, bad regions, and derived beam/optics fields.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Geometry file not found: {path}")

    parameters: dict = {}
    per_name: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split(";", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            if "/" in key:
                name, sub = key.split("/", 1)
                per_name.setdefault(name, {})[sub] = _coerce(value)
            else:
                parameters[key] = _coerce(value)

    panels: dict[str, dict] = {}
    bad_regions: list[BadRegion] = []
    for name, data in per_name.items():
        if PANEL_REQUIRED.issubset(data.keys()):
            panels[name] = data
        elif MASK_REQUIRED.issubset(data.keys()):
            bad_regions.append(
                BadRegion(
                    name=name,
                    min_fs=int(data["min_fs"]),
                    max_fs=int(data["max_fs"]),
                    min_ss=int(data["min_ss"]),
                    max_ss=int(data["max_ss"]),
                )
            )

    distance = parameters.get("clen")
    if isinstance(distance, str):
        distance = None

    dim_map = _collect_dims(parameters)
    data_layout = _build_data_layout(parameters.get("data"), dim_map)
    mask_spec = _build_mask_spec(parameters)
    panel_layouts, panel_masks = _panel_overrides(per_name, panels, parameters, dim_map)

    return Geometry(
        parameters=parameters,
        panels=panels,
        bad_regions=bad_regions,
        detector_type=next(
            (parameters[k] for k in DETECTOR_KEYS if k in parameters),
            None,
        ),
        distance=distance,
        beam_center=_beam_center(panels),
        wavelength=_wavelength(parameters),
        pixel_size=_pixel_size(parameters),
        data_layout=data_layout,
        mask_spec=mask_spec,
        panel_layouts=panel_layouts,
        panel_masks=panel_masks,
    )


def _beam_center(panels: dict[str, dict]) -> Optional[tuple[float, float]]:
    rows: list[float] = []
    cols: list[float] = []
    for p in panels.values():
        try:
            rows.append(float(p.get("min_ss", 0.0)) - float(p["corner_y"]))
            cols.append(float(p.get("min_fs", 0.0)) - float(p["corner_x"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return None
    return (sum(rows) / len(rows), sum(cols) / len(cols))


def _collect_dims(d: dict) -> dict[int, object]:
    out: dict[int, object] = {}
    for key, value in d.items():
        if isinstance(key, str) and key.startswith("dim") and key[3:].isdigit():
            out[int(key[3:])] = value
    return out


def _build_data_layout(
    data_path: Optional[object], dim_map: dict[int, object]
) -> Optional[DataLayout]:
    if data_path is None and not dim_map:
        return None
    ndim = (max(dim_map) + 1) if dim_map else 0
    dims: list = [dim_map.get(i) for i in range(ndim)]
    event_axis = ss_axis = fs_axis = None
    fixed: dict[int, int] = {}
    for i, value in enumerate(dims):
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            fixed[i] = int(value)
        elif value == "%":
            event_axis = i
        elif value == "ss":
            ss_axis = i
        elif value == "fs":
            fs_axis = i
    return DataLayout(
        data_path=str(data_path) if data_path is not None else None,
        dims=dims,
        event_axis=event_axis,
        ss_axis=ss_axis,
        fs_axis=fs_axis,
        fixed=fixed,
    )


def _parse_mask_bits(value: object, default: int = 0) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return int(str(value).strip(), 0)


def _build_mask_spec(d: dict) -> Optional[MaskSpec]:
    if not any(k in d for k in _MASK_KEYS):
        return None
    mask_path = d.get("mask")
    mask_file = d.get("mask_file")
    return MaskSpec(
        mask_file=str(mask_file) if mask_file is not None else None,
        mask_path=str(mask_path) if mask_path is not None else None,
        mask_good=_parse_mask_bits(d.get("mask_good"), 0),
        mask_bad=_parse_mask_bits(d.get("mask_bad"), 0),
    )


def _panel_overrides(
    per_name: dict[str, dict],
    panels: dict[str, dict],
    parameters: dict,
    dim_map: dict[int, object],
) -> tuple[dict[str, DataLayout], dict[str, MaskSpec]]:
    layouts: dict[str, DataLayout] = {}
    masks: dict[str, MaskSpec] = {}
    for name in panels:
        pdict = per_name.get(name, {})
        pdims = _collect_dims(pdict)
        if "data" in pdict or pdims:
            layout = _build_data_layout(
                pdict.get("data", parameters.get("data")), {**dim_map, **pdims}
            )
            if layout is not None:
                layouts[name] = layout
        if any(k in pdict for k in _MASK_KEYS):
            merged = {k: pdict.get(k, parameters.get(k)) for k in _MASK_KEYS}
            spec = _build_mask_spec(merged)
            if spec is not None:
                masks[name] = spec
    return layouts, masks


def _wavelength(parameters: dict) -> Optional[float]:
    w = parameters.get("wavelength")
    if isinstance(w, (int, float)):
        return float(w)
    e = parameters.get("photon_energy")
    if isinstance(e, (int, float)) and e > 0:
        return EV_ANGSTROM / float(e)
    return None


def _pixel_size(parameters: dict) -> Optional[float]:
    res = parameters.get("res")
    if isinstance(res, (int, float)) and res > 0:
        return 1.0 / float(res)
    return None


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def resolve_dynamic_fields(geometry: Geometry, data_file: PathLike) -> Geometry:
    """Fill in HDF5-path-valued ``clen``/``photon_energy`` from the data file."""
    clen_path = geometry.parameters.get("clen")
    pe_path = geometry.parameters.get("photon_energy")
    need_clen = geometry.distance is None and isinstance(clen_path, str)
    need_wl = geometry.wavelength is None and isinstance(pe_path, str)
    if not (need_clen or need_wl):
        return geometry
    try:
        with h5py.File(data_file, "r") as f:
            if need_clen:
                geometry.distance = _resolve_clen(f, str(clen_path), geometry)
            if need_wl:
                geometry.wavelength = _resolve_wavelength(f, str(pe_path))
    except OSError:
        warnings.warn(
            f"could not open {data_file!r} to resolve geometry paths "
            f"({clen_path!r}, {pe_path!r})"
        )
    return geometry


def _dataset_mean(f: h5py.File, path: str) -> Optional[float]:
    node = f.get(path)
    if not isinstance(node, h5py.Dataset):
        warnings.warn(
            f"geometry path {path!r} is not a dataset in {f.filename!r}; leaving unset"
        )
        return None
    values = np.asarray(node[()], dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        warnings.warn(
            f"geometry path {path!r} in {f.filename!r} has no finite values; leaving unset"
        )
        return None
    return float(finite.mean())


def _mean_coffset(geometry: Geometry) -> float:
    values = [float(p["coffset"]) for p in geometry.panels.values() if "coffset" in p]
    top = geometry.parameters.get("coffset")
    if not values and isinstance(top, (int, float)):
        return float(top)
    return sum(values) / len(values) if values else 0.0


def _resolve_clen(f: h5py.File, path: str, geometry: Geometry) -> Optional[float]:
    raw = _dataset_mean(f, path)
    if raw is None:
        return None
    metres = raw * 1e-3 if abs(raw) > _CLEN_MM_THRESHOLD_M else raw
    return metres + _mean_coffset(geometry)


def _resolve_wavelength(f: h5py.File, path: str) -> Optional[float]:
    ev = _dataset_mean(f, path)
    if ev is None or ev <= 0:
        return None
    return EV_ANGSTROM / ev
