from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import h5py
import hdf5plugin  # noqa: F401  (registers bitshuffle)
import numpy as np

from .cbf import is_cbf

PathLike = Union[str, Path]

# CrystFEL fs/ss axis spec, e.g. "+0.003976x +0.999992y" (exponents allowed)
_AXIS_RE = re.compile(r"([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*([xy])")


def parse_axis_vector(spec: object) -> Optional[tuple[float, float]]:
    # z components are ignored; None when the spec has neither x nor y
    if not isinstance(spec, str):
        return None
    comps = {axis: float(val) for val, axis in _AXIS_RE.findall(spec)}
    if "x" not in comps and "y" not in comps:
        return None
    return (comps.get("x", 0.0), comps.get("y", 0.0))


EV_ANGSTROM = 12398.419843320026  # hc in eV Å
_CLEN_MM_THRESHOLD_M = 2.0  # an HDF5 clen above this is in mm

DETECTOR_KEYS = ("detector", "detector_type", "type")
FLAG_KEYS = ("flag_lessthan", "flag_morethan", "flag_equal")
PANEL_REQUIRED = {"min_fs", "max_fs", "min_ss", "max_ss", "corner_x", "corner_y"}
MASK_REQUIRED = {"min_fs", "max_fs", "min_ss", "max_ss"}
LAB_MASK_REQUIRED = {"min_x", "max_x", "min_y", "max_y"}
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
    # CrystFEL rule: good iff (v & mask_good) == mask_good and (v & mask_bad) == 0
    mask_file: Optional[str]
    mask_path: Optional[str]
    mask_good: int = 0
    mask_bad: int = 0


@dataclass
class BadRegion:
    name: str
    min_fs: Optional[int] = None
    max_fs: Optional[int] = None
    min_ss: Optional[int] = None
    max_ss: Optional[int] = None
    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None
    panel: Optional[str] = None

    @property
    def is_lab_frame(self) -> bool:
        return self.min_x is not None


@dataclass
class Geometry:
    """Parsed CrystFEL ``.geom`` file.

    Parameters
    ----------
    parameters : dict
        Top-level parameters, numbers coerced, for example ``clen`` and ``res``.
    panels : dict[str, dict]
        Panel definitions by name: every section with bounds and a corner.
    bad_regions : list[BadRegion]
        Masked regions, in data space (``min_fs`` ...) or the lab frame
        (``min_x`` ...).
    detector_type : str, optional
        Declared detector type.
    distance : float, optional
        Detector distance in m, ``clen`` plus the mean ``coffset``. ``None`` until
        an HDF5-path ``clen`` is read from the data.
    beam_center : tuple[float, float], optional
        Beam position ``(row, col)`` in data-space pixels: the panel mean of
        ``min_ss - corner_y`` and ``min_fs - corner_x``, which assumes
        ``fs = +x`` and ``ss = +y``.
    wavelength : float, optional
        Wavelength in Å, from ``wavelength`` or ``photon_energy``. ``None`` until
        an HDF5-path photon energy is read from the data.
    pixel_size : float, optional
        Pixel size in m, ``1 / res``.
    adu_per_photon : float, optional
        Gain in ADU per photon, from ``adu_per_photon`` or
        ``adu_per_eV * photon_energy`` when both are numbers.
    data_layout : DataLayout, optional
        The ``data`` path and the ``dimN`` axis roles.
    mask_spec : MaskSpec, optional
        Top-level ``mask``, ``mask_file``, ``mask_good`` and ``mask_bad``.
    panel_layouts, panel_masks : dict
        Per-panel overrides of ``data_layout`` and ``mask_spec``.
    panel_flags : dict, optional
        Per panel, the ``flag_lessthan``, ``flag_morethan`` and ``flag_equal``
        entries as ``(key, value)`` in file order, with integer values as in
        CrystFEL. A flag outside a panel section applies to the panels that first
        appear after it. ``None`` when the file has none.
    """

    parameters: dict
    panels: dict[str, dict]
    bad_regions: list[BadRegion] = field(default_factory=list)
    detector_type: Optional[str] = None
    distance: Optional[float] = None
    beam_center: Optional[tuple[float, float]] = None
    wavelength: Optional[float] = None
    pixel_size: Optional[float] = None
    adu_per_photon: Optional[float] = None
    data_layout: Optional["DataLayout"] = None
    mask_spec: Optional["MaskSpec"] = None
    panel_layouts: dict[str, "DataLayout"] = field(default_factory=dict)
    panel_masks: dict[str, "MaskSpec"] = field(default_factory=dict)
    panel_flags: Optional[dict[str, list[tuple[str, int]]]] = None

    def to_dict(self) -> dict:
        """Return the indexer geometry; ValueError if a required field is unset."""
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
            "adu_per_photon": self.adu_per_photon,
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
        Panels, bad regions, and the derived beam and optics fields. An HDF5-path
        ``clen`` or ``photon_energy`` stays unresolved until :class:`DataLoader`
        reads the first data file.

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
    flags: list[tuple[str, int]] = []
    panel_flags: dict[str, list[tuple[str, int]]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split(";", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            if "/" in key:
                name, sub = key.split("/", 1)
                per_name.setdefault(name, {})[sub] = _coerce(value)
                # A panel inherits the top-level flags that precede its first line
                found = panel_flags.setdefault(name, list(flags))
            else:
                sub = key
                parameters[key] = _coerce(value)
                found = flags
            if sub in FLAG_KEYS and (number := _flag_value(value)) is not None:
                found.append((sub, number))

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
                    panel=str(data["panel"]) if "panel" in data else None,
                )
            )
        elif LAB_MASK_REQUIRED.issubset(data.keys()):
            bad_regions.append(
                BadRegion(
                    name=name,
                    min_x=float(data["min_x"]),
                    max_x=float(data["max_x"]),
                    min_y=float(data["min_y"]),
                    max_y=float(data["max_y"]),
                    panel=str(data["panel"]) if "panel" in data else None,
                )
            )
        elif _looks_like_bad_region(name, data):
            warnings.warn(
                f"geometry: ignoring bad region {name!r} with keys "
                f"{sorted(data)} -- expected either {sorted(MASK_REQUIRED)} or "
                f"{sorted(LAB_MASK_REQUIRED)}",
                stacklevel=2,
            )

    distance = parameters.get("clen")
    if isinstance(distance, str):
        distance = None
    elif isinstance(distance, (int, float)) and not isinstance(distance, bool):
        # CrystFEL detector distance is clen + coffset
        distance = float(distance) + _mean_coffset(panels, parameters)

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
        adu_per_photon=_adu_per_photon(parameters),
        data_layout=data_layout,
        mask_spec=mask_spec,
        panel_layouts=panel_layouts,
        panel_masks=panel_masks,
        panel_flags={k: v for k, v in panel_flags.items() if v} or None,
    )


def _beam_center(panels: dict[str, dict]) -> Optional[tuple[float, float]]:
    # Data-space (row, col) of lab (0, 0), assuming fs = +x and ss = +y per panel
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
    # dimN roles: "%" the event axis, "ss"/"fs" the panel axes, an integer fixed
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


def _looks_like_bad_region(name: str, data: dict) -> bool:
    if name.lower().startswith("bad"):
        return True
    return any(k.startswith(("min_", "max_")) for k in data)


def _parse_mask_bits(value: object, default: int = 0) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return int(str(value).strip(), 0)  # base 0 accepts 0x... hex


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


def _with_unit(value, units: dict[str, float]) -> Optional[float]:
    # CrystFEL "<number> <unit>" values; a bare number is in the first unit
    if isinstance(value, (int, float)):
        return float(value)
    number, _, unit = str(value).partition(" ")
    try:
        return float(number) * units[unit.strip()]
    except (KeyError, ValueError):
        return None


def _wavelength(parameters: dict) -> Optional[float]:
    w = _with_unit(parameters.get("wavelength"), {"A": 1.0, "m": 1e10})
    if w is not None:
        return w
    e = _photon_energy(parameters)
    if e is not None and e > 0:
        return EV_ANGSTROM / e
    return None


def _photon_energy(parameters: dict) -> Optional[float]:
    return _with_unit(parameters.get("photon_energy"), {"eV": 1.0, "keV": 1e3})


def _pixel_size(parameters: dict) -> Optional[float]:
    res = parameters.get("res")  # pixels per metre
    if isinstance(res, (int, float)) and res > 0:
        return 1.0 / float(res)
    return None


def _adu_per_photon(parameters: dict) -> Optional[float]:
    g = parameters.get("adu_per_photon")
    if isinstance(g, (int, float)) and g > 0:
        return float(g)
    per_ev = parameters.get("adu_per_eV")
    e = _photon_energy(parameters)
    if isinstance(per_ev, (int, float)) and e is not None and per_ev > 0 and e > 0:
        return float(per_ev) * e
    return None


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _flag_value(value: str) -> Optional[int]:
    try:
        number = float(value)
    except ValueError:
        try:
            number = int(value, 0)
        except ValueError:
            return None
    return int(number) if np.isfinite(number) else None


def resolve_dynamic_fields(geometry: Geometry, data_file: PathLike) -> Geometry:
    # Fill in HDF5-path-valued clen/photon_energy from the data file
    clen_path = geometry.parameters.get("clen")
    pe_path = geometry.parameters.get("photon_energy")
    need_clen = geometry.distance is None and isinstance(clen_path, str)
    need_wl = geometry.wavelength is None and isinstance(pe_path, str)
    if not (need_clen or need_wl):
        return geometry
    if is_cbf(str(data_file)):
        raise ValueError(
            f"geometry reads clen/photon_energy ({clen_path!r}, {pe_path!r}) from "
            f"the data file, but {str(data_file)!r} is a CBF file; give them as numbers"
        )
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


def match_events(f: h5py.File, pattern: str) -> list[str]:
    # Natural-sorted names n under the "%" parent that hold a dataset at n + rest
    parent, _, rest = pattern.partition("%")
    if "%" in rest:
        raise ValueError(f"more than one '%' in {pattern!r} is not supported")
    group = f.get(parent or "/")
    if not isinstance(group, h5py.Group):
        return []
    names = [n for n in group if isinstance(group.get(n + rest), h5py.Dataset)]
    return sorted(
        names,
        key=lambda n: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)],
    )


def _dataset_mean(f: h5py.File, path: str) -> Optional[float]:
    # A per-event ("%") path is averaged over every event
    paths = [path]
    if "%" in path:
        paths = [path.replace("%", n) for n in match_events(f, path)]
    nodes = [f.get(p) for p in paths]
    if not nodes or not all(isinstance(node, h5py.Dataset) for node in nodes):
        warnings.warn(
            f"geometry path {path!r} is not a dataset in {f.filename!r}; leaving unset"
        )
        return None
    values = np.concatenate(
        [np.asarray(node[()], dtype=np.float32).ravel() for node in nodes]
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        warnings.warn(
            f"geometry path {path!r} in {f.filename!r} has no finite values; leaving unset"
        )
        return None
    return float(finite.mean())


def _mean_coffset(panels: dict, parameters: dict) -> float:
    values = [float(p["coffset"]) for p in panels.values() if "coffset" in p]
    top = parameters.get("coffset")
    if not values and isinstance(top, (int, float)):
        return float(top)
    return sum(values) / len(values) if values else 0.0


def _resolve_clen(f: h5py.File, path: str, geometry: Geometry) -> Optional[float]:
    raw = _dataset_mean(f, path)
    if raw is None:
        return None
    metres = raw * 1e-3 if abs(raw) > _CLEN_MM_THRESHOLD_M else raw
    return metres + _mean_coffset(geometry.panels, geometry.parameters)


def _resolve_wavelength(f: h5py.File, path: str) -> Optional[float]:
    ev = _dataset_mean(f, path)
    if ev is None or ev <= 0:
        return None
    return EV_ANGSTROM / ev
