from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

EV_ANGSTROM = 12398.419843320026

DETECTOR_KEYS = ("detector", "detector_type", "type")
PANEL_REQUIRED = {"min_fs", "max_fs", "min_ss", "max_ss", "corner_x", "corner_y"}
MASK_REQUIRED = {"min_fs", "max_fs", "min_ss", "max_ss"}


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
    with path.open("r") as fh:
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
    )


def _beam_center(panels: dict[str, dict]) -> Optional[tuple[float, float]]:
    if len(panels) != 1:
        return None
    p = next(iter(panels.values()))
    return (-float(p["corner_y"]), -float(p["corner_x"]))


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
