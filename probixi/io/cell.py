from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


@dataclass
class CellParams:
    """Parsed unit cell from a CrystFEL `.cell`` file.

    Parameters
    ----------
    a, b, c : float
        Cell edge lengths (same unit as the file, typically nm/A).
    alpha, beta, gamma : float
        Inter-axial angles in radians (alpha between b/c, beta between a/c,
        gamma between a/b).
    lattice_type : str, optional
        Bravais lattice type (e.g. ``"triclinic"``).
    unique_axis : str, optional
        Unique axis label (e.g. ``"a"``/``"b"``/``"c"``).
    centering : str, optional
        Lattice centering symbol (e.g. ``"P"``, ``"C"``, ``"F"``).
    """

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    lattice_type: Optional[str] = None
    unique_axis: Optional[str] = None
    centering: Optional[str] = None

    @property
    def volume(self) -> float:
        # Triclinic Volume = abc*sqrt(1 - cos^2a - cos^2b - cos^2g + 2 cosa cosb cosg)
        ca = math.cos(self.alpha)
        cb = math.cos(self.beta)
        cg = math.cos(self.gamma)
        return (
            self.a
            * self.b
            * self.c
            * math.sqrt(
                max(1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg, 0.0)
            )
        )

    def as_dict_degrees(self) -> dict:
        d = {
            "a_A": self.a,
            "b_A": self.b,
            "c_A": self.c,
            "alpha_deg": math.degrees(self.alpha),
            "beta_deg": math.degrees(self.beta),
            "gamma_deg": math.degrees(self.gamma),
            "volume_A3": self.volume,
        }
        for k in ("lattice_type", "unique_axis", "centering"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


def read_crystfel_cell(path: PathLike) -> CellParams:
    """Read a CrystFEL ``.cell`` file into :class:`CellParams`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.cell`` file.

    Returns
    -------
    CellParams
        Parsed cell with angles converted to radians.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If any of ``a``/``b``/``c``/``alpha``/``beta``/``gamma`` is absent.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cell file not found: {path}")

    aliases = {
        "al": "alpha",
        "be": "beta",
        "ga": "gamma",
        "alpha": "alpha",
        "beta": "beta",
        "gamma": "gamma",
    }
    values: dict[str, float] = {}
    meta: dict[str, str] = {}
    with path.open("r") as fh:
        for line in fh:
            line = line.split(";", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = (s.strip() for s in line.split("=", 1))
            tokens = value.split()
            if not tokens:
                continue
            if key in ("a", "b", "c"):
                try:
                    values[key] = float(tokens[0])
                except ValueError:
                    pass
            elif key in aliases:
                try:
                    values[aliases[key]] = math.radians(float(tokens[0]))
                except ValueError:
                    pass
            elif key in ("lattice_type", "unique_axis", "centering"):
                meta[key] = tokens[0]

    missing = {"a", "b", "c", "alpha", "beta", "gamma"} - values.keys()
    if missing:
        raise ValueError(f"cell file missing keys: {sorted(missing)}")
    return CellParams(
        a=values["a"],
        b=values["b"],
        c=values["c"],
        alpha=values["alpha"],
        beta=values["beta"],
        gamma=values["gamma"],
        lattice_type=meta.get("lattice_type"),
        unique_axis=meta.get("unique_axis"),
        centering=meta.get("centering"),
    )
