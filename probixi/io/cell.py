from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional, Union

PathLike = Union[str, Path]


@dataclass
class CellParams:
    """Unit cell from a CrystFEL ``.cell`` file.

    Parameters
    ----------
    a, b, c : float
        Edge lengths in Å.
    alpha, beta, gamma : float
        Angles in radians: alpha between b and c, beta between a and c, gamma
        between a and b.
    lattice_type : str, optional
        Lattice system, for example ``"monoclinic"``.
    unique_axis : str, optional
        ``"a"``, ``"b"`` or ``"c"``.
    centering : str, optional
        Centring symbol, for example ``"P"``, ``"C"`` or ``"F"``.
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
        # General triclinic volume; the clamp absorbs rounding at degenerate angles
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
        Edges as written, read as Å (the unit token is ignored), and angles in
        radians.

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
    with path.open("r", encoding="utf-8") as fh:
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


def median_cell(cells, template: Optional[CellParams] = None) -> CellParams:
    # Component-wise median; symmetry labels come from template, else the first cell
    cells = list(cells)
    if not cells:
        raise ValueError("median_cell needs at least one cell")
    meta = template if template is not None else cells[0]
    return CellParams(
        a=median(c.a for c in cells),
        b=median(c.b for c in cells),
        c=median(c.c for c in cells),
        alpha=median(c.alpha for c in cells),
        beta=median(c.beta for c in cells),
        gamma=median(c.gamma for c in cells),
        lattice_type=meta.lattice_type,
        unique_axis=meta.unique_axis,
        centering=meta.centering,
    )
