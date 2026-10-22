from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import duckdb
import numpy as np
import torch

from ..indexer.lattice import B_to_cell
from .geometry import EV_ANGSTROM
from .writer import (
    A_INV_TO_NM_INV,
    _panel_bounds,
    _profile_radius,
    _reflection_flags,
    _StreamWriter,
)

if TYPE_CHECKING:
    from ..indexer.indexer import FrameIndexResult, IndexResult
    from .cell import CellParams

PathLike = Union[str, Path]
_DB_SUFFIXES = {".duckdb", ".db"}
_FLUSH_ROWS = 50_000


def is_duckdb_path(path: Optional[PathLike]) -> bool:
    """Whether ``path`` has a ``.duckdb`` or ``.db`` suffix."""
    return path is not None and Path(path).suffix.lower() in _DB_SUFFIXES


_FRAME_COLUMNS = (
    "frame_id",
    "frame_index",
    "filename",
    "event",
    "indexed",
    "serial",
    "n_peaks",
    "peak_resolution_nm_inv",
    "scale",
    "scale_sigma",
    "num_reflections",
    "adu_per_photon",
    "event_name",
    "photon_energy_eV",
    "bg_low",
    "bg_mid",
    "bg_ring",
    "bg_high",
    "bg_edge",
    "ice_score",
    "peak_falloff",
    "peak_size_px",
    "peak_ecc",
)
_CRYSTAL_COLUMNS = (
    "crystal_id",
    "frame_id",
    "lattice_index",
    "n_indexed",
    "rmsd",
    "mosaicity_deg",
    "profile_radius_nm_inv",
    "enrichment",
    "n_bright",
    "enrich_p",
    "diffraction_limit_nm_inv",
    "num_reflections",
    "cell_a_A",
    "cell_b_A",
    "cell_c_A",
    "cell_alpha_deg",
    "cell_beta_deg",
    "cell_gamma_deg",
    "astar_x",
    "astar_y",
    "astar_z",
    "bstar_x",
    "bstar_y",
    "bstar_z",
    "cstar_x",
    "cstar_y",
    "cstar_z",
    "matched_frac",
    "matched_posterior",
)

_REFLECTION_COLUMNS = (
    "crystal_id",
    "frame_id",
    "h",
    "k",
    "l",
    "intensity",
    "sigma",
    "peak",
    "background",
    "fs",
    "ss",
    "panel",
    "resolution_nm_inv",
    "n_pixels",
    "background_model",
    "background_model_var",
    "n_pixels_total",
    "n_masked",
    "n_overlap",
    "pred_fs",
    "pred_ss",
    "peak_posterior",
    "edge_px",
    "peak_max",
)

_PANEL_COLUMNS = ("name", "min_fs", "max_fs", "min_ss", "max_ss")

_PEAK_COLUMNS = (
    "frame_id",
    "fs",
    "ss",
    "intensity",
    "resolution_nm_inv",
    "panel",
    "n_pixels",
    "photons",
    "background_photons",
    "n_masked",
    "posterior",
    "intensity_max",
    "eccentricity",
)

_SCHEMA = """
CREATE TABLE geometry (
    beam_center_row  DOUBLE,
    beam_center_col  DOUBLE,
    clen             DOUBLE,
    pixel_size       DOUBLE,
    wavelength       DOUBLE,
    photon_energy_eV DOUBLE,
    adu_per_photon   DOUBLE,
    n_panels         INTEGER,
    geometry_file    VARCHAR
);

CREATE TABLE panels (
    name   VARCHAR,
    min_fs INTEGER,
    max_fs INTEGER,
    min_ss INTEGER,
    max_ss INTEGER
);

CREATE TABLE cell (
    a_A          DOUBLE,
    b_A          DOUBLE,
    c_A          DOUBLE,
    alpha_deg    DOUBLE,
    beta_deg     DOUBLE,
    gamma_deg    DOUBLE,
    volume_A3    DOUBLE,
    lattice_type VARCHAR,
    centering    VARCHAR,
    unique_axis  VARCHAR
);

CREATE TABLE integration (
    r_peak            DOUBLE,
    r_gap             DOUBLE,
    r_bg              DOUBLE,
    adu_per_photon    DOUBLE,
    bg_annulus_pixels DOUBLE,
    aperture          VARCHAR
);

CREATE TABLE frames (
    frame_id                 VARCHAR PRIMARY KEY,
    frame_index              BIGINT,
    filename                 VARCHAR,
    event                    BIGINT,
    indexed                  BOOLEAN,
    serial                   BIGINT,
    n_peaks                  INTEGER,
    peak_resolution_nm_inv   DOUBLE,
    scale                    DOUBLE,
    scale_sigma              DOUBLE,
    num_reflections          INTEGER,
    adu_per_photon           DOUBLE,
    event_name               VARCHAR,
    photon_energy_eV         DOUBLE,
    bg_low                   DOUBLE,
    bg_mid                   DOUBLE,
    bg_ring                  DOUBLE,
    bg_high                  DOUBLE,
    bg_edge                  DOUBLE,
    ice_score                DOUBLE,
    peak_falloff             DOUBLE,
    peak_size_px             DOUBLE,
    peak_ecc                 DOUBLE
);

CREATE TABLE crystals (
    crystal_id               VARCHAR PRIMARY KEY,
    frame_id                 VARCHAR,
    lattice_index            INTEGER,
    n_indexed                INTEGER,
    rmsd                     DOUBLE,
    mosaicity_deg            DOUBLE,
    profile_radius_nm_inv    DOUBLE,
    enrichment               DOUBLE,
    n_bright                 INTEGER,
    enrich_p                 DOUBLE,
    diffraction_limit_nm_inv DOUBLE,
    num_reflections          INTEGER,
    cell_a_A                 DOUBLE,
    cell_b_A                 DOUBLE,
    cell_c_A                 DOUBLE,
    cell_alpha_deg           DOUBLE,
    cell_beta_deg            DOUBLE,
    cell_gamma_deg           DOUBLE,
    astar_x                  DOUBLE,
    astar_y                  DOUBLE,
    astar_z                  DOUBLE,
    bstar_x                  DOUBLE,
    bstar_y                  DOUBLE,
    bstar_z                  DOUBLE,
    cstar_x                  DOUBLE,
    cstar_y                  DOUBLE,
    cstar_z                  DOUBLE,
    matched_frac             DOUBLE,
    matched_posterior        DOUBLE
);

CREATE TABLE reflections (
    crystal_id        VARCHAR,
    frame_id          VARCHAR,
    h                 INTEGER,
    k                 INTEGER,
    l                 INTEGER,
    intensity         DOUBLE,
    sigma             DOUBLE,
    peak              DOUBLE,
    background        DOUBLE,
    fs                DOUBLE,
    ss                DOUBLE,
    panel             VARCHAR,
    resolution_nm_inv DOUBLE,
    n_pixels          INTEGER,
    background_model  DOUBLE,
    background_model_var DOUBLE,
    n_pixels_total    INTEGER,
    n_masked          INTEGER,
    n_overlap         INTEGER,
    pred_fs           DOUBLE,
    pred_ss           DOUBLE,
    peak_posterior    DOUBLE,
    edge_px           DOUBLE,
    peak_max          DOUBLE
);

CREATE TABLE peaks (
    frame_id           VARCHAR,
    fs                 DOUBLE,
    ss                 DOUBLE,
    intensity          DOUBLE,
    resolution_nm_inv  DOUBLE,
    panel              VARCHAR,
    n_pixels           DOUBLE,
    photons            DOUBLE,
    background_photons DOUBLE,
    n_masked           INTEGER,
    posterior          DOUBLE,
    intensity_max      DOUBLE,
    eccentricity       DOUBLE
);
"""

_INDEXES = """
CREATE INDEX idx_reflections_frame ON reflections(frame_id);
CREATE INDEX idx_reflections_crystal ON reflections(crystal_id);
CREATE INDEX idx_crystals_frame ON crystals(frame_id);
CREATE INDEX idx_peaks_frame ON peaks(frame_id);
"""


def frame_id(filename: str, event: int) -> str:
    """Return the database key of one file-event.

    Parameters
    ----------
    filename : str
        Source file, as recorded in ``frames.filename``.
    event : int
        Event index within the file.

    Returns
    -------
    str
        First 16 hex digits of the SHA-1 of ``"{filename}//{event}"``.

    Notes
    -----
    The key depends only on its inputs, so a peaks-only pass, an indexed pass and
    multi-GPU shards give the same frame the same ``frame_id``.
    """
    digest = hashlib.sha1(f"{filename}//{event}".encode("utf-8"))
    return digest.hexdigest()[:16]


class DuckDBOffloader(_StreamWriter):
    """Write index results to a DuckDB database.

    Run metadata goes into the ``geometry``, ``panels``, ``cell`` and
    ``integration`` tables. Every file-event becomes a row of ``frames`` and every
    accepted lattice a row of ``crystals``. Searched ``peaks`` are keyed by
    :func:`frame_id`, and integrated ``reflections`` by ``frame_id`` and
    ``crystal_id``. The frames that never indexed are added, and the indexes built,
    when the context exits.

    Parameters
    ----------
    path : str or Path
        Destination ``.duckdb`` or ``.db`` file, replaced if it exists.
    geometry : dict
        Indexer geometry: ``beam_center``, ``clen``, ``pixel_size``,
        ``wavelength``, ``adu_per_photon`` and, when available, ``panels``.
    cell : CellParams, optional
        Target cell, written to the ``cell`` table.
    geometry_file : str or Path, optional
        Geometry file stored verbatim in ``geometry.geometry_file``.
    files : dict, optional
        Loader file map. It resolves frame indices to file and event, and lists
        the frames that never indexed, which get ``indexed = FALSE``. Without it,
        ``frames`` holds only the frames passed to :meth:`write`.
    frame_range : tuple of int, optional
        Half-open range ``[lo, hi)`` of global frame indices this writer owns.
        Only frames in it receive unindexed rows, as ``--start``/``--stop`` and
        multi-GPU blocks require. ``None`` covers every file-event.
    indexer_name : str, default "probixi"
        Accepted for parity with :class:`DataOffloader`; not stored.
    panel : str, default "0"
        Panel name for peaks and reflections outside every geometry panel.
    integration : dict, optional
        Integration recipe (``radii``, ``adu_per_photon``, ``bg_annulus_pixels``,
        ``aperture``), as :attr:`~probixi.Probixi.integration_recipe` returns it.

    Notes
    -----
    Use as a context manager, as for :class:`DataOffloader`, or through
    :meth:`~probixi.indexer.IndexStream.to_db`::

        with DuckDBOffloader(out, geometry=geom, cell=cell, files=files) as off:
            stream.to_stream(off)
    """

    def __init__(
        self,
        path: PathLike,
        geometry: dict,
        *,
        cell: Optional["CellParams"] = None,
        geometry_file: Optional[PathLike] = None,
        files: Optional[dict] = None,
        frame_range: Optional[tuple[int, int]] = None,
        indexer_name: str = "probixi",
        panel: str = "0",
        integration: Optional[dict] = None,
    ):
        super().__init__(
            path,
            geometry,
            geometry_file=geometry_file,
            files=files,
            indexer_name=indexer_name,
            panel=panel,
        )
        self.cell = cell
        self.integration = integration
        self._crystal_rows: list[tuple] = []
        self._frame_range = frame_range
        self._conn = None
        self._frame_rows: list[tuple] = []
        self._refl_chunks: list[tuple] = []
        self._peak_chunks: list[tuple] = []
        self._n_refl = 0
        self._n_peak = 0
        self._seen: set[int] = set()
        self._recip_scale: dict[str, float] = {}  # lambda0 / frame lambda, by frame id

    def __enter__(self) -> "DuckDBOffloader":
        if self.path.exists():
            self.path.unlink()
        self._conn = duckdb.connect(str(self.path))
        self._conn.execute("BEGIN TRANSACTION")
        self._conn.execute(_SCHEMA)
        self._write_metadata_tables()
        self._conn.execute("COMMIT")
        return self

    def __exit__(self, *exc) -> None:
        if self._conn is None:
            return
        try:
            self._emit_unindexed()
            self._flush()
            self._conn.execute(_INDEXES)
        finally:
            self._conn.close()
            self._conn = None

    def write(self, result: Union["IndexResult", "FrameIndexResult"]) -> None:
        """Buffer one frame and its lattices.

        Parameters
        ----------
        result : IndexResult or FrameIndexResult
            One lattice, or every lattice found on one frame.
        """
        if self._conn is None:
            raise RuntimeError("DuckDBOffloader must be used as a context manager")
        crystals = getattr(result, "crystals", [result])
        self._serial += 1
        filename, event = self._locate(result.frame_index)
        fid = frame_id(filename, event)
        self._note_wavelength(fid, filename, event)
        peaks = self._peaks(result)
        total = 0
        refl_chunks, crystal_rows = [], []
        for lattice_index, crystal in enumerate(crystals):
            cid = f"{fid}:{lattice_index}"
            refl = self._reflections(crystal)
            n = refl[0].shape[1]
            total += n
            refl_chunks.append((cid, fid, *refl))
            crystal_rows.append(self._crystal_row(crystal, cid, fid, lattice_index, n))
        frame = self._frame_row(
            fid,
            filename,
            event,
            result.frame_index,
            result.n_peaks,
            indexed=bool(crystals),
            num_reflections=total if crystals else None,
            scale=crystals[0].scale if crystals else None,
            scale_sigma=crystals[0].scale_sigma if crystals else None,
            gain=self._gain(crystals[0] if crystals else None),
            frame_stats=getattr(result, "frame_stats", None),
        )
        self._peak_chunks.append((fid, self._gain(result), peaks))
        self._refl_chunks += refl_chunks
        self._crystal_rows += crystal_rows
        self._frame_rows.append(frame)
        self._n_peak += peaks.shape[1]
        self._n_refl += total
        if result.frame_index is not None:
            self._seen.add(int(result.frame_index))
        self._maybe_flush()

    def write_peaks(self, result) -> None:
        """Buffer one frame's peak search for a peaks-only run.

        The frame gets a ``frames`` row with ``indexed = FALSE`` and its peaks go to
        ``peaks``; ``crystals`` and ``reflections`` stay empty.

        Parameters
        ----------
        result : PeakResult
            One item of :meth:`~probixi.Probixi.peak_stream`.
        """
        if self._conn is None:
            raise RuntimeError("DuckDBOffloader must be used as a context manager")
        self._serial += 1
        filename, event = self._locate(result.frame_index)
        fid = frame_id(filename, event)
        self._note_wavelength(fid, filename, event)

        stats = result.kept_stats
        peaks = _to_array(
            torch.stack(
                [
                    stats.row_centroid,
                    stats.col_centroid,
                    stats.intensity_sum,
                    stats.background_sum,
                    stats.size.to(stats.intensity_sum.dtype),
                ]
            )
        )
        peaks = np.vstack([peaks, _nan_array(stats.n_masked, peaks.shape[1])])
        peaks = np.vstack([peaks, _nan_array(stats.posterior_mean, peaks.shape[1])])
        peaks = np.vstack([peaks, _nan_array(stats.intensity_max, peaks.shape[1])])
        peaks = np.vstack([peaks, _nan_array(stats.eccentricity, peaks.shape[1])])
        gain = self._gain()
        frame = self._frame_row(
            fid,
            filename,
            event,
            result.frame_index,
            peaks.shape[1],
            indexed=False,
            gain=gain,
        )
        self._peak_chunks.append((fid, gain, peaks))
        self._n_peak += peaks.shape[1]
        self._frame_rows.append(frame)
        if result.frame_index is not None:
            self._seen.add(int(result.frame_index))
        self._maybe_flush()

    def _note_wavelength(self, fid: str, filename: str, event: int) -> None:
        wl = self._frame_geometry(filename, event)["wavelength"]
        if wl != self.geometry["wavelength"]:
            self._recip_scale[fid] = float(self.geometry["wavelength"]) / float(wl)

    def _recips(self, fid, n, row, col) -> np.ndarray:
        scale = np.repeat([self._recip_scale.get(f, 1.0) for f in fid], n)
        return np.array(self._resolution_nm_inv_many(zip(row, col))) * scale

    def _maybe_flush(self) -> None:
        if (
            self._n_refl >= _FLUSH_ROWS
            or self._n_peak >= _FLUSH_ROWS
            or len(self._frame_rows) >= _FLUSH_ROWS
        ):
            self._flush()

    def _write_metadata_tables(self) -> None:
        assert self._conn is not None
        g = self.geometry
        bc = g.get("beam_center") or (None, None)
        wavelength = g.get("wavelength")
        photon_eV = (
            EV_ANGSTROM / float(wavelength) if wavelength not in (None, 0.0) else None
        )
        geom_text = None
        if self._geometry_file and self._geometry_file.is_file():
            geom_text = self._geometry_text()
        panels = _panel_bounds(g.get("panels"))
        self._conn.execute(
            "INSERT INTO geometry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                _as_float(bc[0]),
                _as_float(bc[1]),
                _as_float(g.get("clen")),
                _as_float(g.get("pixel_size")),
                _as_float(wavelength),
                photon_eV,
                _as_float(g.get("adu_per_photon")),
                len(panels),
                geom_text,
            ],
        )
        if panels:
            self._insert("panels", _columns(_PANEL_COLUMNS, panels))
        if self.integration:
            r = self.integration
            radii = r.get("radii") or (None, None, None)
            self._conn.execute(
                "INSERT INTO integration VALUES (?, ?, ?, ?, ?, ?)",
                [
                    *(_as_float(x) for x in radii),
                    _as_float(r.get("adu_per_photon")),
                    _as_float(r.get("bg_annulus_pixels")),
                    r.get("aperture"),
                ],
            )
        if self.cell is not None:
            c = self.cell
            self._conn.execute(
                "INSERT INTO cell VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    c.a,
                    c.b,
                    c.c,
                    math.degrees(c.alpha),
                    math.degrees(c.beta),
                    math.degrees(c.gamma),
                    c.volume,
                    c.lattice_type,
                    c.centering,
                    c.unique_axis,
                ],
            )

    def _gain(self, result=None) -> float:
        # ADU per photon the frame was processed with, else the geometry's; 1 if unset
        g = getattr(result, "adu_per_photon", None) if result is not None else None
        if g is None:
            g = (self.geometry or {}).get("adu_per_photon")
        try:
            g = float(g)
        except (TypeError, ValueError):
            return 1.0
        return g if g > 0.0 else 1.0

    def _peaks(self, result: "IndexResult") -> np.ndarray:
        n = len(result.intensities)
        return np.array(
            [
                *_to_array(result.positions).reshape(-1, 2).T,
                _to_array(result.intensities),
                _to_array(result.peak_background_sum, n),
                _to_array(result.peak_n_pixels, n),
                _nan_array(result.peak_n_masked, n),
                _nan_array(result.peak_posterior, n),
                *_aux_rows(getattr(result, "peak_aux", None), n),
            ]
        )

    def _reflections(self, result: "IndexResult") -> tuple:
        # pos (2, N), hkl (3, N), I/sigma/peak/bg (4, N), extras (10, N); NaN unrecorded
        if result.predicted_hkl is not None:
            assert (
                result.predicted_positions is not None
                and result.predicted_intensities is not None
                and result.predicted_sigmas is not None
                and result.predicted_peak is not None
                and result.predicted_background is not None
            )
            pos = _to_array(result.predicted_positions).T
            hkl = _to_array(result.predicted_hkl, dtype=np.int64).T
            vals = _to_array(
                torch.stack(
                    [
                        result.predicted_intensities,
                        result.predicted_sigmas,
                        result.predicted_peak,
                        result.predicted_background,
                    ]
                )
            )
            extras = _reflection_extras(result)
            extras = np.array(extras) if extras else np.full((3, pos.shape[1]), np.nan)
            flags = _reflection_flags(result)
            flags = np.array(flags) if flags else np.full((3, pos.shape[1]), np.nan)
            extras = np.vstack([extras, flags])
            # unsnapped prediction, NaN if not recorded
            raw = result.predicted_raw_positions
            raw = (
                _nan_array(None, 2 * pos.shape[1]).reshape(2, -1)
                if raw is None
                else _to_array(raw).T
            )
            extras = np.vstack([extras, raw])
            extras = np.vstack(
                [extras, _nan_array(result.predicted_peak_posterior, pos.shape[1])]
            )
            extras = np.vstack(
                [extras, _nan_array(result.predicted_peak_max, pos.shape[1])]
            )
        else:
            indexed = _to_array(result.indexed_mask, dtype=bool)
            pos = _to_array(result.positions)[indexed].T
            hkl = _to_array(result.hkl, dtype=np.int64)[indexed].T
            zero = np.zeros(pos.shape[1])
            vals = np.array(
                [
                    _to_array(result.intensities)[indexed],
                    _to_array(result.sigmas)[indexed],
                    zero,
                    zero,
                ]
            )
            extras = np.full((10, pos.shape[1]), np.nan)
        # an integrated disk with no usable pixel (sigma 0) is kept, flagged
        keep = np.isfinite(vals[1])
        if result.predicted_hkl is None:
            keep &= vals[1] > 0.0
        return pos[:, keep], hkl[:, keep], vals[:, keep], extras[:, keep]

    def _frame_row(
        self,
        fid: str,
        filename: str,
        event: int,
        frame_index: Optional[int],
        n_peaks: int,
        *,
        indexed: bool,
        num_reflections: Optional[int] = None,
        scale: Optional[float] = None,
        scale_sigma: Optional[float] = None,
        gain: Optional[float] = None,
        frame_stats=None,
    ) -> tuple:
        return (
            fid,
            None if frame_index is None else int(frame_index),
            filename,
            int(event),
            indexed,
            int(self._serial),
            int(n_peaks),
            None,  # peak_resolution_nm_inv: set from the peaks on flush
            _as_float(scale),
            _as_float(scale_sigma),
            None if num_reflections is None else int(num_reflections),
            _as_float(gain),
            *self._event_meta(filename, event),
            *_frame_stat_values(frame_stats),
            None,  # peak_falloff, peak_size_px, peak_ecc: set on flush
            None,
            None,
        )

    def _event_meta(self, filename: str, event: int) -> tuple:
        names, energies = self._event_names.get(filename), self._energies.get(filename)
        return (
            None if names is None else names[event],
            None if energies is None else _as_float(energies[event]),
        )

    def _crystal_row(
        self,
        result: "IndexResult",
        cid: str,
        fid: str,
        lattice_index: int,
        num_reflections: int,
    ) -> tuple:
        recovered = B_to_cell(result.A)
        A = result.A.detach().cpu().tolist()
        # columns of A are the reciprocal basis vectors a*/b*/c* (A^-1 -> nm^-1)
        astar = [A[r][0] * A_INV_TO_NM_INV for r in range(3)]
        bstar = [A[r][1] * A_INV_TO_NM_INV for r in range(3)]
        cstar = [A[r][2] * A_INV_TO_NM_INV for r in range(3)]
        limit = result.diffraction_limit
        drl = limit if (limit is not None and math.isfinite(limit)) else None
        return (
            cid,
            fid,
            int(lattice_index),
            int(result.n_indexed),
            float(result.rmsd),
            None if result.mosaicity is None else math.degrees(result.mosaicity),
            _profile_radius(result),
            _as_float(result.enrichment),
            None if result.n_bright is None else int(result.n_bright),
            _as_float(result.enrich_p),
            drl,
            int(num_reflections),
            recovered.a,  # B_to_cell returns edges in Å
            recovered.b,
            recovered.c,
            math.degrees(recovered.alpha),
            math.degrees(recovered.beta),
            math.degrees(recovered.gamma),
            astar[0],
            astar[1],
            astar[2],
            bstar[0],
            bstar[1],
            bstar[2],
            cstar[0],
            cstar[1],
            cstar[2],
            *_matched_values(result.predicted_peak_posterior),
        )

    def _emit_unindexed(self) -> None:
        assert self._conn is not None
        lo, hi = self._frame_range if self._frame_range is not None else (None, None)
        for start, stop, fname in self._ranges:
            for event in range(stop - start):
                idx = start + event
                if lo is not None and hi is not None and not (lo <= idx < hi):
                    continue
                if idx in self._seen:
                    continue
                filename, physical_event = self._locate(idx)
                self._frame_rows.append(
                    _unindexed_frame_row(
                        frame_id(filename, physical_event),
                        idx,
                        filename,
                        physical_event,
                        *self._event_meta(filename, physical_event),
                    )
                )
                if len(self._frame_rows) >= _FLUSH_ROWS:
                    self._flush()

    def _flush(self) -> None:
        assert self._conn is not None
        self._conn.execute("BEGIN TRANSACTION")
        try:
            self._flush_rows()
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def _flush_rows(self) -> None:
        assert self._conn is not None
        peaks, recips, summary = self._peak_table()
        i = _FRAME_COLUMNS.index("peak_resolution_nm_inv")
        j = _FRAME_COLUMNS.index("peak_falloff")
        frames = [
            (*row[:i], recip, *row[i + 1 : j], *map(_as_float, s), *row[j + 3 :])
            for row, recip, s in zip(
                self._frame_rows, recips.tolist(), summary.tolist()
            )
        ]
        frames += self._frame_rows[len(recips) :]
        for table, columns in (
            ("frames", _columns(_FRAME_COLUMNS, frames)),
            ("crystals", _columns(_CRYSTAL_COLUMNS, self._crystal_rows)),
            ("reflections", self._refl_table()),
            ("peaks", peaks),
        ):
            if columns:
                self._insert(table, columns)
        self._frame_rows.clear()
        self._crystal_rows.clear()
        self._refl_chunks.clear()
        self._peak_chunks.clear()
        self._recip_scale.clear()
        self._n_refl = self._n_peak = 0

    def _peak_table(self) -> tuple[dict, np.ndarray]:
        if not self._peak_chunks:
            return {}, np.zeros(0), np.zeros((0, 3))
        fid, gain, chunks = zip(*self._peak_chunks)
        n = [c.shape[1] for c in chunks]
        row, col, intensity, bg, npix, nmask, post, imax, ecc = np.concatenate(
            chunks, axis=1
        )
        recip = self._recips(fid, n, row, col)
        has = np.array(n) > 0
        starts = (np.cumsum(n) - n)[has]
        recips = np.zeros(len(n))
        recips[has] = np.fmax(0.0, np.fmax.reduceat(recip, starts))
        summary = np.full((len(n), 3), np.nan)
        if has.any():
            summary[has] = _frame_peak_summary(
                recip, intensity, npix, ecc, starts, np.array(n)[has]
            )
        gain = np.repeat(gain, n)
        with np.errstate(all="ignore"):
            photons, bg_photons = (intensity + bg) / gain, bg / gain
        columns = (
            np.repeat(fid, n),
            col,
            row,
            intensity,
            recip,
            self._panels_for(col, row),
            npix,
            photons,
            bg_photons,
            _int_column(nmask),
            (post, np.isfinite(post)),
            (imax, np.isfinite(imax)),
            (ecc, np.isfinite(ecc)),
        )
        return dict(zip(_PEAK_COLUMNS, columns)), recips, summary

    def _refl_table(self) -> dict:
        if not self._refl_chunks:
            return {}
        cid, fid, pos, hkl, vals, extras = zip(*self._refl_chunks)
        n = [p.shape[1] for p in pos]
        row, col = np.concatenate(pos, axis=1)
        (
            n_pix,
            bg_model,
            bg_var,
            n_total,
            n_masked,
            n_overlap,
            raw_row,
            raw_col,
            post,
            pmax,
        ) = np.concatenate(extras, axis=1)
        recorded = ~np.isnan(n_pix)
        edge = self._edge_px(col, row)
        columns = (
            np.repeat(cid, n),
            np.repeat(fid, n),
            *np.concatenate(hkl, axis=1),
            *np.concatenate(vals, axis=1),
            col,
            row,
            self._panels_for(col, row),
            self._recips(fid, n, row, col),
            (np.where(recorded, n_pix, 0).astype(np.int64), recorded),
            (bg_model, np.isfinite(bg_model)),
            (bg_var, np.isfinite(bg_var)),
            _int_column(n_total),
            _int_column(n_masked),
            _int_column(n_overlap),
            (raw_col, np.isfinite(raw_col)),
            (raw_row, np.isfinite(raw_row)),
            (post, np.isfinite(post)),
            (edge, np.isfinite(edge)),
            (pmax, np.isfinite(pmax)),
        )
        return dict(zip(_REFLECTION_COLUMNS, columns))

    def _panels_for(self, fs: np.ndarray, ss: np.ndarray) -> np.ndarray:
        if not self._panels:
            return np.full(len(fs), self.panel)
        name, min_fs, max_fs, min_ss, max_ss = map(np.array, zip(*self._panels))
        fs, ss = fs[:, None], ss[:, None]
        inside = (min_fs <= fs) & (fs <= max_fs) & (min_ss <= ss) & (ss <= max_ss)
        return np.where(inside.any(axis=1), name[inside.argmax(axis=1)], self.panel)

    def _edge_px(self, fs: np.ndarray, ss: np.ndarray) -> np.ndarray:
        # px to the nearest edge of the containing panel, NaN off every panel
        if not self._panels:
            return np.full(len(fs), np.nan)
        _, min_fs, max_fs, min_ss, max_ss = map(np.array, zip(*self._panels))
        fs, ss = fs[:, None], ss[:, None]
        inside = (min_fs <= fs) & (fs <= max_fs) & (min_ss <= ss) & (ss <= max_ss)
        dist = np.minimum(
            np.minimum(fs - min_fs, max_fs - fs), np.minimum(ss - min_ss, max_ss - ss)
        )
        dist = np.where(inside, dist, np.inf).min(axis=1)
        return np.where(np.isfinite(dist), dist, np.nan)

    def _insert(self, table: str, columns: dict) -> None:
        assert self._conn is not None
        cols = [c if isinstance(c, tuple) else (c, None) for c in columns.values()]
        n = len(cols[0][0])
        nan = np.zeros(n, dtype=bool)
        for values, valid in cols:
            if values.dtype.kind == "f":
                nan |= np.isnan(values) if valid is None else np.isnan(values) & valid
        names = ", ".join(columns)
        # A numpy scan reads NaN as NULL, so rows with a NaN to keep are bound singly
        start = 0
        for stop in (*np.flatnonzero(nan), n):
            if stop > start:
                data, select = {}, []
                for i, (values, valid) in enumerate(cols):
                    data[f"v{i}"] = values[start:stop]
                    if valid is None:
                        select.append(f"v{i}")
                    else:
                        data[f"ok{i}"] = valid[start:stop]
                        select.append(f"CASE WHEN ok{i} THEN v{i} END")
                self._conn.register("buf", data)
                try:
                    self._conn.execute(
                        f"INSERT INTO {table} ({names}) "
                        f"SELECT {', '.join(select)} FROM buf"
                    )
                finally:
                    self._conn.unregister("buf")
            if stop < n:
                self._conn.execute(
                    f"INSERT INTO {table} ({names}) VALUES ({', '.join('?' * len(cols))})",
                    [
                        v[stop].item() if ok is None or ok[stop] else None
                        for v, ok in cols
                    ],
                )
            start = stop + 1


def _to_array(t, n: int = 0, dtype=np.float64) -> np.ndarray:
    # Optional tensor -> array, or zeros(n) when it was not recorded
    if t is None:
        return np.zeros(n, dtype=dtype)
    return t.detach().cpu().numpy().astype(dtype)


def _reflection_extras(result) -> tuple:
    # (n_pixels, background_model, background_model_var) lists; () if not integrated
    cols = (
        result.predicted_n_pixels,
        result.predicted_bg_model,
        result.predicted_bg_model_var,
    )
    if any(c is None for c in cols):
        return ()
    return tuple(c.detach().cpu().tolist() for c in cols)


def _nan_array(t, n: int) -> np.ndarray:
    return np.full(n, np.nan) if t is None else _to_array(t)


def _aux_rows(aux, n: int) -> tuple:
    # (brightest pixel, eccentricity) per peak, NaN if absent
    if aux is None:
        return np.full(n, np.nan), np.full(n, np.nan)
    a = _to_array(aux).reshape(n, -1)
    return a[:, 0], a[:, 1]


def _frame_stat_values(stats) -> list:
    # the six frame background stats as floats, None if unavailable
    if stats is None:
        return [None] * 6
    return [_as_float(v) for v in _to_array(stats).reshape(-1).tolist()]


def _matched_values(post) -> tuple:
    # snapped fraction of predictions and their mean posterior
    if post is None or post.numel() == 0:
        return None, None
    hit = torch.isfinite(post)
    mean = float(post[hit].mean()) if bool(hit.any()) else None
    return float(hit.float().mean()), mean


def _frame_peak_summary(recip, intensity, npix, ecc, starts, counts) -> np.ndarray:
    # per frame: falloff -d ln I/d(1/d^2) (nm^2, NaN < 8 peaks), footprint, eccentricity
    def sums(v):
        return np.add.reduceat(v, starts)

    pos = intensity > 0
    with np.errstate(all="ignore"):
        x = np.where(pos, recip * recip, 0.0)
        y = np.where(pos, np.log(np.where(pos, intensity, 1.0)), 0.0)
        n = np.fmax(sums(pos.astype(float)), 1.0)
        sx, sy = sums(x), sums(y)
        den = sums(x * x) - sx * sx / n
        slope = (sums(x * y) - sx * sy / n) / den
        falloff = np.where((sums(pos.astype(float)) >= 8) & (den > 0), -slope, np.nan)
        fin = np.isfinite(ecc)
        n_ecc = sums(fin.astype(float))
        mean_ecc = np.where(
            n_ecc > 0, sums(np.where(fin, ecc, 0.0)) / np.fmax(n_ecc, 1), np.nan
        )
        size = sums(npix) / counts
    return np.column_stack([falloff, size, mean_ecc])


def _int_column(values: np.ndarray) -> tuple:
    # float array with NaN for NULL -> (int64 values, valid) for _insert
    ok = np.isfinite(values)
    return np.where(ok, values, 0).astype(np.int64), ok


def _columns(names: tuple, rows: list) -> dict:
    # rows of Python values (None is NULL) -> arrays, or (array, valid) with NULLs
    columns = {}
    for name, values in zip(names, zip(*rows)):
        valid = np.array([v is not None for v in values])
        zero = type(next((v for v in values if v is not None), 0.0))()
        array = np.array([zero if v is None else v for v in values])
        columns[name] = array if valid.all() else (array, valid)
    return columns


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _unindexed_frame_row(
    fid: str, idx: int, fname: str, event: int, name=None, energy=None
) -> tuple:
    row: list = [None] * len(_FRAME_COLUMNS)
    row[:5] = (fid, int(idx), fname, int(event), False)
    row[_FRAME_COLUMNS.index("event_name")] = name
    row[_FRAME_COLUMNS.index("photon_energy_eV")] = energy
    return tuple(row)


def attach_frame_values(path: PathLike, table: PathLike, sep: str = "\t") -> list[str]:
    # Columns past filename/event become DOUBLE frames columns; empty or nan is NULL
    import csv

    with open(table, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=sep))
    names = [
        c for c in (rows[0].keys() if rows else []) if c not in ("filename", "event")
    ]
    if not names:
        raise ValueError(f"{table}: no value columns beyond filename and event")
    conn = duckdb.connect(str(path))
    try:
        have = {r[0] for r in conn.execute("DESCRIBE frames").fetchall()}
        for name in names:
            if name not in have:
                conn.execute(f'ALTER TABLE frames ADD COLUMN "{name}" DOUBLE')
        recorded = {
            f
            for (f,) in conn.execute("SELECT DISTINCT filename FROM frames").fetchall()
        }
        by_base = {Path(f).name: f for f in recorded}  # fallback match on basename
        filenames = [
            (
                r["filename"]
                if r["filename"] in recorded
                else by_base.get(Path(r["filename"]).name, r["filename"])
            )
            for r in rows
        ]
        data = {
            "filename": np.array(filenames),
            "event": np.array(  # "//N" or "N"
                [int(str(r["event"]).strip().lstrip("/") or 0) for r in rows]
            ),
        }
        for name in names:
            data[name] = np.array(
                [
                    (
                        float(r[name])
                        if r[name] not in (None, "", "nan", "NaN")
                        else np.nan
                    )
                    for r in rows
                ]
            )
        conn.register("frame_values", data)
        try:
            sets = ", ".join(
                f'"{n}" = CASE WHEN isnan(v."{n}") THEN NULL ELSE v."{n}" END'
                for n in names
            )
            conn.execute(
                f"UPDATE frames SET {sets} FROM frame_values v "
                "WHERE frames.filename = v.filename AND frames.event = v.event"
            )
        finally:
            conn.unregister("frame_values")
    finally:
        conn.close()
    return names
