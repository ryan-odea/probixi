from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import duckdb
import torch

from ..indexer.lattice import B_to_cell
from .geometry import EV_ANGSTROM
from .writer import A_INV_TO_NM_INV, _panel_bounds, _profile_radius, _StreamWriter

if TYPE_CHECKING:
    from ..indexer.indexer import IndexResult
    from .cell import CellParams

PathLike = Union[str, Path]
_DB_SUFFIXES = {".duckdb", ".db"}
_FLUSH_ROWS = 50_000


def is_duckdb_path(path: Optional[PathLike]) -> bool:
    """True if ``path`` names a DuckDB output (``.duckdb`` / ``.db`` suffix)."""
    return path is not None and Path(path).suffix.lower() in _DB_SUFFIXES


_FRAME_COLUMNS = (
    "frame_id",
    "frame_index",
    "filename",
    "event",
    "indexed",
    "serial",
    "n_peaks",
    "n_indexed",
    "rmsd",
    "scale",
    "scale_sigma",
    "mosaicity_deg",
    "profile_radius_nm_inv",
    "enrichment",
    "n_bright",
    "enrich_p",
    "diffraction_limit_nm_inv",
    "peak_resolution_nm_inv",
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

CREATE TABLE frames (
    frame_id                 VARCHAR PRIMARY KEY,
    frame_index              BIGINT,
    filename                 VARCHAR,
    event                    BIGINT,
    indexed                  BOOLEAN,
    serial                   BIGINT,
    n_peaks                  INTEGER,
    n_indexed                INTEGER,
    rmsd                     DOUBLE,
    scale                    DOUBLE,
    scale_sigma              DOUBLE,
    mosaicity_deg            DOUBLE,
    profile_radius_nm_inv    DOUBLE,
    enrichment               DOUBLE,
    n_bright                 INTEGER,
    enrich_p                 DOUBLE,
    diffraction_limit_nm_inv DOUBLE,
    peak_resolution_nm_inv   DOUBLE,
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
    cstar_z                  DOUBLE
);

CREATE TABLE reflections (
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
    resolution_nm_inv DOUBLE
);

CREATE TABLE peaks (
    frame_id          VARCHAR,
    fs                DOUBLE,
    ss                DOUBLE,
    intensity         DOUBLE,
    resolution_nm_inv DOUBLE,
    panel             VARCHAR
);
"""

_INDEXES = """
CREATE INDEX idx_reflections_frame ON reflections(frame_id);
CREATE INDEX idx_peaks_frame ON peaks(frame_id);
"""


def frame_id(filename: str, event: int) -> str:
    digest = hashlib.sha1(f"{filename}//{event}".encode("utf-8"))
    return digest.hexdigest()[:16]


class DuckDBOffloader(_StreamWriter):
    """Write ``IndexResult``s to a DuckDB database.

    A relational alternative to the CrystFEL ``.stream``: run metadata lands in
    small ``geometry``/``panels``/``cell`` tables, every file-event becomes a row
    in ``frames`` (flagged indexed or not, with its per-frame statistics), and
    the integrated ``reflections`` and searched ``peaks`` are stored keyed by the
    frame's :func:`frame_id`.

    Same interface as :class:`~probixi.io.writer.DataOffloader`, so it drops into
    the same driver loop::

        with DuckDBOffloader(out, geometry=geom, cell=cell, files=files) as off:
            stream.to_stream(off)

    or, more directly, via :meth:`~probixi.indexer.indexer.IndexStream.to_db`.

    When ``files`` is supplied every file-event is enumerated, so frames that
    never indexed are recorded with ``indexed = FALSE`` and null statistics; the
    indexed rate is then ``AVG(indexed::INT)`` over ``frames``. Without ``files``
    only indexed frames are written.

    Parameters
    ----------
    path : str or Path
        Destination ``.duckdb`` file. Overwritten if it exists.
    geometry : dict
        Indexer geometry (``beam_center``, ``clen``, ``pixel_size``,
        ``wavelength``, ``adu_per_photon``, and -- when available -- ``panels``).
    cell : CellParams, optional
        Target unit cell; written to the ``cell`` table and used for any
        symmetry labels.
    geometry_file : str or Path, optional
        Geometry file whose text is stored verbatim in ``geometry.geometry_file``.
    files : dict, optional
        Loader file map, used both to resolve a global frame index to its source
        file/event and to enumerate the non-indexed frames.
    frame_range : tuple[int, int], optional
        Half-open ``[lo, hi)`` global-frame-index range this writer is
        responsible for. When set, the non-indexed backfill is restricted to
        this range -- required when only a sub-range is processed (``--start`` /
        ``--stop``) or when several writers each cover a disjoint block (the
        multi-GPU path), so the whole dataset is not marked non-indexed by every
        writer. ``None`` backfills every file-event.
    indexer_name : str, default "probixi"
        Recorded for provenance parity with the stream writer (unused in the DB).
    panel : str, default "0"
        Fallback panel name for peaks/reflections outside every geometry panel.
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
        self._frame_range = frame_range
        self._conn = None
        self._frame_rows: list[tuple] = []
        self._refl_rows: list[tuple] = []
        self._peak_rows: list[tuple] = []
        self._seen: set[int] = set()

    def __enter__(self) -> "DuckDBOffloader":
        if self.path.exists():
            self.path.unlink()
        self._conn = duckdb.connect(str(self.path))
        self._conn.execute(_SCHEMA)
        self._write_metadata_tables()
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

    def write(self, result: "IndexResult") -> None:
        """Buffer one indexed frame (its stats, reflections and peaks)."""
        if self._conn is None:
            raise RuntimeError("DuckDBOffloader must be used as a context manager")
        self._serial += 1
        filename, event = self._locate(result.frame_index)
        fid = frame_id(filename, event)

        peak_recip = self._append_peaks(result, fid)
        refl = self._reflections(result)
        for (row, col), miller, intensity, sigma, peak, background in refl:
            h, k, l = miller  # noqa: E741
            self._refl_rows.append(
                (
                    fid,
                    int(h),
                    int(k),
                    int(l),
                    float(intensity),
                    float(sigma),
                    float(peak),
                    float(background),
                    float(col),
                    float(row),
                    self._panel_for(col, row),
                    self._resolution_nm_inv(row, col),
                )
            )

        self._frame_rows.append(
            self._frame_row(result, fid, filename, event, len(refl), peak_recip)
        )
        if result.frame_index is not None:
            self._seen.add(int(result.frame_index))
        self._maybe_flush()

    def write_peaks(self, result) -> None:
        """Buffer one frame's peak-search result (peaks-only; no indexing).

        Records a ``frames`` row with ``indexed = FALSE`` and the peak count, and
        the searched peaks in the ``peaks`` table; the ``reflections`` table stays
        empty. ``result`` is a
        :class:`~probixi.peakfinding.peaks.peakfinder.PeakResult`.
        """
        if self._conn is None:
            raise RuntimeError("DuckDBOffloader must be used as a context manager")
        self._serial += 1
        filename, event = self._locate(result.frame_index)
        fid = frame_id(filename, event)

        stats = result.kept_stats
        rows, cols, intensities = (
            torch.stack([stats.row_centroid, stats.col_centroid, stats.intensity_sum])
            .detach()
            .cpu()
            .tolist()
        )
        max_recip = 0.0
        for row, col, intensity in zip(rows, cols, intensities):
            recip = self._resolution_nm_inv(row, col)
            max_recip = max(max_recip, recip)
            self._peak_rows.append(
                (
                    fid,
                    float(col),
                    float(row),
                    float(intensity),
                    recip,
                    self._panel_for(col, row),
                )
            )

        self._frame_rows.append(
            self._peaks_frame_row(
                fid, filename, event, result.frame_index, len(rows), max_recip
            )
        )
        if result.frame_index is not None:
            self._seen.add(int(result.frame_index))
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        if (
            len(self._refl_rows) >= _FLUSH_ROWS
            or len(self._peak_rows) >= _FLUSH_ROWS
            or len(self._frame_rows) >= _FLUSH_ROWS
        ):
            self._flush()

    # META =================================

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
            geom_text = self._geometry_file.read_text()
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
            self._conn.executemany("INSERT INTO panels VALUES (?, ?, ?, ?, ?)", panels)
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

    # FRAME =========================

    def _append_peaks(self, result: "IndexResult", fid: str) -> float:
        positions = result.positions.detach().cpu().tolist()
        intensities = result.intensities.detach().cpu().tolist()
        max_recip = 0.0
        for (row, col), intensity in zip(positions, intensities):
            recip = self._resolution_nm_inv(row, col)
            max_recip = max(max_recip, recip)
            self._peak_rows.append(
                (
                    fid,
                    float(col),
                    float(row),
                    float(intensity),
                    recip,
                    self._panel_for(col, row),
                )
            )
        return max_recip

    def _reflections(self, result: "IndexResult") -> list:
        if result.predicted_hkl is not None:
            assert (
                result.predicted_positions is not None
                and result.predicted_intensities is not None
                and result.predicted_sigmas is not None
                and result.predicted_peak is not None
                and result.predicted_background is not None
            )
            p_pos = result.predicted_positions.detach().cpu().tolist()
            p_hkl = result.predicted_hkl.detach().cpu().tolist()
            p_int, p_sig, p_pk, p_bg = (
                torch.stack(
                    [
                        result.predicted_intensities,
                        result.predicted_sigmas,
                        result.predicted_peak,
                        result.predicted_background,
                    ]
                )
                .detach()
                .cpu()
                .tolist()
            )
            refl = list(zip(p_pos, p_hkl, p_int, p_sig, p_pk, p_bg))
        else:
            positions = result.positions.detach().cpu().tolist()
            intensities = result.intensities.detach().cpu().tolist()
            sigmas = result.sigmas.detach().cpu().tolist()
            indexed = result.indexed_mask.detach().cpu().tolist()
            hkl = result.hkl.detach().cpu().tolist()
            refl = [
                (rc, hk, i, s, 0.0, 0.0)
                for rc, hk, keep, i, s in zip(
                    positions, hkl, indexed, intensities, sigmas
                )
                if keep
            ]
        return [r for r in refl if math.isfinite(r[3]) and r[3] > 0.0]

    def _frame_row(
        self,
        result: "IndexResult",
        fid: str,
        filename: str,
        event: int,
        num_reflections: int,
        peak_recip: float,
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
            fid,
            None if result.frame_index is None else int(result.frame_index),
            filename,
            int(event),
            True,
            int(self._serial),
            int(result.n_peaks),
            int(result.n_indexed),
            float(result.rmsd),
            _as_float(result.scale),
            _as_float(result.scale_sigma),
            None if result.mosaicity is None else math.degrees(result.mosaicity),
            _profile_radius(result),
            _as_float(result.enrichment),
            None if result.n_bright is None else int(result.n_bright),
            _as_float(result.enrich_p),
            drl,
            peak_recip,
            int(num_reflections),
            recovered.a,  # B_to_cell returns edges in Angstroms
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
        )

    def _peaks_frame_row(
        self,
        fid: str,
        filename: str,
        event: int,
        frame_index: Optional[int],
        n_peaks: int,
        peak_recip: float,
    ) -> tuple:
        # peaks-only frame: no indexing, so only the peak count/resolution are set
        row: list = [None] * len(_FRAME_COLUMNS)
        for name, value in (
            ("frame_id", fid),
            ("frame_index", None if frame_index is None else int(frame_index)),
            ("filename", filename),
            ("event", int(event)),
            ("indexed", False),
            ("serial", int(self._serial)),
            ("n_peaks", int(n_peaks)),
            ("peak_resolution_nm_inv", peak_recip),
        ):
            row[_FRAME_COLUMNS.index(name)] = value
        return tuple(row)

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
                self._frame_rows.append(
                    _unindexed_frame_row(frame_id(fname, event), idx, fname, event)
                )
                if len(self._frame_rows) >= _FLUSH_ROWS:
                    self._flush()

    def _flush(self) -> None:
        assert self._conn is not None
        if self._frame_rows:
            self._conn.executemany(
                f"INSERT INTO frames VALUES ({', '.join('?' * len(_FRAME_COLUMNS))})",
                self._frame_rows,
            )
            self._frame_rows.clear()
        if self._refl_rows:
            self._conn.executemany(
                "INSERT INTO reflections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._refl_rows,
            )
            self._refl_rows.clear()
        if self._peak_rows:
            self._conn.executemany(
                "INSERT INTO peaks VALUES (?, ?, ?, ?, ?, ?)",
                self._peak_rows,
            )
            self._peak_rows.clear()


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _unindexed_frame_row(fid: str, idx: int, fname: str, event: int) -> tuple:
    return (
        fid,
        int(idx),
        fname,
        int(event),
        False,
        *([None] * (len(_FRAME_COLUMNS) - 5)),
    )
