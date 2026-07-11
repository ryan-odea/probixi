from __future__ import annotations

import math

import pytest
import torch

duckdb = pytest.importorskip("duckdb")

from probixi.indexer.indexer import IndexResult  # noqa: E402
from probixi.indexer.lattice import B_to_cell  # noqa: E402
from probixi.io.db import DuckDBOffloader, frame_id  # noqa: E402
from probixi.io.metadata import H5Info  # noqa: E402


def _make_index_result(cell, frame_index: int = 0) -> IndexResult:
    torch.manual_seed(0)
    # three indexed peaks, one un-indexed: the crystal block keeps only the three
    positions = torch.tensor(
        [[40.0, 55.0], [70.0, 30.0], [90.0, 110.0], [10.0, 12.0]],
        dtype=torch.float64,
    )
    intensities = torch.tensor([1200.0, 800.0, 450.0, 50.0], dtype=torch.float64)
    # last indexed peak carries sigma <= 0 -> must be dropped from reflections
    sigmas = torch.tensor([35.0, 28.0, 0.0, 7.0], dtype=torch.float64)
    indexed_mask = torch.tensor([True, True, True, False])
    hkl = torch.tensor([[1, 0, 0], [0, 1, -1], [2, -1, 3], [0, 0, 0]], dtype=torch.long)
    A = torch.tensor(
        [[0.016, 0.001, 0.000], [0.000, 0.016, 0.002], [0.001, 0.000, 0.009]],
        dtype=torch.float64,
    )
    return IndexResult(
        frame_index=frame_index,
        n_peaks=4,
        n_indexed=3,
        rmsd=0.0042,
        A=A,
        U=torch.eye(3, dtype=torch.float64),
        B=A.clone(),
        cell=cell,
        indexed_mask=indexed_mask,
        hkl=hkl,
        positions=positions,
        intensities=intensities,
        sigmas=sigmas,
        loss_history=torch.zeros(1, dtype=torch.float64),
        scale=0.97,
        scale_sigma=0.01,
    )


def _make_peak_result(frame_index: int = 0):
    from probixi.peakfinding.peaks.blobs import BlobStats
    from probixi.peakfinding.peaks.peakfinder import PeakResult

    rows = torch.tensor([40.0, 70.0, 90.0], dtype=torch.float64)
    cols = torch.tensor([55.0, 30.0, 110.0], dtype=torch.float64)
    isum = torch.tensor([1200.0, 800.0, 450.0], dtype=torch.float64)
    n = rows.numel()
    el = torch.arange(1, n + 1, dtype=torch.long)
    z = torch.zeros(n, dtype=torch.float64)
    stats = BlobStats(
        label_id=el,
        size=torch.full((n,), 9, dtype=torch.long),
        row_centroid=rows,
        col_centroid=cols,
        bbox_r0=el,
        bbox_r1=el,
        bbox_c0=el,
        bbox_c1=el,
        intensity_sum=isum,
        intensity_sigma=z,
        intensity_max=z,
        z_max=z,
        log_bf_sum=z,
        posterior_mean=z,
        eccentricity=z,
        peakedness=z,
    )
    return PeakResult(
        frame_index=frame_index,
        scores={},
        labels=torch.zeros(1, 1, dtype=torch.long),
        stats=stats,
        keep=torch.ones(n, dtype=torch.bool),
    )


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SHOW TABLES").fetchall()}


def test_frame_id_is_stable_and_deterministic():
    assert frame_id("a.h5", 3) == frame_id("a.h5", 3)
    assert frame_id("a.h5", 3) != frame_id("a.h5", 4)
    assert frame_id("a.h5", 3) != frame_id("b.h5", 3)


def test_write_before_context_raises(geometry_dict, cell, tmp_path):
    off = DuckDBOffloader(tmp_path / "x.duckdb", geometry=geometry_dict, cell=cell)
    with pytest.raises(RuntimeError, match="context manager"):
        off.write(_make_index_result(cell))


def test_db_creates_all_tables(geometry_dict, cell, tmp_path):
    out = tmp_path / "out.duckdb"
    with DuckDBOffloader(out, geometry=geometry_dict, cell=cell) as off:
        off.write(_make_index_result(cell))

    conn = duckdb.connect(str(out), read_only=True)
    try:
        assert {
            "geometry",
            "panels",
            "cell",
            "frames",
            "reflections",
            "peaks",
        } <= _tables(conn)
    finally:
        conn.close()


def test_db_geometry_and_cell_metadata(geometry_dict, cell, tmp_path, geom_file):
    out = tmp_path / "out.duckdb"
    with DuckDBOffloader(
        out, geometry=geometry_dict, cell=cell, geometry_file=geom_file
    ) as off:
        off.write(_make_index_result(cell))

    conn = duckdb.connect(str(out), read_only=True)
    try:
        g = conn.execute(
            "SELECT beam_center_row, beam_center_col, clen, pixel_size, wavelength, "
            "n_panels, geometry_file FROM geometry"
        ).fetchone()
        assert g[0] == pytest.approx(geometry_dict["beam_center"][0])
        assert g[1] == pytest.approx(geometry_dict["beam_center"][1])
        assert g[2] == pytest.approx(geometry_dict["clen"])
        assert g[5] == 1  # single Eiger panel
        assert geom_file.read_text() in g[6]

        panels = conn.execute("SELECT name, min_fs, max_fs FROM panels").fetchall()
        assert panels == [("0", 0, 2069)]

        c = conn.execute("SELECT a_A, alpha_deg, lattice_type FROM cell").fetchone()
        assert c[0] == pytest.approx(cell.a)
        assert c[1] == pytest.approx(math.degrees(cell.alpha))
        assert c[2] == cell.lattice_type
    finally:
        conn.close()


def test_db_frame_row_and_reflection_join(geometry_dict, cell, tmp_path):
    out = tmp_path / "out.duckdb"
    result = _make_index_result(cell)
    with DuckDBOffloader(out, geometry=geometry_dict, cell=cell) as off:
        off.write(result)

    conn = duckdb.connect(str(out), read_only=True)
    try:
        frame = conn.execute(
            "SELECT frame_id, indexed, n_peaks, n_indexed, rmsd, scale, "
            "num_reflections FROM frames"
        ).fetchone()
        fid, indexed, n_peaks, n_indexed, rmsd, scale, num_refl = frame
        assert indexed is True
        assert n_peaks == 4
        assert n_indexed == 3
        assert rmsd == pytest.approx(0.0042)
        assert scale == pytest.approx(0.97)
        # sigma <= 0 reflection dropped -> only two of the three indexed remain
        assert num_refl == 2

        # recovered cell recorded in Angstroms, matching B_to_cell
        recovered = B_to_cell(result.A)
        cell_a = conn.execute("SELECT cell_a_A FROM frames").fetchone()[0]
        assert cell_a == pytest.approx(recovered.a)

        # reflections join back to the frame and have positive sigma
        rows = conn.execute(
            "SELECT r.h, r.k, r.l, r.sigma FROM reflections r "
            "JOIN frames f USING (frame_id) WHERE f.frame_id = ?",
            [fid],
        ).fetchall()
        assert len(rows) == 2
        assert all(sigma > 0.0 for *_, sigma in rows)
    finally:
        conn.close()


def test_db_peaks_table(geometry_dict, cell, tmp_path):
    out = tmp_path / "out.duckdb"
    with DuckDBOffloader(out, geometry=geometry_dict, cell=cell) as off:
        off.write(_make_index_result(cell))

    conn = duckdb.connect(str(out), read_only=True)
    try:
        n_peaks, max_res = conn.execute(
            "SELECT COUNT(*), MAX(resolution_nm_inv) FROM peaks"
        ).fetchone()
        assert n_peaks == 4  # every searched peak, indexed or not
        assert max_res > 0.0
        # the frame's peak_resolution matches the brightest-shell peak
        frame_res = conn.execute(
            "SELECT peak_resolution_nm_inv FROM frames"
        ).fetchone()[0]
        assert frame_res == pytest.approx(max_res)
    finally:
        conn.close()


def test_db_backfills_unindexed_frames(geometry_dict, cell, tmp_path):
    out = tmp_path / "out.duckdb"
    files = {"a.h5": H5Info("a.h5", "/d", n_frames=3, frame_shape=(8, 8))}
    with DuckDBOffloader(out, geometry=geometry_dict, cell=cell, files=files) as off:
        off.write(_make_index_result(cell, frame_index=0))  # only event 0 indexes

    conn = duckdb.connect(str(out), read_only=True)
    try:
        total, indexed = conn.execute(
            "SELECT COUNT(*), SUM(indexed::INT) FROM frames"
        ).fetchone()
        assert total == 3  # all file-events recorded
        assert indexed == 1

        # the non-indexed rows carry their file-event identity but null stats
        rows = conn.execute(
            "SELECT frame_index, event, rmsd FROM frames WHERE NOT indexed "
            "ORDER BY frame_index"
        ).fetchall()
        assert [(r[0], r[1]) for r in rows] == [(1, 1), (2, 2)]
        assert all(r[2] is None for r in rows)

        # the indexed frame_id matches the deterministic key for its file-event
        fid = conn.execute("SELECT frame_id FROM frames WHERE indexed").fetchone()[0]
        assert fid == frame_id("a.h5", 0)
    finally:
        conn.close()


def test_merge_dbs_unions_blocks_and_dedups_metadata(geometry_dict, cell, tmp_path):
    from probixi.multigpu import merge_dbs

    files = {"a.h5": H5Info("a.h5", "/d", n_frames=4, frame_shape=(8, 8))}
    # rank 0 owns events [0, 2), indexes event 0; rank 1 owns [2, 4), indexes 2
    part0 = tmp_path / "out.duckdb.rank0"
    part1 = tmp_path / "out.duckdb.rank1"
    with DuckDBOffloader(
        part0, geometry=geometry_dict, cell=cell, files=files, frame_range=(0, 2)
    ) as off:
        off.write(_make_index_result(cell, frame_index=0))
    with DuckDBOffloader(
        part1, geometry=geometry_dict, cell=cell, files=files, frame_range=(2, 4)
    ) as off:
        off.write(_make_index_result(cell, frame_index=2))

    out = tmp_path / "out.duckdb"
    n_indexed = merge_dbs([part0, part1], out)
    assert n_indexed == 2

    conn = duckdb.connect(str(out), read_only=True)
    try:
        total, indexed = conn.execute(
            "SELECT COUNT(*), SUM(indexed::INT) FROM frames"
        ).fetchone()
        assert total == 4  # every file-event across both blocks, once
        assert indexed == 2
        # metadata copied exactly once, not per-rank
        assert conn.execute("SELECT COUNT(*) FROM geometry").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM cell").fetchone()[0] == 1
        # reflections from both indexed frames present and joinable
        joined = conn.execute(
            "SELECT COUNT(*) FROM reflections r JOIN frames f USING (frame_id)"
        ).fetchone()[0]
        assert joined == conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
        assert joined > 0
    finally:
        conn.close()


def test_merge_dbs_skips_missing_parts(geometry_dict, cell, tmp_path):
    from probixi.multigpu import merge_dbs

    part0 = tmp_path / "out.duckdb.rank0"
    with DuckDBOffloader(part0, geometry=geometry_dict, cell=cell) as off:
        off.write(_make_index_result(cell))
    missing = tmp_path / "out.duckdb.rank1"  # never created (crashed worker)

    out = tmp_path / "out.duckdb"
    assert merge_dbs([part0, missing], out) == 1


def test_write_peaks_records_frame_and_peaks(geometry_dict, tmp_path):
    out = tmp_path / "peaks.duckdb"
    files = {"a.h5": H5Info("a.h5", "/d", n_frames=2, frame_shape=(8, 8))}
    # peaks-only: no cell, no indexing; event 0 has peaks, event 1 has none
    with DuckDBOffloader(
        out, geometry=geometry_dict, files=files, frame_range=(0, 2)
    ) as off:
        off.write_peaks(_make_peak_result(frame_index=0))

    conn = duckdb.connect(str(out), read_only=True)
    try:
        # three peaks recorded for the one processed frame
        assert conn.execute("SELECT COUNT(*) FROM peaks").fetchone()[0] == 3
        # no reflections and no cell in peaks-only mode
        assert conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM cell").fetchone()[0] == 0

        # the processed frame is recorded not-indexed with its peak count
        fid, indexed, n_peaks, peak_res = conn.execute(
            "SELECT frame_id, indexed, n_peaks, peak_resolution_nm_inv "
            "FROM frames WHERE frame_index = 0"
        ).fetchone()
        assert indexed is False
        assert n_peaks == 3
        assert peak_res > 0.0
        assert fid == frame_id("a.h5", 0)

        # every peak joins back to its frame
        joined = conn.execute(
            "SELECT COUNT(*) FROM peaks p JOIN frames f USING (frame_id)"
        ).fetchone()[0]
        assert joined == 3

        # the peak-less frame is still recorded (backfilled), not indexed
        total = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        assert total == 2
        assert conn.execute("SELECT SUM(indexed::INT) FROM frames").fetchone()[0] == 0
    finally:
        conn.close()


def test_db_overwrites_existing_file(geometry_dict, cell, tmp_path):
    out = tmp_path / "out.duckdb"
    out.write_text("not a database")
    with DuckDBOffloader(out, geometry=geometry_dict, cell=cell) as off:
        off.write(_make_index_result(cell))
    conn = duckdb.connect(str(out), read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 1
    finally:
        conn.close()
