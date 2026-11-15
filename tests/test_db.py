from __future__ import annotations

import math

import numpy as np
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
        dtype=torch.float32,
    )
    intensities = torch.tensor([1200.0, 800.0, 450.0, 50.0], dtype=torch.float32)
    # last indexed peak carries sigma <= 0 -> must be dropped from reflections
    sigmas = torch.tensor([35.0, 28.0, 0.0, 7.0], dtype=torch.float32)
    indexed_mask = torch.tensor([True, True, True, False])
    hkl = torch.tensor([[1, 0, 0], [0, 1, -1], [2, -1, 3], [0, 0, 0]], dtype=torch.long)
    A = torch.tensor(
        [[0.016, 0.001, 0.000], [0.000, 0.016, 0.002], [0.001, 0.000, 0.009]],
        dtype=torch.float32,
    )
    return IndexResult(
        frame_index=frame_index,
        n_peaks=4,
        n_indexed=3,
        rmsd=0.0042,
        A=A,
        U=torch.eye(3, dtype=torch.float32),
        B=A.clone(),
        cell=cell,
        indexed_mask=indexed_mask,
        hkl=hkl,
        positions=positions,
        intensities=intensities,
        sigmas=sigmas,
        loss_history=torch.zeros(1, dtype=torch.float32),
        scale=0.97,
        scale_sigma=0.01,
    )


def _make_peak_result(frame_index: int = 0):
    from probixi.peakfinding.peaks.blobs import BlobStats
    from probixi.peakfinding.peaks.peakfinder import PeakResult

    rows = torch.tensor([40.0, 70.0, 90.0], dtype=torch.float32)
    cols = torch.tensor([55.0, 30.0, 110.0], dtype=torch.float32)
    isum = torch.tensor([1200.0, 800.0, 450.0], dtype=torch.float32)
    n = rows.numel()
    el = torch.arange(1, n + 1, dtype=torch.long)
    z = torch.zeros(n, dtype=torch.float32)
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
        background_sum=z,
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
        fid, indexed, n_peaks, scale, num_refl = conn.execute(
            "SELECT frame_id, indexed, n_peaks, scale, num_reflections FROM frames"
        ).fetchone()
        assert indexed is True
        assert n_peaks == 4
        assert scale == pytest.approx(0.97)
        # sigma <= 0 reflection dropped -> only two of the three indexed remain
        assert num_refl == 2

        # the per-lattice statistics live on the crystals row
        cid, n_indexed, rmsd, crystal_refl = conn.execute(
            "SELECT crystal_id, n_indexed, rmsd, num_reflections FROM crystals"
        ).fetchone()
        assert n_indexed == 3
        assert rmsd == pytest.approx(0.0042)
        assert crystal_refl == 2

        # recovered cell recorded in Angstroms, matching B_to_cell
        recovered = B_to_cell(result.A)
        cell_a = conn.execute("SELECT cell_a_A FROM crystals").fetchone()[0]
        assert cell_a == pytest.approx(recovered.a)

        # reflections join back to both the crystal and the frame
        rows = conn.execute(
            "SELECT r.h, r.k, r.l, r.sigma FROM reflections r "
            "JOIN crystals c ON c.crystal_id = r.crystal_id "
            "JOIN frames f ON f.frame_id = r.frame_id "
            "WHERE f.frame_id = ? AND c.crystal_id = ?",
            [fid, cid],
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
        # the frame's peak_resolution is that of its highest-resolution peak
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
            "SELECT frame_index, event, num_reflections FROM frames "
            "WHERE NOT indexed ORDER BY frame_index"
        ).fetchall()
        assert [(r[0], r[1]) for r in rows] == [(1, 1), (2, 2)]
        assert all(r[2] is None for r in rows)
        # and they own no crystals
        assert conn.execute("SELECT COUNT(*) FROM crystals").fetchone()[0] == 1
        # a processed image with no lattice reports null stats, like a backfill
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM frames WHERE NOT indexed AND num_reflections IS NULL"
            ).fetchone()[0]
            == 2
        )

        # the indexed frame_id matches the deterministic key for its file-event
        fid = conn.execute("SELECT frame_id FROM frames WHERE indexed").fetchone()[0]
        assert fid == frame_id("a.h5", 0)
    finally:
        conn.close()


def test_merge_dbs_unions_blocks_and_dedups_metadata(geometry_dict, cell, tmp_path):
    from probixi.tools.multigpu import merge_dbs

    files = {"a.h5": H5Info("a.h5", "/d", n_frames=4, frame_shape=(8, 8))}
    # rank 0 owns events [0, 2), indexes event 0; rank 1 owns [2, 4), indexes 2
    part0 = tmp_path / "out.duckdb.rank0"
    part1 = tmp_path / "out.duckdb.rank1"
    for rank, part, lo in ((0, part0, 0), (1, part1, 2)):
        with DuckDBOffloader(
            part,
            geometry=geometry_dict,
            cell=cell,
            files=files,
            frame_range=(lo, lo + 2),
            calibration=dict(gain=1.0 + rank),
            rank=rank,
        ) as off:
            off.write(_make_index_result(cell, frame_index=lo))

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
        gains = conn.execute(
            "SELECT list(gain ORDER BY frame_index) FROM frames JOIN calibration USING (rank)"
        ).fetchone()[0]
        assert gains == [1.0, 1.0, 2.0, 2.0]
        # reflections from both indexed frames present and joinable
        joined = conn.execute(
            "SELECT COUNT(*) FROM reflections r JOIN frames f USING (frame_id)"
        ).fetchone()[0]
        assert joined == conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
        assert joined > 0
        serials = conn.execute(
            "SELECT list(serial ORDER BY frame_index) FROM frames WHERE serial IS NOT NULL"
        ).fetchone()[0]
        assert serials == [1, 2]
    finally:
        conn.close()


def test_merge_dbs_skips_missing_parts(geometry_dict, cell, tmp_path):
    from probixi.tools.multigpu import merge_dbs

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


def test_multilattice_frame_identity_and_merge(geometry_dict, cell, tmp_path):
    from probixi.indexer import FrameIndexResult
    from probixi.io.writer import DataOffloader
    from probixi.tools.multigpu import merge_dbs

    files = {"selection": H5Info("a.h5", "/d", 2, (8, 8), event_start=12)}
    crystals = [_make_index_result(cell), _make_index_result(cell)]
    frame = FrameIndexResult(
        0, crystals, crystals[0].positions, crystals[0].intensities
    )
    empty = FrameIndexResult(1, [], frame.positions[:0], frame.intensities[:0])
    path = tmp_path / "multi.duckdb"
    with DuckDBOffloader(path, geometry_dict, files=files) as out:
        out.write(frame)
        out.write(empty)
    merged = tmp_path / "merged.duckdb"
    assert merge_dbs([path], merged) == 1
    with duckdb.connect(str(merged), read_only=True) as conn:
        assert conn.execute(
            "SELECT event, indexed FROM frames ORDER BY frame_index"
        ).fetchall() == [(12, True), (13, False)]
        assert conn.execute(
            "SELECT count(*), count(DISTINCT crystal_id), count(DISTINCT frame_id) FROM crystals"
        ).fetchone() == (2, 2, 1)
        assert conn.execute(
            "SELECT count(*) FROM reflections r JOIN crystals c USING(crystal_id) WHERE r.frame_id=c.frame_id"
        ).fetchone() == (4,)
        assert conn.execute("SELECT count(*) FROM peaks").fetchone() == (4,)
    stream = tmp_path / "multi.stream"
    with DataOffloader(stream, geometry_dict, files=files) as out:
        out.write(frame)
        out.write(empty)
    text = stream.read_text()
    assert text.count("----- Begin chunk -----") == 2
    assert text.count("--- Begin crystal") == 2
    assert "Event: //12" in text and "Event: //13" in text
    assert "indexed_by = none" in text


def test_db_records_integration_recipe_and_model_background(
    geometry_dict, cell, tmp_path
):
    out = tmp_path / "out.duckdb"
    result = _make_index_result(cell)
    result.predicted_hkl = torch.tensor([[1, 0, 0], [0, 1, -1]], dtype=torch.long)
    result.predicted_positions = torch.tensor([[40.0, 55.0], [70.0, 30.0]])
    result.predicted_intensities = torch.tensor([1200.0, 800.0])
    result.predicted_sigmas = torch.tensor([35.0, 28.0])
    result.predicted_peak = torch.tensor([300.0, 200.0])
    result.predicted_background = torch.tensor([2.5, 2.25])
    result.predicted_n_pixels = torch.tensor([9.0, 8.0])
    result.predicted_bg_model = torch.tensor([22.5, 18.0])
    result.predicted_bg_model_var = torch.tensor([0.81, 0.64])
    recipe = dict(
        radii=(1.5, 6.0, 8.6), adu_per_photon=9.9, bg_annulus_pixels=280, aperture="snr"
    )
    with DuckDBOffloader(
        out, geometry=geometry_dict, cell=cell, integration=recipe
    ) as off:
        off.write(result)

    conn = duckdb.connect(str(out), read_only=True)
    try:
        row = conn.execute("SELECT * FROM integration").fetchone()
        assert row == (1.5, 6.0, 8.6, 9.9, 280.0, "snr", None, None, None, None, None)
        rows = conn.execute(
            "SELECT n_pixels, background_model, background_model_var FROM reflections "
            "ORDER BY h DESC"
        ).fetchall()
        assert rows == [(9, 22.5, pytest.approx(0.81)), (8, 18.0, pytest.approx(0.64))]
    finally:
        conn.close()


def test_db_reflections_without_prediction_leave_model_columns_null(
    geometry_dict, cell, tmp_path
):
    out = tmp_path / "out.duckdb"
    with DuckDBOffloader(out, geometry=geometry_dict, cell=cell) as off:
        off.write(_make_index_result(cell))
    conn = duckdb.connect(str(out), read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM integration").fetchone()[0] == 0
        rows = conn.execute(
            "SELECT n_pixels, background_model FROM reflections"
        ).fetchall()
        assert rows and all(r == (None, None) for r in rows)
    finally:
        conn.close()


def test_db_keeps_flagged_and_unmeasured_reflections(geometry_dict, cell, tmp_path):
    out = tmp_path / "out.duckdb"
    result = _make_index_result(cell)
    result.predicted_hkl = torch.tensor(
        [[1, 0, 0], [0, 1, -1], [2, 0, 0]], dtype=torch.long
    )
    result.predicted_positions = torch.tensor(
        [[40.0, 55.0], [70.0, 30.0], [10.0, 10.0]]
    )
    result.predicted_intensities = torch.tensor([1200.0, 800.0, 0.0])
    # the third disk had no usable pixel: sigma 0, fully masked
    result.predicted_sigmas = torch.tensor([35.0, 28.0, 0.0])
    result.predicted_peak = torch.tensor([300.0, 200.0, 0.0])
    result.predicted_background = torch.tensor([2.5, 2.25, 0.0])
    result.predicted_n_pixels = torch.tensor([9.0, 8.0, 0.0])
    result.predicted_bg_model = torch.tensor([22.5, 18.0, 0.0])
    result.predicted_bg_model_var = torch.tensor([0.81, 0.64, 0.0])
    result.predicted_n_pixels_total = torch.tensor([9, 9, 9])
    result.predicted_n_masked = torch.tensor([0, 1, 9])
    result.predicted_n_overlap = torch.tensor([0, 3, 0])
    result.peak_n_masked = torch.tensor([0.0, 2.0, 0.0, 0.0])
    with DuckDBOffloader(out, geometry=geometry_dict, cell=cell) as off:
        off.write(result)
    conn = duckdb.connect(str(out), read_only=True)
    try:
        rows = conn.execute(
            "SELECT h, sigma, n_pixels, n_pixels_total, n_masked, n_overlap "
            "FROM reflections ORDER BY h"
        ).fetchall()
        assert rows == [
            (0, 28.0, 8, 9, 1, 3),
            (1, 35.0, 9, 9, 0, 0),
            (2, 0.0, 0, 9, 9, 0),
        ]
        assert conn.execute("SELECT num_reflections FROM crystals").fetchone()[0] == 3
        assert conn.execute("SELECT MAX(n_masked) FROM peaks").fetchone()[0] == 2
    finally:
        conn.close()


def test_db_records_calibration(geometry_dict, cell, tmp_path):
    """The calibration provenance row, resolved on exit."""
    from probixi.io.db import SCHEMA_VERSION

    out = tmp_path / "out.duckdb"
    cal = dict(
        n_seed_frames=100,
        gain=1.1,
        read_var=0.5,
        flux_variance=True,
        background_fluctuation=3.2,
        mf_threshold=5.9,
        target_noise_peaks=5.0,
        tail_ratio=10.2,
        n_quiet_frames=0,
        kappa=2.3,
        prior_peak=0.03,
        var_scale=0.96,
        aperture="snr",
        clen=0.1,
        cell=(61.4, 141.6, 94.4, 90.0, 90.0, 90.0),
    )
    with DuckDBOffloader(
        out,
        geometry=geometry_dict,
        cell=cell,
        calibration=lambda: cal,
    ) as off:
        off.write(_make_index_result(cell))
    conn = duckdb.connect(str(out), read_only=True)
    try:
        row = conn.execute(
            "SELECT schema_version, gain, flux_variance, target_noise_peaks, tail_ratio, aperture, cell_b_A FROM calibration"
        ).fetchone()
        assert row == (SCHEMA_VERSION, 1.1, True, 5.0, 10.2, "snr", 141.6)
        assert "shells" not in _tables(conn)
    finally:
        conn.close()


def test_db_writes_a_frame_without_peaks(geometry_dict, cell, tmp_path):
    """A hit with every peak filtered away still gets its frames row."""
    from probixi.indexer.indexer import FrameIndexResult

    out = tmp_path / "out.duckdb"
    empty = FrameIndexResult(0, [], torch.zeros((0, 2)), torch.zeros(0))
    with DuckDBOffloader(out, geometry=geometry_dict, cell=cell) as off:
        off.write(empty)
    conn = duckdb.connect(str(out), read_only=True)
    try:
        assert conn.execute("SELECT indexed, n_peaks FROM frames").fetchall() == [
            (False, 0)
        ]
    finally:
        conn.close()


def _model_var_db(
    path, n=20_000, k_true=0.5, gain=2.0, n_bg=280.0, seed=0, n_fine=0, n_false=0
):
    """Write background-only reflections whose model variance is 1/k_true times their scatter.

    Crystal c0 (limit 3.0 nm^-1) holds the even rows at 3.1 nm^-1 and c1 (limit
    3.2 nm^-1) the odd rows. Both pass the enrichment test, and the run's
    resolution is 3.1458 A. The first 10 rows have masked pixels. ``n_fine``
    more c0 rows at 4.0 nm^-1 scatter 3 times their model variance. ``n_false``
    more rows at 3.1 nm^-1 lie on x0 (limit 3.0 nm^-1), which fails the
    enrichment test, and scatter 0.01 k_true times their model variance.
    """
    from probixi.io.db import _SCHEMA

    rng = np.random.default_rng(seed)
    npix = rng.integers(20, 60, n)
    var = rng.uniform(1e4, 3e4, n)
    vb = var * (1 + n_bg / npix)
    im = rng.normal(0.0, np.sqrt(k_true * vb))
    bm = rng.uniform(100.0, 200.0, n)
    crystal = np.where(np.arange(n) % 2 == 0, "c0", "c1").astype(object)
    res = np.full(n, 3.1)
    for m, cid, r, k in ((n_fine, "c0", 4.0, 3.0), (n_false, "x0", 3.1, 0.01 * k_true)):
        if m:
            npix, var, bm = (np.concatenate([a, a[:m]]) for a in (npix, var, bm))
            im = np.concatenate([im, rng.normal(0.0, np.sqrt(k * vb[:m]))])
            crystal = np.concatenate([crystal, np.full(m, cid, dtype=object)])
            res = np.concatenate([res, np.full(m, r)])
    bg = 5.0
    rows = dict(
        crystal_id=crystal,
        frame_id=np.full(im.size, "f0", dtype=object),
        intensity=im - npix * bg + bm,
        sigma=np.ones(im.size),
        background=np.full(im.size, bg),
        resolution_nm_inv=res,
        n_pixels=npix,
        background_model=bm,
        background_model_var=var,
        n_masked=np.where(np.arange(im.size) < 10, 3, 0),
        n_overlap=np.zeros(im.size, dtype=np.int64),
    )
    conn = duckdb.connect(str(path))
    conn.execute(_SCHEMA)
    conn.execute(
        "INSERT INTO integration (r_peak, r_gap, r_bg, adu_per_photon, "
        "bg_annulus_pixels, aperture) VALUES (1.5, 4.0, 7.4, ?, ?, 'snr')",
        [gain, n_bg],
    )
    conn.execute(
        "INSERT INTO crystals (crystal_id, frame_id, diffraction_limit_nm_inv, enrich_p) "
        "VALUES ('c0', 'f0', 3.0, 1e-6), ('c1', 'f0', 3.2, 1e-6)"
    )
    if n_false:
        conn.execute(
            "INSERT INTO crystals (crystal_id, frame_id, diffraction_limit_nm_inv, "
            "enrich_p) VALUES ('x0', 'f0', 3.0, 0.5)"
        )
    cols = ", ".join(rows)
    conn.execute(f"INSERT INTO reflections ({cols}) SELECT {cols} FROM rows")
    return conn, rows


def test_fit_model_var_scale_recovers_a_known_factor():
    from probixi.io.db import fit_model_var_scale

    rng = np.random.default_rng(1)
    vb = rng.uniform(5e3, 1.5e4, 100_000)
    im = rng.normal(0.0, np.sqrt(0.4 * vb))
    im[:300] += 1e6
    assert fit_model_var_scale(im, vb, gain=1.0) == pytest.approx(0.4, rel=0.03)
    assert fit_model_var_scale(im[:10], vb[:10], gain=1.0) is None


def test_calibrate_model_variance_rescales_once(tmp_path):
    from probixi.io.db import calibrate_model_variance

    conn, rows = _model_var_db(tmp_path / "run.duckdb", k_true=0.5)
    try:
        k = calibrate_model_variance(conn)
        assert k == pytest.approx(0.5, rel=0.05)
        scale, d_min = conn.execute(
            "SELECT bg_model_var_scale, bg_model_var_dmin_A FROM integration"
        ).fetchone()
        assert scale == pytest.approx(k)
        assert d_min == pytest.approx(10 / 3.2 + 0.1 * (10 / 3.0 - 10 / 3.2))
        got = conn.execute(
            "SELECT background_model_var FROM reflections ORDER BY rowid"
        ).fetchnumpy()["background_model_var"]
        np.testing.assert_allclose(got, rows["background_model_var"] * k)
        vb = got * (1 + 280.0 / rows["n_pixels"])
        im = (
            rows["intensity"]
            + rows["n_pixels"] * rows["background"]
            - rows["background_model"]
        )
        z = im / np.sqrt(vb + 2.0 * np.clip(im, 0, None))
        assert 1.4826 * np.median(np.abs(z - np.median(z))) == pytest.approx(
            1.0, abs=0.03
        )
        assert conn.execute(
            "SELECT min(sigma), max(sigma) FROM reflections"
        ).fetchone() == (
            1.0,
            1.0,
        )
        assert calibrate_model_variance(conn) == pytest.approx(k)
        again = conn.execute(
            "SELECT background_model_var FROM reflections ORDER BY rowid"
        ).fetchnumpy()["background_model_var"]
        np.testing.assert_array_equal(again, got)
    finally:
        conn.close()


def test_calibrate_model_variance_skips_without_reference(tmp_path):
    from probixi.io.db import calibrate_model_variance

    conn, rows = _model_var_db(tmp_path / "run.duckdb")
    try:
        conn.execute("UPDATE crystals SET diffraction_limit_nm_inv = 9.0")
        assert calibrate_model_variance(conn) is None
        assert conn.execute(
            "SELECT bg_model_var_scale FROM integration"
        ).fetchone() == (None,)
        got = conn.execute(
            "SELECT background_model_var FROM reflections ORDER BY rowid"
        ).fetchnumpy()["background_model_var"]
        np.testing.assert_array_equal(got, rows["background_model_var"])
        conn.execute("UPDATE crystals SET diffraction_limit_nm_inv = 3.0")
        conn.execute("UPDATE integration SET bg_annulus_pixels = NULL")
        assert calibrate_model_variance(conn) is None
    finally:
        conn.close()


def test_merge_dbs_fits_one_model_variance_scale(tmp_path):
    from probixi.io.db import _SCHEMA, calibrate_model_variance
    from probixi.tools.multigpu import merge_dbs

    parts = []
    for rank, k_true in enumerate((0.4, 0.6)):
        conn, _ = _model_var_db(
            tmp_path / f"run.rank{rank}", k_true=k_true, seed=rank, n_false=20_000
        )
        conn.execute(f"UPDATE reflections SET crystal_id = crystal_id || '_{rank}'")
        conn.execute(f"UPDATE crystals SET crystal_id = crystal_id || '_{rank}'")
        conn.close()
        parts.append(tmp_path / f"run.rank{rank}")
    merge_dbs(parts, tmp_path / "run.duckdb")
    with duckdb.connect(str(tmp_path / "run.duckdb"), read_only=True) as conn:
        (k,) = conn.execute("SELECT bg_model_var_scale FROM integration").fetchone()
    assert 0.4 < k < 0.6
    with duckdb.connect(str(tmp_path / "union.duckdb")) as conn:
        conn.execute(_SCHEMA)
        for i, part in enumerate(parts):
            conn.execute(f"ATTACH '{part}' AS p{i} (READ_ONLY)")
            for tbl in ("integration",) * (i == 0) + ("crystals", "reflections"):
                conn.execute(f"INSERT INTO {tbl} SELECT * FROM p{i}.{tbl}")
        assert calibrate_model_variance(conn) == pytest.approx(k, rel=1e-9)
    with duckdb.connect(str(parts[0]), read_only=True) as conn:
        assert conn.execute(
            "SELECT bg_model_var_scale FROM integration"
        ).fetchone() == (None,)


def test_calibrate_model_variance_fits_within_the_run_resolution(tmp_path):
    from probixi.io.db import calibrate_model_variance, run_resolution_limit

    conn, _ = _model_var_db(tmp_path / "run.duckdb", k_true=0.5, n_fine=20_000)
    try:
        assert run_resolution_limit(conn) == pytest.approx(3.1458, abs=1e-3)
        assert calibrate_model_variance(conn) == pytest.approx(0.5, rel=0.05)
    finally:
        conn.close()
    conn, _ = _model_var_db(tmp_path / "wide.duckdb", k_true=0.5, n_fine=20_000)
    try:
        k_wide = calibrate_model_variance(conn, d_min=2.0)
        assert k_wide > 0.8
        assert conn.execute(
            "SELECT bg_model_var_dmin_A FROM integration"
        ).fetchone() == (2.0,)
    finally:
        conn.close()


def test_calibrate_model_variance_fits_on_lattices_that_pass_the_enrichment_test(
    tmp_path,
):
    from probixi.io.db import calibrate_model_variance

    kw = dict(n=40_000, k_true=0.6, n_false=40_000)
    conn, _ = _model_var_db(tmp_path / "run.duckdb", **kw)
    try:
        assert calibrate_model_variance(conn) == pytest.approx(0.6, rel=0.02)
        assert conn.execute(
            "SELECT bg_model_var_ref, bg_model_var_alpha, bg_model_var_n_ref "
            "FROM integration"
        ).fetchone() == ("enrich_pass", 1e-3, 19_995)
    finally:
        conn.close()
    conn, _ = _model_var_db(tmp_path / "all.duckdb", **kw)
    try:
        assert calibrate_model_variance(conn, alpha=None) < 0.1
        assert conn.execute(
            "SELECT bg_model_var_ref, bg_model_var_alpha FROM integration"
        ).fetchone() == ("all", None)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "gate",
    [
        "UPDATE crystals SET enrich_p = 0.5 WHERE crystal_id = 'c0'",
        "UPDATE crystals SET enrich_p = NULL",
    ],
)
def test_calibrate_model_variance_falls_back_to_every_lattice(tmp_path, gate):
    from probixi.io.db import calibrate_model_variance

    conn, _ = _model_var_db(tmp_path / "run.duckdb", k_true=0.5)
    try:
        conn.execute(gate)
        with pytest.warns(UserWarning, match="fitted on every lattice"):
            assert calibrate_model_variance(conn) == pytest.approx(0.5, rel=0.05)
        assert conn.execute(
            "SELECT bg_model_var_ref, bg_model_var_alpha FROM integration"
        ).fetchone() == ("all", None)
    finally:
        conn.close()


def test_run_resolution_limit_counts_every_lattice_unless_gated(tmp_path):
    from probixi.io.db import run_resolution_limit

    conn, _ = _model_var_db(tmp_path / "run.duckdb")
    try:
        conn.execute(
            "INSERT INTO crystals (crystal_id, frame_id, diffraction_limit_nm_inv, "
            "enrich_p) SELECT 'x' || i, 'f0', 5.0, 0.5 FROM range(8) t(i)"
        )
        assert run_resolution_limit(conn) == pytest.approx(2.0)
        assert run_resolution_limit(conn, alpha=1e-3) == pytest.approx(3.1458, abs=1e-3)
    finally:
        conn.close()


def test_recalibrate_model_variance_reproduces_the_gated_fit(tmp_path):
    from probixi.io.db import calibrate_model_variance, recalibrate_model_variance

    kw = dict(k_true=0.6, n_false=20_000)
    conn, rows = _model_var_db(tmp_path / "old.duckdb", **kw)
    try:
        assert calibrate_model_variance(conn, alpha=None) < 0.1
        k = recalibrate_model_variance(conn)
        got = conn.execute(
            "SELECT background_model_var FROM reflections ORDER BY rowid"
        ).fetchnumpy()["background_model_var"]
        np.testing.assert_allclose(got, rows["background_model_var"] * k, rtol=1e-12)
        ref = conn.execute(
            "SELECT bg_model_var_ref, bg_model_var_dmin_A FROM integration"
        ).fetchone()
        assert ref == ("enrich_pass", pytest.approx(3.1667, abs=1e-3))
    finally:
        conn.close()
    fresh, _ = _model_var_db(tmp_path / "fresh.duckdb", **kw)
    try:
        assert k == pytest.approx(calibrate_model_variance(fresh), rel=1e-9)
    finally:
        fresh.close()
