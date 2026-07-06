from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import sim
import torch
from click.testing import CliRunner

import probixi.cli as cli
from probixi.indexer import IndexStream
from probixi.indexer.indexer import IndexResult, RefineConfig, SeedConfig
from probixi.io import CellParams, read_crystfel_cell, read_geometry
from probixi.io.geometry import EV_ANGSTROM
from probixi.peakfinding import PeakStream
from probixi.peakfinding.noise import CalibrationResult, ThresholdCalibration
from probixi.peakfinding.peaks import PeakResult
from probixi.probixi import Probixi

# small single-panel detector: enough on-frame bR reflections to index, tiny on CPU
DET = 256
CLEN = 0.10
PIXEL_SIZE = 150e-6
WAVELENGTH = 1.3
N_NOISE = 8  # seed/calibration frames (indices 0..N_NOISE-1)
N_PLANTED = 2  # planted indexable frames following the noise frames
# planted spots are dim + compact so blobs stay under the finder's size_max
PEAK_INTENSITY = 350.0
PSF_SIGMA = 1.0
CELL_FIXTURE = Path(__file__).parent / "fixtures" / "bR.cell"

# narrow seed/refine search so a full index is ~1s per frame on CPU
SEED = SeedConfig(n_directions=1500, n_spin=60, top_directions=12, max_candidates=32)
REFINE = RefineConfig(max_iters=150, reassign_every=10)


def _geom_text() -> str:
    # corner_{x,y} put the beam at the frame center: read_geometry derives
    # beam_center=(-corner_y,-corner_x); res=1/pixel_size; wl from photon_energy.
    bc = (DET - 1) / 2.0
    return (
        f"clen = {CLEN}\n"
        f"photon_energy = {EV_ANGSTROM / WAVELENGTH}\n"
        f"res = {1.0 / PIXEL_SIZE}\n"
        "max_adu = 65535\n"
        "data = /entry/data/data\n"
        "0/min_fs = 0\n"
        "0/min_ss = 0\n"
        f"0/max_fs = {DET - 1}\n"
        f"0/max_ss = {DET - 1}\n"
        f"0/corner_x = {-bc}\n"
        f"0/corner_y = {-bc}\n"
        "0/fs = +1.0x +0.0y\n"
        "0/ss = +0.0x +1.0y\n"
    )


@pytest.fixture(scope="module")
def geom_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pipeline") / "sim.geom"
    path.write_text(_geom_text(), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def pipeline_cell() -> CellParams:
    return read_crystfel_cell(CELL_FIXTURE)


@pytest.fixture(scope="module")
def planted_U() -> torch.Tensor:
    # near-identity rotation so coarse seeding recovers it quickly
    return sim.proper_rotation(0, max_angle_deg=8)


@pytest.fixture(scope="module")
def run_files(tmp_path_factory, geom_path, pipeline_cell, planted_U):
    # noise frames first (calibration), then N_PLANTED copies of one planted frame
    geom = read_geometry(geom_path).to_dict()
    frame, truth = sim.simulate_indexable_frame(
        geom,
        pipeline_cell,
        planted_U,
        peak_intensity=PEAK_INTENSITY,
        psf_sigma=PSF_SIGMA,
        seed=1,
    )
    noise = sim.simulate_noise_frames((DET, DET), N_NOISE, seed=100)
    planted = np.repeat(frame[None], N_PLANTED, axis=0)
    frames = np.concatenate([noise, planted], axis=0)
    h5_path, lst_path = sim.write_run(tmp_path_factory.mktemp("run"), frames)
    return lst_path, h5_path, truth


def _build_pipeline(lst_path, geom_path, cell_file=CELL_FIXTURE) -> Probixi:
    return Probixi(
        list_file=lst_path,
        geometry_file=geom_path,
        cell_file=cell_file,
        seed=SEED,
        refine=REFINE,
        warmup_frames=4,
    )


def _calibrated_pipeline(lst_path, geom_path, cell_file=CELL_FIXTURE) -> Probixi:
    px = _build_pipeline(lst_path, geom_path, cell_file)
    px.calibrate(n_seed=N_NOISE, target_noise_peaks=5.0)
    return px


def test_calibrate_returns_result_and_sets_threshold_calibration(run_files, geom_path):
    lst_path, _, _ = run_files
    px = _build_pipeline(lst_path, geom_path)
    assert px.threshold_calibration is None
    result = px.calibrate(n_seed=N_NOISE, target_noise_peaks=5.0)
    assert isinstance(result, CalibrationResult)
    assert result.kappa > 1.0
    assert 0.0 < result.prior_peak < 1.0
    assert result.var_scale > 0.0
    assert isinstance(px.threshold_calibration, ThresholdCalibration)
    # calibrating installs the learned matched-filter threshold on the finder
    assert px.finder.mf_threshold == pytest.approx(px.threshold_calibration.threshold)


def test_calibrate_without_threshold_leaves_calibration_unset(run_files, geom_path):
    lst_path, _, _ = run_files
    px = _build_pipeline(lst_path, geom_path)
    result = px.calibrate(n_seed=N_NOISE, target_noise_peaks=None)
    assert isinstance(result, CalibrationResult)
    assert px.threshold_calibration is None


def test_peak_stream_yields_peak_results_with_detections(run_files, geom_path):
    lst_path, _, truth = run_files
    px = _calibrated_pipeline(lst_path, geom_path)
    stream = px.peak_stream(
        px.frames(start=N_NOISE, stop=N_NOISE + N_PLANTED),
        start_index=N_NOISE,
        estimate_scale=False,
    )
    assert isinstance(stream, PeakStream)
    results = list(stream)
    assert len(results) == N_PLANTED
    assert all(isinstance(r, PeakResult) for r in results)
    assert all(r.frame_index is not None for r in results)
    # a planted lattice frame yields many real peaks
    assert all(len(r) > 20 for r in results)


def test_peak_stream_recovers_planted_positions(run_files, geom_path):
    lst_path, _, truth = run_files
    px = _calibrated_pipeline(lst_path, geom_path)
    peaks = px.peak_stream(
        px.frames(start=N_NOISE, stop=N_NOISE + 1),
        start_index=N_NOISE,
        estimate_scale=False,
    ).collect_peaks()
    det = torch.tensor([[p.row, p.col] for p in peaks], dtype=torch.float64)
    assert det.shape[0] > 0
    matched = 0
    for target in truth.positions:
        if float(torch.linalg.vector_norm(det - target, dim=1).min()) < 2.0:
            matched += 1
    # most planted spots are recovered
    assert matched >= int(0.7 * truth.positions.shape[0])


def test_index_stream_yields_index_results_recovering_cell(
    run_files, geom_path, pipeline_cell
):
    lst_path, _, truth = run_files
    px = _calibrated_pipeline(lst_path, geom_path)
    stream = px.index_stream(
        px.frames(start=N_NOISE, stop=N_NOISE + N_PLANTED),
        batch_size=N_PLANTED,
        start_index=N_NOISE,
    )
    assert isinstance(stream, IndexStream)
    results = stream.collect()
    assert len(results) == N_PLANTED
    tgt_edges = sorted([pipeline_cell.a, pipeline_cell.b, pipeline_cell.c])
    for r in results:
        assert isinstance(r, IndexResult)
        assert r.frame_index in (N_NOISE, N_NOISE + 1)
        assert r.n_peaks > 0
        assert 0 < r.n_indexed <= r.n_peaks
        assert r.rmsd < 0.01
        assert r.A.shape == (3, 3)
        assert r.U.shape == (3, 3)
        assert torch.isfinite(r.A).all()
        # the recovered cell matches the planted bR target (sorted edges)
        edges = sorted([r.cell.a, r.cell.b, r.cell.c])
        assert all(abs(o - t) / t < 0.02 for o, t in zip(edges, tgt_edges))
        # merge-ready stream predicts + integrates the full lattice
        assert r.predicted_hkl is not None
        assert r.predicted_positions.shape[0] == r.predicted_hkl.shape[0]


def test_index_stream_stats_track_frames_hits_and_indexed(run_files, geom_path):
    lst_path, _, _ = run_files
    px = _calibrated_pipeline(lst_path, geom_path)
    stream = px.index_stream(
        px.frames(start=N_NOISE, stop=N_NOISE + N_PLANTED),
        batch_size=N_PLANTED,
        start_index=N_NOISE,
    )
    # funnel starts empty; the tap wrapper in Probixi.index_stream must not drop it
    assert stream.stats.frames == 0
    n = len(stream.collect())
    # every planted frame is a hit (many peaks) and indexes here
    assert stream.stats.frames == N_PLANTED
    assert stream.stats.hits == N_PLANTED
    assert stream.stats.indexed == n == N_PLANTED
    assert stream.stats.hit_rate == pytest.approx(1.0)
    assert stream.stats.index_rate_of_hits == pytest.approx(1.0)


def test_index_stream_requires_cell(run_files, geom_path):
    lst_path, _, _ = run_files
    px = _build_pipeline(lst_path, geom_path, cell_file=None)
    assert px.indexer is None
    with pytest.raises(RuntimeError):
        px.index_stream(px.frames(start=N_NOISE, stop=N_NOISE + 1)).collect()


def test_cli_indexing_run_writes_nonempty_stream(tmp_path, run_files, geom_path):
    lst_path, _, _ = run_files
    out = tmp_path / "indexed.stream"
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "-i",
            str(lst_path),
            "-g",
            str(geom_path),
            "-p",
            str(CELL_FIXTURE),
            "-o",
            str(out),
            "--seed-frames",
            str(N_NOISE),
            "--warmup-frames",
            "4",
            "--start",
            str(N_NOISE),
            "--stop",
            str(N_NOISE + N_PLANTED),
            "--batch-size",
            str(N_PLANTED),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    text = out.read_text()
    assert len(text) > 0
    assert "CrystFEL stream format" in text
    assert "--- Begin crystal" in text
    # a CrystFEL-recognised method so partialator doesn't reject the indexer list
    assert "indexed_by = fromfile" in text
    assert "Generated by probixi" in text


def test_cli_peaks_only_without_cell_writes_cxi_dir(tmp_path, run_files, geom_path):
    lst_path, _, _ = run_files
    out = tmp_path / "peaks_out"
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "-i",
            str(lst_path),
            "-g",
            str(geom_path),
            "-o",
            str(out),
            "--peaks-only",
            "--seed-frames",
            str(N_NOISE),
            "--warmup-frames",
            "4",
            "--start",
            str(N_NOISE),
            "--stop",
            str(N_NOISE + N_PLANTED),
        ],
    )
    assert result.exit_code == 0, result.output
    # --peaks-only exports a CXI peak set directory for 'indexamajig --peaks=cxi'
    assert out.is_dir()
    cxi_files = sorted(out.glob("*.cxi"))
    assert len(cxi_files) == 1
    cxi = cxi_files[0]
    lst = out / "peaks.lst"
    assert lst.read_text().strip() == str(cxi.resolve())
    # companion geometry carries the CXI peak-list keys
    geom_out = out / (geom_path.stem + "_cxi.geom")
    geom_text = geom_out.read_text()
    assert "peak_list = /entry_1/result_1" in geom_text
    assert "peak_list_type = cxi" in geom_text
    with h5py.File(cxi, "r") as f:
        npeaks = f["/entry_1/result_1/nPeaks"][:]
    # the planted frames contribute peaks
    assert int(npeaks.sum()) > 0


def test_cli_missing_cell_without_peaks_only_fails(tmp_path, run_files, geom_path):
    lst_path, _, _ = run_files
    out = tmp_path / "fail.stream"
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "-i",
            str(lst_path),
            "-g",
            str(geom_path),
            "-o",
            str(out),
            "--seed-frames",
            str(N_NOISE),
        ],
    )
    assert result.exit_code != 0
    import click

    assert isinstance(result.exception, (click.UsageError, SystemExit))
    assert "unit cell" in result.output
