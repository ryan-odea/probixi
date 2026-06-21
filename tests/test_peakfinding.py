from __future__ import annotations

import numpy as np
import pytest
import sim
import torch

from probixi.peakfinding.noise.calibrate import calibrate_noise
from probixi.peakfinding.noise.model import NoiseModel
from probixi.peakfinding.peaks.blobs import (
    BlobStats,
    compute_blob_stats,
    filter_blobs,
    label_connected_components,
)
from probixi.peakfinding.peaks.neighborhood import gaussian_kernel_2d
from probixi.peakfinding.peaks.peakfinder import PeakFinder

SHAPE = (160, 160)
BG = 100.0
NS = 10.0


def _calibrated(seed: int = 100, **finder_kw) -> tuple[NoiseModel, PeakFinder]:
    # signal-free seed stack -> running model + learned blend/kappa/prior
    frames = sim.simulate_noise_frames(
        SHAPE, 30, background=BG, noise_sigma=NS, seed=seed
    )
    nm = NoiseModel(SHAPE, mode="online")
    cal = calibrate_noise(nm, [torch.from_numpy(f) for f in frames])
    finder = PeakFinder(nm, **finder_kw)
    cal.apply(nm, finder)
    return nm, finder


def _grid(rows, cols) -> np.ndarray:
    rr, cc = np.meshgrid(
        np.asarray(rows, float), np.asarray(cols, float), indexing="ij"
    )
    return np.stack([rr.ravel(), cc.ravel()], axis=1)


def _one_result(finder: PeakFinder, frame: np.ndarray):
    return next(iter(finder.peak_stream([torch.from_numpy(frame)])))


def test_recall_detects_every_injected_spot_with_subpixel_centroids():
    _, finder = _calibrated(size_max=80)
    pos = _grid([30, 80, 130], [30, 80, 130])  # 9 well-separated spots
    intensity = 6000.0
    frame = sim.render_frame(
        SHAPE, pos, [intensity] * pos.shape[0], background=BG, noise_sigma=NS, seed=3
    )
    peaks = finder.peak_stream([torch.from_numpy(frame)]).collect_peaks()

    assert len(peaks) == pos.shape[0]
    det = torch.tensor([[p.row, p.col] for p in peaks], dtype=torch.float64)
    for truth in pos:
        dist = torch.linalg.vector_norm(det - torch.tensor(truth), dim=1)
        assert float(dist.min()) < 1.0


def test_recovered_intensity_approximates_injected_intensity():
    _, finder = _calibrated(size_max=80)
    intensity = 12000.0
    frame = sim.render_frame(
        SHAPE, [[80.0, 80.0]], [intensity], background=BG, noise_sigma=NS, seed=11
    )
    peaks = finder.peak_stream([torch.from_numpy(frame)]).collect_peaks()
    assert len(peaks) == 1
    # box sum over the thresholded core under-counts the gaussian tails; bright
    # spot recovers most of the injected counts and never over-counts.
    assert peaks[0].intensity == pytest.approx(intensity, rel=0.15)


def test_signal_free_frame_yields_few_detections():
    _, finder = _calibrated(size_max=80)
    clean = sim.simulate_noise_frames(
        SHAPE, 1, background=BG, noise_sigma=NS, seed=999
    )[0]
    peaks = finder.peak_stream([torch.from_numpy(clean)]).collect_peaks()
    assert len(peaks) <= 3


def test_low_snr_spot_below_threshold_is_not_detected():
    _, finder = _calibrated(size_max=80)
    truth = np.array([[80.0, 80.0]])
    faint = sim.render_frame(
        SHAPE, truth, [60.0], background=BG, noise_sigma=NS, seed=5
    )
    peaks = finder.peak_stream([torch.from_numpy(faint)]).collect_peaks()
    det = torch.tensor([[p.row, p.col] for p in peaks], dtype=torch.float64)
    for p in peaks:
        # nothing detected at the faint spot
        assert (
            torch.linalg.vector_norm(
                torch.tensor([p.row, p.col], dtype=torch.float64)
                - torch.tensor(truth[0], dtype=torch.float64)
            )
            > 2.0
        )
    assert det.shape[0] <= 3


def _streak_stats(finder: PeakFinder) -> BlobStats:
    # a dense, tightly spaced line forms one connected, elongated blob
    rows = np.linspace(74.0, 86.0, 40)
    cols = np.full_like(rows, 80.0)
    spos = np.stack([rows, cols], axis=1)
    frame = sim.render_frame(
        SHAPE,
        spos,
        [600.0] * spos.shape[0],
        background=BG,
        noise_sigma=NS,
        seed=21,
        psf_sigma=1.0,
    )
    return _one_result(finder, frame).stats


def test_eccentricity_max_rejects_an_elongated_streak():
    # local_background off so the streak's own annulus does not inflate var_eff
    _, finder = _calibrated(
        local_background=False,
        size_min=1,
        size_max=100000,
        eccentricity_max=1000.0,
        peakedness_min=0.0,
    )
    stats = _streak_stats(finder)
    assert len(stats) == 1
    assert float(stats.eccentricity[0]) > 5.0

    permissive = dict(
        size_min=1, size_max=100000, eccentricity_max=1000.0, peakedness_min=0.0
    )
    assert int(filter_blobs(stats, **permissive).sum()) == 1
    # toggling only eccentricity_max removes the streak
    rejected = {**permissive, "eccentricity_max": 5.0}
    assert int(filter_blobs(stats, **rejected).sum()) == 0


def test_size_min_rejects_a_single_pixel_blob():
    stats = BlobStats(
        label_id=torch.tensor([1]),
        size=torch.tensor([1]),
        row_centroid=torch.tensor([5.0]),
        col_centroid=torch.tensor([5.0]),
        bbox_r0=torch.tensor([5]),
        bbox_r1=torch.tensor([6]),
        bbox_c0=torch.tensor([5]),
        bbox_c1=torch.tensor([6]),
        intensity_sum=torch.tensor([500.0]),
        intensity_sigma=torch.tensor([10.0]),
        intensity_max=torch.tensor([500.0]),
        z_max=torch.tensor([50.0]),
        log_bf_sum=torch.tensor([10.0]),
        posterior_mean=torch.tensor([0.9]),
        eccentricity=torch.tensor([1.0]),
        peakedness=torch.tensor([1.5]),
    )
    # passes every other filter; size_min=2 is the only thing that can reject it
    assert (
        int(
            filter_blobs(
                stats,
                size_min=1,
                size_max=100000,
                eccentricity_max=1000.0,
                peakedness_min=0.0,
            ).sum()
        )
        == 1
    )
    assert (
        int(
            filter_blobs(
                stats,
                size_min=2,
                size_max=100000,
                eccentricity_max=1000.0,
                peakedness_min=0.0,
            ).sum()
        )
        == 0
    )


def test_size_max_rejects_an_oversized_blob():
    # a very bright, broad spot floods well past the default size_max
    _, finder = _calibrated(
        local_background=False,
        size_min=1,
        size_max=100000,
        eccentricity_max=1000.0,
        peakedness_min=0.0,
    )
    big = sim.render_frame(
        SHAPE,
        [[80.0, 80.0]],
        [80000.0],
        background=BG,
        noise_sigma=NS,
        seed=7,
        psf_sigma=4.5,
    )
    stats = _one_result(finder, big).stats
    assert len(stats) == 1
    assert int(stats.size[0]) > 30

    permissive = dict(
        size_min=1, size_max=100000, eccentricity_max=1000.0, peakedness_min=0.0
    )
    assert int(filter_blobs(stats, **permissive).sum()) == 1
    # toggling only size_max removes the oversized blob
    rejected = {**permissive, "size_max": 30}
    assert int(filter_blobs(stats, **rejected).sum()) == 0


def test_gaussian_kernel_2d_is_normalized_odd_and_symmetric():
    k = gaussian_kernel_2d(7, 1.4, dtype=torch.float64)
    assert k.shape == (7, 7)
    assert k.shape[0] % 2 == 1
    assert float(k.sum()) == pytest.approx(1.0, abs=1e-12)
    # symmetric under both flips and its own transpose
    assert torch.allclose(k, k.flip(0))
    assert torch.allclose(k, k.flip(1))
    assert torch.allclose(k, k.t())
    # peak is at the center pixel
    assert int(k.argmax()) == (7 * 7) // 2


def test_gaussian_kernel_2d_rejects_even_size():
    with pytest.raises(ValueError):
        gaussian_kernel_2d(4, 1.0)


def test_blob_stats_centroid_recovers_isolated_spot_position():
    # a single connected blob: centroid lands on the injected sub-pixel position
    _, finder = _calibrated(size_max=80)
    truth_r, truth_c = 81.4, 79.6
    frame = sim.render_frame(
        SHAPE, [[truth_r, truth_c]], [8000.0], background=BG, noise_sigma=NS, seed=13
    )
    scores = finder.score(torch.from_numpy(frame))
    binary = (
        scores["posterior"] > finder.posterior_threshold
    ) & finder.noise.valid_mask
    labels, n = label_connected_components(binary, connectivity=finder.connectivity)
    assert n == 1
    stats = compute_blob_stats(
        labels,
        n,
        excess=scores["excess"],
        z=scores["z"],
        log_bf=scores["log_bf"],
        posterior=scores["posterior"],
        var=scores["var_eff"],
    )
    assert float(stats.row_centroid[0]) == pytest.approx(truth_r, abs=1.0)
    assert float(stats.col_centroid[0]) == pytest.approx(truth_c, abs=1.0)
