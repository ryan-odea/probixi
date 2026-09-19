from __future__ import annotations

import math

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
    footprint_cap,
    label_connected_components,
)
from probixi.peakfinding.peaks.neighborhood import gaussian_kernel_2d
from probixi.peakfinding.peaks.peakfinder import PeakFinder
from probixi.probixi import (
    _BEAMSTOP_MAX_EXTEND_BINS,
    _beamstop_qmin_from_histogram,
)

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
    det = torch.tensor([[p.row, p.col] for p in peaks], dtype=torch.float32)
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
    det = torch.tensor([[p.row, p.col] for p in peaks], dtype=torch.float32)
    for p in peaks:
        # nothing detected at the faint spot
        assert (
            torch.linalg.vector_norm(
                torch.tensor([p.row, p.col], dtype=torch.float32)
                - torch.tensor(truth[0], dtype=torch.float32)
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
        background_sum=torch.tensor([0.0]),
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
    k = gaussian_kernel_2d(7, 1.4, dtype=torch.float32)
    assert k.shape == (7, 7)
    assert k.shape[0] % 2 == 1
    assert float(k.sum()) == pytest.approx(1.0, abs=1e-6)
    # symmetric under both flips and its own transpose
    assert torch.allclose(k, k.flip(0))
    assert torch.allclose(k, k.flip(1))
    assert torch.allclose(k, k.t())
    # peak is at the center pixel
    assert int(k.argmax()) == (7 * 7) // 2


def test_gaussian_kernel_2d_rejects_even_size():
    with pytest.raises(ValueError):
        gaussian_kernel_2d(4, 1.0)


def test_gaussian_kernel_1d_is_outer_product_factor():
    from probixi.peakfinding.peaks.neighborhood import gaussian_kernel_1d

    for size, sigma in [(5, 1.0), (7, 1.4), (15, 2.4)]:
        a = gaussian_kernel_1d(size, sigma, dtype=torch.float32)
        k2 = gaussian_kernel_2d(size, sigma, dtype=torch.float32)
        assert torch.allclose(torch.outer(a, a), k2, atol=1e-6)
        assert float(a.sum()) == pytest.approx(1.0, abs=1e-6)


def test_separable_convs_match_dense_conv2d():
    # The separable (2x 1D) matched filter / smoothing must reproduce the dense
    # 2D convolution they replaced, including reflect-padded corners.
    import torch.nn.functional as F

    from probixi.peakfinding.peaks.neighborhood import (
        mask_denominator,
        matched_filter_z,
        smooth_logits,
        smooth_logits_batch,
    )

    torch.manual_seed(0)
    H, W = 137, 151
    z = torch.randn(H, W, dtype=torch.float32)
    mask = torch.rand(H, W) > 0.1
    m = mask.to(torch.float32)

    for size, sigma in [(7, 1.0), (11, 1.6), (15, 2.4)]:
        k2 = gaussian_kernel_2d(size, sigma, dtype=torch.float32)
        u = k2 / k2.norm()
        pad = (size // 2,) * 4
        num = F.conv2d(
            F.pad((z * m).view(1, 1, H, W), pad), u.view(1, 1, size, size)
        ).reshape(H, W)
        den = F.conv2d(
            F.pad(m.view(1, 1, H, W), pad), (u * u).view(1, 1, size, size)
        ).reshape(H, W)
        ref = num / den.clamp_min(1e-12).sqrt()
        assert torch.allclose(ref, matched_filter_z(z, k2, mask, den=den), atol=1e-6)

    k2 = gaussian_kernel_2d(5, 1.0, dtype=torch.float32)
    den = mask_denominator(m, k2)
    ref = (
        F.conv2d(
            F.pad((z * m).view(1, 1, H, W), (2, 2, 2, 2), mode="reflect"),
            k2.view(1, 1, 5, 5),
        ).reshape(H, W)
        / den
    )
    assert torch.allclose(ref, smooth_logits(z, k2, mask=mask, den=den), atol=1e-6)
    zb = torch.randn(3, H, W, dtype=torch.float32)
    refb = torch.stack(
        [
            F.conv2d(
                F.pad((zb[i] * m).view(1, 1, H, W), (2, 2, 2, 2), mode="reflect"),
                k2.view(1, 1, 5, 5),
            ).reshape(H, W)
            / den
            for i in range(3)
        ]
    )
    assert torch.allclose(
        refb, smooth_logits_batch(zb, k2, mask=mask, den=den), atol=1e-6
    )


def test_integral_box_sum_matches_conv_reference():
    # The summed-area-table box sum must reproduce the zero-padded sliding-window
    # sum it replaced, including clamped edge boxes, for both single and batched
    # inputs and across a DC offset (which stresses float32 cumsum precision).
    import torch.nn.functional as F

    from probixi.peakfinding.peaks.neighborhood import _box_sum

    def conv_box(x, radius):
        if radius < 1:
            return x.clone()
        xb = x.view(1, 1, *x.shape)
        k = 2 * radius + 1
        xb = F.conv2d(F.pad(xb, (0, 0, radius, radius)), x.new_ones(1, 1, k, 1))
        xb = F.conv2d(F.pad(xb, (radius, radius, 0, 0)), x.new_ones(1, 1, 1, k))
        return xb.reshape(x.shape)

    torch.manual_seed(1)
    for offset in (0.0, 25.0):
        x = torch.randn(97, 103) * 10.0 + offset
        for radius in (1, 4, 9):
            assert torch.allclose(_box_sum(x, radius), conv_box(x, radius), atol=5e-2)
    xb = torch.randn(4, 60, 55) * 10.0 + 8.0
    for radius in (2, 5):
        ref = torch.stack([conv_box(xb[i], radius) for i in range(xb.shape[0])])
        assert torch.allclose(_box_sum(xb, radius), ref, atol=5e-2)


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


def test_compute_blob_stats_every_field_matches_per_blob_reference():
    # Guards the fused (batched-scatter) reductions: each field is checked against a
    # straightforward per-blob loop, so a mixed-up column would be caught.
    torch.manual_seed(0)
    H = W = 12
    labels = torch.zeros(H, W, dtype=torch.long)
    labels[2:4, 2:5] = 1  # 2x3 blob
    labels[7:10, 6:8] = 2  # 3x2 blob
    excess = torch.rand(H, W) * 100.0  # positive -> intensity-weighted centroid
    z = torch.randn(H, W)
    log_bf = torch.randn(H, W)
    posterior = torch.rand(H, W)
    var = torch.rand(H, W) + 0.1
    stats = compute_blob_stats(
        labels, 2, excess=excess, z=z, log_bf=log_bf, posterior=posterior, var=var
    )
    for b in (1, 2):
        i, m = b - 1, labels == b
        idx = torch.nonzero(m)
        rr, cc = idx[:, 0].to(excess.dtype), idx[:, 1].to(excess.dtype)
        w = excess[m].clamp_min(0)
        wsum = float(w.sum())
        rc = float((w * rr).sum()) / wsum
        cc_ = float((w * cc).sum()) / wsum
        Crr = float((w * (rr - rc) ** 2).sum()) / wsum
        Ccc = float((w * (cc - cc_) ** 2).sum()) / wsum
        Crc = float((w * (rr - rc) * (cc - cc_)).sum()) / wsum
        tr = Crr + Ccc
        disc = (tr * tr - 4.0 * (Crr * Ccc - Crc * Crc)) ** 0.5
        ecc = (0.5 * (tr + disc)) / (0.5 * (tr - disc))
        assert int(stats.size[i]) == int(m.sum())
        assert float(stats.intensity_sum[i]) == pytest.approx(
            float(excess[m].sum()), abs=1e-3
        )
        assert float(stats.intensity_sigma[i]) == pytest.approx(
            float(var[m].sum().sqrt()), abs=1e-4
        )
        assert float(stats.log_bf_sum[i]) == pytest.approx(
            float(log_bf[m].sum()), abs=1e-4
        )
        assert float(stats.posterior_mean[i]) == pytest.approx(
            float(posterior[m].mean()), abs=1e-5
        )
        assert float(stats.intensity_max[i]) == pytest.approx(
            float(excess[m].max()), abs=1e-5
        )
        assert float(stats.z_max[i]) == pytest.approx(float(z[m].max()), abs=1e-5)
        assert float(stats.row_centroid[i]) == pytest.approx(rc, abs=1e-3)
        assert float(stats.col_centroid[i]) == pytest.approx(cc_, abs=1e-3)
        assert float(stats.eccentricity[i]) == pytest.approx(ecc, rel=1e-3)
        assert int(stats.bbox_r0[i]) == int(idx[:, 0].min())
        assert int(stats.bbox_r1[i]) == int(idx[:, 0].max()) + 1
        assert int(stats.bbox_c0[i]) == int(idx[:, 1].min())
        assert int(stats.bbox_c1[i]) == int(idx[:, 1].max()) + 1


# --- the beamstop learner is anchored on the geometry's own mask -------------

_BS_BINS = 40
_BS_QMAX = 0.552


def _b2ar_run0061_histogram():
    # measured on b2ar run0061: bins 0-1 masked out by the geometry, bin 2 a
    # sliver of surviving pixels, real low-angle reflections in bins 3-8.
    ph = [0, 0, 6, 14, 15, 29, 27, 48, 30, 19, 12, 19, 11, 9] + [8] * 26
    pix = [
        0,
        0,
        1746,
        12845,
        20049,
        25621,
        31167,
        36265,
        41723,
        47170,
        53127,
        56936,
        64344,
        71563,
    ]
    pix += [71563 + 3000 * i for i in range(26)]
    return (
        torch.tensor(ph, dtype=torch.float32),
        torch.tensor(pix, dtype=torch.float32),
    )


def _bs_edges():
    return torch.linspace(0.0, _BS_QMAX, _BS_BINS + 1)


def test_beamstop_not_inferred_from_real_low_angle_reflections():
    # Six real reflections in a bin that is 81% masked divided by almost
    # nothing, which the old rule read as an artifact ring and answered with a
    # 9.1 A exclusion -- discarding every peak inside 1/d = 1.1 nm^-1.
    ph, pix = _b2ar_run0061_histogram()
    assert _beamstop_qmin_from_histogram(ph, pix, _bs_edges(), _BS_BINS) is None


def test_beamstop_decision_does_not_move_with_seed_count():
    # Scaling every bin together must not change the decision. The old rule
    # zeroed bins under a peak-count floor before taking its reference median,
    # so more seed frames admitted sparse outer bins, pulled the median down
    # and made a spurious spike MORE likely.
    ph, pix = _b2ar_run0061_histogram()
    edges = _bs_edges()
    base = _beamstop_qmin_from_histogram(ph, pix, edges, _BS_BINS)
    for scale in (0.5, 3.3, 10.0):
        assert (
            _beamstop_qmin_from_histogram(ph * scale, pix, edges, _BS_BINS) == base
        ), f"decision moved at scale {scale}"


def test_beamstop_still_found_when_the_ring_is_real():
    # A real beamstop edge: a large pile-up immediately outside the masked
    # shadow, in a bin with enough pixels to trust.
    ph, pix = _b2ar_run0061_histogram()
    ph, pix = ph.clone(), pix.clone()
    ph[2], pix[2] = 4000.0, 15000.0
    q = _beamstop_qmin_from_histogram(ph, pix, _bs_edges(), _BS_BINS)
    assert q is not None
    # and it stops at the ring, rather than growing out over real reflections
    assert q == pytest.approx(float(_bs_edges()[3]), rel=1e-6)


def test_beamstop_exclusion_is_bounded():
    # Even with every inner bin elevated, the exclusion may not run away: the
    # old growth walk went from bin 2 out to bin 8 and swallowed 163 real peaks.
    ph, pix = _b2ar_run0061_histogram()
    ph, pix = ph.clone(), pix.clone()
    pix[2] = 15000.0
    ph[2:9] = torch.tensor([4000.0] * 7)
    q = _beamstop_qmin_from_histogram(ph, pix, _bs_edges(), _BS_BINS)
    assert q is not None
    max_bin = 2 + _BEAMSTOP_MAX_EXTEND_BINS
    assert q <= float(_bs_edges()[max_bin]) + 1e-9


def test_footprint_cap_keeps_a_bright_broad_spot_under_size_max_30():
    # size_max=30 alone rejects a bright, broad Bragg peak; the brightness-scaled
    # footprint cap must keep it while an explicit size_max=30 without the cap
    # still rejects it
    _, finder = _calibrated(
        matched_filter=True,
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
        psf_sigma=2.5,
    )
    stats = _one_result(finder, big).stats
    assert len(stats) == 1
    assert int(stats.size[0]) > 30
    assert stats.response_max is not None and float(stats.response_max[0]) > 0
    base = dict(size_min=1, size_max=30, eccentricity_max=1000.0, peakedness_min=0.0)
    assert int(filter_blobs(stats, **base).sum()) == 0
    kept = filter_blobs(
        stats,
        **base,
        footprint_scale=max(finder.mf_scales),
        threshold=finder.mf_threshold,
    )
    assert int(kept.sum()) == 1


def test_footprint_cap_is_brightness_invariant():
    # the same spot 100x brighter has a bigger footprint but must still pass
    _, finder = _calibrated(
        matched_filter=True,
        local_background=False,
        size_min=1,
        size_max=100000,
        eccentricity_max=1000.0,
        peakedness_min=0.0,
    )
    sizes = []
    for amp in (3000.0, 300000.0):
        frame = sim.render_frame(
            SHAPE,
            [[80.0, 80.0]],
            [amp],
            background=BG,
            noise_sigma=NS,
            seed=3,
            psf_sigma=1.5,
        )
        stats = _one_result(finder, frame).stats
        assert len(stats) == 1
        sizes.append(int(stats.size[0]))
        kept = filter_blobs(
            stats,
            size_min=1,
            size_max=30,
            eccentricity_max=1000.0,
            peakedness_min=0.0,
            footprint_scale=max(finder.mf_scales),
            threshold=finder.mf_threshold,
        )
        assert int(kept.sum()) == 1
    assert sizes[1] > sizes[0]


def test_footprint_cap_still_rejects_an_extended_plateau():
    # a flat 40x40 block is far larger than any spot of its peak response
    _, finder = _calibrated(
        matched_filter=True,
        local_background=False,
        size_min=1,
        size_max=100000,
        eccentricity_max=1000.0,
        peakedness_min=0.0,
    )
    frame = sim.render_frame(
        SHAPE, [], [], background=BG, noise_sigma=NS, seed=5, psf_sigma=1.0
    )
    frame = np.array(frame, copy=True)
    frame[60:100, 60:100] += 8.0 * NS
    stats = _one_result(finder, frame).stats
    assert len(stats) >= 1
    kept = filter_blobs(
        stats,
        size_min=1,
        size_max=30,
        eccentricity_max=1000.0,
        peakedness_min=0.0,
        footprint_scale=max(finder.mf_scales),
        threshold=finder.mf_threshold,
    )
    big = stats.size >= 1000
    assert bool(big.any())
    assert not bool(kept[big].any())


def test_footprint_cap_formula():
    resp = torch.tensor([1.0, math.e, math.e**2])
    cap = footprint_cap(resp, scale=2.0, threshold=1.0, tolerance=1.0)
    want = 2.0 * math.pi * 4.0 * torch.tensor([0.0, 1.0, 2.0])
    assert torch.allclose(cap, want)
    # responses below threshold never produce a negative cap
    assert float(footprint_cap(torch.tensor([0.1]), 2.0, 1.0, 1.0)[0]) == 0.0
