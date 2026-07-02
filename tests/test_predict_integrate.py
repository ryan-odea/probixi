from __future__ import annotations

import pytest
import sim
import torch

from probixi.indexer.integrate import integrate_predicted
from probixi.indexer.lattice import cell_to_B
from probixi.indexer.predict import detector_q_max, predict_reflections

SHAPE = (256, 256)
WAVELENGTH = 1.0


def _synthetic_geometry() -> dict:
    return sim.synthetic_geometry(
        h=SHAPE[0], w=SHAPE[1], clen=0.2, pixel_size=75e-6, wavelength=WAVELENGTH
    )


def _orientation() -> torch.Tensor:
    return sim.proper_rotation(0, max_angle_deg=8)


def _A(cell) -> torch.Tensor:
    U = _orientation()
    B = cell_to_B(cell, dtype=torch.float64)
    return U @ B


def test_detector_q_max_is_positive_and_below_ewald_diameter(cell):
    geom = _synthetic_geometry()
    q_max = detector_q_max(geom, SHAPE)
    assert q_max > 0.0
    # elastic scattering cannot exceed |q| = 2/lambda
    assert q_max <= 2.0 / WAVELENGTH + 1e-9


def test_detector_q_max_grows_with_detector_size(cell):
    geom = _synthetic_geometry()
    small = detector_q_max(geom, (64, 64))
    large = detector_q_max(geom, (512, 512))
    assert large > small


def test_predicted_hkl_are_integers(cell):
    geom = _synthetic_geometry()
    A = _A(cell)
    pred = predict_reflections(A, geom, detector_q_max(geom, SHAPE))
    assert len(pred) > 0
    assert pred.hkl.dtype == torch.int64
    hkl_f = pred.hkl.to(torch.float64)
    assert torch.equal(hkl_f, torch.round(hkl_f))


def test_predicted_q_equals_A_times_hkl(cell):
    geom = _synthetic_geometry()
    A = _A(cell)
    pred = predict_reflections(A, geom, detector_q_max(geom, SHAPE))
    q_from_hkl = pred.hkl.to(torch.float64) @ A.transpose(-1, -2)
    assert torch.allclose(pred.q, q_from_hkl)


def test_predicted_resolution_equals_q_norm(cell):
    geom = _synthetic_geometry()
    A = _A(cell)
    pred = predict_reflections(A, geom, detector_q_max(geom, SHAPE))
    assert torch.allclose(pred.resolution, torch.linalg.vector_norm(pred.q, dim=-1))


def test_excitation_errors_within_partiality_threshold(cell):
    geom = _synthetic_geometry()
    A = _A(cell)
    thr = 0.0025
    pred = predict_reflections(
        A, geom, detector_q_max(geom, SHAPE), partiality_threshold=thr
    )
    assert len(pred) > 0
    assert float(pred.excitation_error.abs().max()) < thr


def test_predicted_positions_cover_injected_lattice_peaks(cell):
    geom = _synthetic_geometry()
    A = _A(cell)
    lattice_pos, _ = sim.lattice_peaks(geom, cell, _orientation())
    assert lattice_pos.shape[0] > 0
    pred = predict_reflections(A, geom, detector_q_max(geom, SHAPE), frame_shape=SHAPE)
    # every injected lattice peak coincides with a predicted position
    d = torch.cdist(lattice_pos, pred.positions)
    nearest = d.min(dim=1).values
    assert float(nearest.max()) < 1e-6


def test_off_frame_reflections_are_filtered_when_frame_shape_given(cell):
    geom = _synthetic_geometry()
    A = _A(cell)
    q_max = detector_q_max(geom, SHAPE)
    pred_all = predict_reflections(A, geom, q_max, frame_shape=None)
    pred_on = predict_reflections(A, geom, q_max, frame_shape=SHAPE)
    # filtering can only drop reflections, never add
    assert len(pred_on) <= len(pred_all)
    assert len(pred_on) < len(pred_all)
    rows, cols = pred_on.positions[:, 0], pred_on.positions[:, 1]
    assert bool((rows >= 0).all() and (rows <= SHAPE[0] - 1).all())
    assert bool((cols >= 0).all() and (cols <= SHAPE[1] - 1).all())


def test_predict_rejects_non_3x3_A(cell):
    geom = _synthetic_geometry()
    with pytest.raises(ValueError):
        predict_reflections(torch.eye(2, dtype=torch.float64), geom, 1.0)


def test_integrate_recovers_injected_intensity_and_background(cell):
    geom = _synthetic_geometry()
    lattice_pos, _ = sim.lattice_peaks(geom, cell, _orientation())
    n = lattice_pos.shape[0]
    assert n > 0
    truth_I = 5000.0
    background = 100.0
    noise_sigma = 10.0
    intensities = torch.full((n,), truth_I, dtype=torch.float64)
    frame = sim.render_frame(
        SHAPE,
        lattice_pos.numpy(),
        intensities.numpy(),
        background=background,
        noise_sigma=noise_sigma,
        seed=1,
    )
    fr = torch.from_numpy(frame).to(torch.float64)
    excess = fr - background  # background-subtracted excess map
    var = torch.full(SHAPE, noise_sigma**2, dtype=torch.float64)
    mean = torch.full(SHAPE, background, dtype=torch.float64)

    positions, intensity, sigma, snapped, peak, bg = integrate_predicted(
        lattice_pos,
        excess,
        var,
        obs_positions=torch.empty(0, 2, dtype=torch.float64),
        obs_intensity=torch.empty(0, dtype=torch.float64),
        obs_sigma=torch.empty(0, dtype=torch.float64),
        box_radius=4,
        mean=mean,
    )
    # box sum recovers the bulk of each spot's total counts
    assert abs(float(intensity.median()) - truth_I) < 0.1 * truth_I
    # background reported is the injected mean
    assert torch.allclose(bg, torch.full((n,), background, dtype=torch.float64))
    # box sigma combines the summed per-pixel background variance with the
    # signal's own shot noise (adu_per_photon defaults to 1.0); for spots whose
    # full box lies inside the frame the count is the complete (2r+1)^2 box
    box_radius = 4
    # integrate_boxes rounds the centre to the nearest pixel before slicing
    centre = torch.round(lattice_pos).to(torch.long)
    rows, cols = centre[:, 0], centre[:, 1]
    interior = (
        (rows >= box_radius)
        & (rows <= SHAPE[0] - 1 - box_radius)
        & (cols >= box_radius)
        & (cols <= SHAPE[1] - 1 - box_radius)
    )
    assert bool(interior.any())
    box_pixels = (2 * box_radius + 1) ** 2
    expected_sigma = torch.sqrt(box_pixels * noise_sigma**2 + intensity.clamp_min(0.0))
    assert torch.allclose(sigma[interior], expected_sigma[interior])


def test_integrate_snaps_predicted_to_nearby_observed_peak(cell):
    geom = _synthetic_geometry()
    lattice_pos, _ = sim.lattice_peaks(geom, cell, _orientation())
    n = lattice_pos.shape[0]
    assert n > 0
    excess = torch.zeros(SHAPE, dtype=torch.float64)
    var = torch.ones(SHAPE, dtype=torch.float64)
    # predicted positions are offset; observed peaks sit on the true lattice
    pred_pos = lattice_pos + 0.4
    obs_pos = lattice_pos
    positions, intensity, sigma, snapped, peak, bg = integrate_predicted(
        pred_pos,
        excess,
        var,
        obs_positions=obs_pos,
        obs_intensity=torch.full((n,), 1.0, dtype=torch.float64),
        obs_sigma=torch.full((n,), 1.0, dtype=torch.float64),
        snap_radius=5.0,
        box_radius=3,
    )
    assert bool(snapped.all())
    # snapped centres land exactly on the observed peaks
    assert torch.allclose(positions, obs_pos)


def test_integrate_does_not_snap_observed_peak_outside_snap_radius():
    shape = (64, 64)
    excess = torch.zeros(shape, dtype=torch.float64)
    var = torch.ones(shape, dtype=torch.float64)
    pred_pos = torch.tensor([[20.0, 20.0]], dtype=torch.float64)
    obs_pos = torch.tensor([[40.0, 40.0]], dtype=torch.float64)
    positions, intensity, sigma, snapped, peak, bg = integrate_predicted(
        pred_pos,
        excess,
        var,
        obs_positions=obs_pos,
        obs_intensity=torch.tensor([1.0], dtype=torch.float64),
        obs_sigma=torch.tensor([1.0], dtype=torch.float64),
        snap_radius=5.0,
        box_radius=2,
    )
    assert not bool(snapped[0])
    # without a snap the predicted position is retained
    assert torch.allclose(positions, pred_pos)


def test_integrate_with_no_observed_peaks_keeps_predicted_positions():
    shape = (64, 64)
    excess = torch.zeros(shape, dtype=torch.float64)
    var = torch.ones(shape, dtype=torch.float64)
    pred_pos = torch.tensor([[10.0, 12.0], [30.0, 40.0]], dtype=torch.float64)
    positions, intensity, sigma, snapped, peak, bg = integrate_predicted(
        pred_pos,
        excess,
        var,
        obs_positions=torch.empty(0, 2, dtype=torch.float64),
        obs_intensity=torch.empty(0, dtype=torch.float64),
        obs_sigma=torch.empty(0, dtype=torch.float64),
        box_radius=2,
    )
    assert not bool(snapped.any())
    assert torch.allclose(positions, pred_pos)


def test_integrate_peak_is_box_maximum_of_excess():
    shape = (32, 32)
    excess = torch.zeros(shape, dtype=torch.float64)
    excess[16, 16] = 42.0
    var = torch.ones(shape, dtype=torch.float64)
    pred_pos = torch.tensor([[16.0, 16.0]], dtype=torch.float64)
    positions, intensity, sigma, snapped, peak, bg = integrate_predicted(
        pred_pos,
        excess,
        var,
        obs_positions=torch.empty(0, 2, dtype=torch.float64),
        obs_intensity=torch.empty(0, dtype=torch.float64),
        obs_sigma=torch.empty(0, dtype=torch.float64),
        box_radius=2,
    )
    assert float(peak[0]) == pytest.approx(42.0)
    # the single hot pixel is the only excess in the box, so the sum matches it
    assert float(intensity[0]) == pytest.approx(42.0)
