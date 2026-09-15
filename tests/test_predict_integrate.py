from __future__ import annotations

import pytest
import sim
import torch

from probixi.indexer.integrate import integrate_rings
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
    B = cell_to_B(cell, dtype=torch.float32)
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
    hkl_f = pred.hkl.to(torch.float32)
    assert torch.equal(hkl_f, torch.round(hkl_f))


def test_predicted_q_equals_A_times_hkl(cell):
    geom = _synthetic_geometry()
    A = _A(cell)
    pred = predict_reflections(A, geom, detector_q_max(geom, SHAPE))
    q_from_hkl = pred.hkl.to(torch.float32) @ A.transpose(-1, -2)
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
        predict_reflections(torch.eye(2, dtype=torch.float32), geom, 1.0)


def test_integrate_recovers_injected_intensity_and_background(cell):
    geom = _synthetic_geometry()
    lattice_pos, _ = sim.lattice_peaks(geom, cell, _orientation())
    n = lattice_pos.shape[0]
    assert n > 0
    truth_I = 5000.0
    background = 100.0
    noise_sigma = 10.0
    intensities = torch.full((n,), truth_I, dtype=torch.float32)
    frame = sim.render_frame(
        SHAPE,
        lattice_pos.numpy(),
        intensities.numpy(),
        background=background,
        noise_sigma=noise_sigma,
        seed=1,
    )
    fr = torch.from_numpy(frame).to(torch.float32)
    excess = fr - background  # background-subtracted excess map
    var = torch.full(SHAPE, noise_sigma**2, dtype=torch.float32)
    mean = torch.full(SHAPE, background, dtype=torch.float32)

    positions, intensity, sigma, snapped, peak, bg = integrate_rings(
        lattice_pos,
        excess,
        var,
        obs_positions=torch.empty(0, 2, dtype=torch.float32),
        mean=mean,
        radii=(4.0, 6.0, 9.0),
    )
    # the disk recovers the bulk of each spot's total counts
    assert abs(float(intensity.median()) - truth_I) < 0.1 * truth_I
    # the annulus recovers the flat background it was given
    assert float(bg.median()) == pytest.approx(background, rel=0.05)
    assert bool((sigma > 0).any())


def test_integrate_snaps_predicted_to_nearby_observed_peak(cell):
    geom = _synthetic_geometry()
    lattice_pos, _ = sim.lattice_peaks(geom, cell, _orientation())
    assert lattice_pos.shape[0] > 0
    excess = torch.zeros(SHAPE, dtype=torch.float32)
    var = torch.ones(SHAPE, dtype=torch.float32)
    mean = torch.zeros(SHAPE, dtype=torch.float32)
    # predicted positions are offset; observed peaks sit on the true lattice
    positions, _, _, snapped, _, _ = integrate_rings(
        lattice_pos + 0.4,
        excess,
        var,
        obs_positions=lattice_pos,
        snap_radius=5.0,
        mean=mean,
        radii=(3.0, 5.0, 7.0),
    )
    assert bool(snapped.all())
    # snapped centres land exactly on the observed peaks
    assert torch.allclose(positions, lattice_pos)


def test_integrate_does_not_snap_observed_peak_outside_snap_radius():
    shape = (64, 64)
    pred_pos = torch.tensor([[20.0, 20.0]], dtype=torch.float32)
    positions, _, _, snapped, _, _ = integrate_rings(
        pred_pos,
        torch.zeros(shape, dtype=torch.float32),
        torch.ones(shape, dtype=torch.float32),
        obs_positions=torch.tensor([[40.0, 40.0]], dtype=torch.float32),
        snap_radius=5.0,
        mean=torch.zeros(shape, dtype=torch.float32),
        radii=(2.0, 4.0, 6.0),
    )
    assert not bool(snapped[0])
    # without a snap the predicted position is retained
    assert torch.allclose(positions, pred_pos)


def test_integrate_with_no_observed_peaks_keeps_predicted_positions():
    shape = (64, 64)
    pred_pos = torch.tensor([[10.0, 12.0], [30.0, 40.0]], dtype=torch.float32)
    positions, _, _, snapped, _, _ = integrate_rings(
        pred_pos,
        torch.zeros(shape, dtype=torch.float32),
        torch.ones(shape, dtype=torch.float32),
        obs_positions=torch.empty(0, 2, dtype=torch.float32),
        mean=torch.zeros(shape, dtype=torch.float32),
        radii=(2.0, 4.0, 6.0),
    )
    assert not bool(snapped.any())
    assert torch.allclose(positions, pred_pos)


def test_integrate_deblend_owns_shared_pixel_by_nearest_then_lowest_index():
    # M>1 nearest-owner deblend: a pixel in the overlap of two disks is assigned
    # to exactly one centre (nearest, ties broken by lowest index), never
    # double-counted.
    shape = (48, 48)
    positions = torch.tensor([[24.0, 22.0], [24.0, 26.0]], dtype=torch.float32)
    excess = torch.zeros(shape, dtype=torch.float32)
    excess[24, 22] = 100.0  # only in disk 0
    excess[24, 26] = 80.0  # only in disk 1
    excess[24, 24] = 30.0  # equidistant (d=2) -> tie -> lowest index (disk 0)
    _, intensity, _, _, _, _ = integrate_rings(
        positions,
        excess,
        torch.ones(shape, dtype=torch.float32),
        obs_positions=torch.empty(0, 2, dtype=torch.float32),
        mean=torch.zeros(shape, dtype=torch.float32),
        radii=(3.0, 5.0, 7.0),
    )
    # shared pixel counted once, in disk 0; total conserved (no double-count)
    assert float(intensity[0]) == pytest.approx(130.0, abs=1e-3)
    assert float(intensity.sum()) == pytest.approx(210.0, abs=1e-3)


@pytest.mark.mps
def test_integrate_deblend_runs_on_mps_and_matches_cpu():
    # The M>1 deblend tie-break uses an int64 scatter_reduce, which has no MPS
    # kernel in some builds; guard that it runs and agrees with CPU.
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    shape = (48, 48)
    positions = torch.tensor([[24.0, 22.0], [24.0, 26.0]], dtype=torch.float32)
    excess = torch.zeros(shape, dtype=torch.float32)
    excess[24, 22] = 100.0
    excess[24, 26] = 80.0
    excess[24, 24] = 30.0  # overlap pixel -> exercises the tie-break scatter
    var = torch.ones(shape, dtype=torch.float32)
    mean = torch.zeros(shape, dtype=torch.float32)

    def run(dev: torch.device):
        out = integrate_rings(
            positions.to(dev),
            excess.to(dev),
            var.to(dev),
            obs_positions=torch.empty(0, 2, dtype=torch.float32, device=dev),
            mean=mean.to(dev),
            radii=(3.0, 5.0, 7.0),
        )
        return [t.cpu() for t in out]

    _, I_c, s_c, _, p_c, b_c = run(torch.device("cpu"))
    _, I_m, s_m, _, p_m, b_m = run(torch.device("mps"))
    assert torch.allclose(I_c, I_m, atol=1e-4)
    assert torch.allclose(s_c, s_m, atol=1e-4)
    assert torch.allclose(p_c, p_m, atol=1e-4)
    assert torch.allclose(b_c, b_m, atol=1e-4)
    assert float(I_m[0]) == pytest.approx(130.0, abs=1e-3)


def test_integrate_peak_is_disk_maximum_of_excess():
    shape = (32, 32)
    excess = torch.zeros(shape, dtype=torch.float32)
    excess[16, 16] = 42.0
    _, intensity, _, _, peak, _ = integrate_rings(
        torch.tensor([[16.0, 16.0]], dtype=torch.float32),
        excess,
        torch.ones(shape, dtype=torch.float32),
        obs_positions=torch.empty(0, 2, dtype=torch.float32),
        mean=torch.zeros(shape, dtype=torch.float32),
        radii=(2.0, 4.0, 6.0),
    )
    assert float(peak[0]) == pytest.approx(42.0)
    # the single hot pixel is the only excess in the disk, so the sum matches it
    assert float(intensity[0]) == pytest.approx(42.0)
