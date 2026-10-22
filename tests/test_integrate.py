from __future__ import annotations

import math

import pytest
import sim
import torch

from probixi.indexer import IndexResult, RefineConfig, SeedConfig
from probixi.indexer.indexer import Indexer
from probixi.indexer.integrate import (
    choose_aperture,
    disk_overlap_counts,
    falloff_resolution_limit,
    integrate_rings,
    nearest_neighbour_distance,
    radial_profile,
    radii_from_profile,
    radius_by_shell,
    ring_sums,
    snap_positions,
    snr_disk_radius,
    unexplained_signal_p,
)

SHAPE = (64, 64)


def _flat_frame(background: float, sigma: float = 0.0):
    mean = torch.full(SHAPE, background, dtype=torch.float32)
    excess = torch.zeros(SHAPE, dtype=torch.float32)
    var = torch.full(SHAPE, max(sigma**2, 1.0), dtype=torch.float32)
    return excess, var, mean


def test_snap_positions_moves_prediction_onto_nearby_observed():
    predicted = torch.tensor([[10.0, 10.0]])
    observed = torch.tensor([[12.0, 10.0]])
    positions, snapped, _ = snap_positions(predicted, observed, radius=5.0)
    assert bool(snapped.all())
    assert torch.allclose(positions, observed)


def test_snap_positions_leaves_prediction_beyond_radius():
    predicted = torch.tensor([[10.0, 10.0]])
    observed = torch.tensor([[30.0, 10.0]])
    positions, snapped, _ = snap_positions(predicted, observed, radius=5.0)
    assert not bool(snapped.any())
    assert torch.allclose(positions, predicted)


def test_snap_positions_handles_empty_inputs():
    empty = torch.zeros((0, 2))
    positions, snapped, _ = snap_positions(
        empty, torch.tensor([[1.0, 1.0]]), radius=5.0
    )
    assert positions.shape == (0, 2) and snapped.shape == (0,)
    predicted = torch.tensor([[10.0, 10.0]])
    positions, snapped, _ = snap_positions(predicted, empty, radius=5.0)
    assert not bool(snapped.any())


def test_disk_overlap_counts_flags_both_members_of_a_close_pair():
    # radius-4 disks 4 px apart share pixels; the third centre is far away
    positions = torch.tensor([[10.0, 10.0], [10.0, 14.0], [40.0, 40.0]])
    n = disk_overlap_counts(positions, 4.0).tolist()
    assert n[0] == n[1] > 0 and n[2] == 0


def test_disk_overlap_counts_ignores_neighbours_on_another_panel():
    positions = torch.tensor([[10.0, 10.0], [10.0, 14.0]])
    panels = torch.tensor([0, 1])
    assert disk_overlap_counts(positions, 4.0, panels).tolist() == [0, 0]


def test_disk_overlap_counts_spans_block_boundary():
    # blocks are 512 wide: a conflicting pair straddling the boundary is caught
    far = torch.arange(600, dtype=torch.float32)[:, None].repeat(1, 2) * 20.0
    positions = torch.cat([far, torch.tensor([[0.0, 3.0]])])
    n = disk_overlap_counts(positions, 4.0)
    assert int(n[0]) > 0 and int(n[-1]) > 0 and int(n[1]) == 0


def test_disk_overlap_counts_rejects_bad_input():
    with pytest.raises(ValueError, match="shape"):
        disk_overlap_counts(torch.zeros(4), 4.0)
    with pytest.raises(ValueError, match="finite and positive"):
        disk_overlap_counts(torch.zeros((2, 2)), 0.0)


def test_integrate_rings_requires_the_background_mean():
    excess, var, _ = _flat_frame(5.0)
    with pytest.raises(ValueError, match="background mean"):
        integrate_rings(
            torch.tensor([[32.0, 32.0]]),
            excess,
            var,
            torch.zeros((0, 2)),
            radii=(3.0, 5.0, 7.0),
        )


def test_integrate_rings_returns_empty_for_no_predictions():
    excess, var, mean = _flat_frame(5.0)
    positions, intensity, sigma, snapped, peak, background, *extras = integrate_rings(
        torch.zeros((0, 2)),
        excess,
        var,
        torch.zeros((0, 2)),
        mean=mean,
        radii=(3.0, 5.0, 7.0),
    )
    for t in (positions, intensity, sigma, snapped, peak, background, *extras):
        assert len(t) == 0
    assert len(extras) == 5


def test_integrate_rings_recovers_intensity_above_flat_background():
    excess, var, mean = _flat_frame(7.0)
    excess[32, 32] = 100.0
    _, intensity, _, _, peak, background, *_ = integrate_rings(
        torch.tensor([[32.0, 32.0]]),
        excess,
        var,
        torch.zeros((0, 2)),
        mean=mean,
        radii=(3.0, 5.0, 7.0),
    )
    # a flat annulus estimates the background exactly, leaving the spike
    assert background.item() == pytest.approx(7.0, abs=1e-4)
    assert intensity.item() == pytest.approx(100.0, abs=1e-3)
    assert peak.item() == pytest.approx(100.0, abs=1e-3)


def test_integrate_rings_takes_one_radius_per_reflection():
    excess, var, mean = _flat_frame(7.0)
    excess[20, 20] = 100.0
    excess[44, 44 + 2] = 100.0  # two pixels off the second centre
    radii = (
        torch.tensor([1.5, 2.5]),
        torch.tensor([3.0, 3.5]),
        torch.tensor([6.0, 6.5]),
    )
    _, intensity, *_, n_total, _ = integrate_rings(
        torch.tensor([[20.0, 20.0], [44.0, 44.0]]),
        excess,
        var,
        torch.zeros((0, 2)),
        mean=mean,
        radii=radii,
    )
    assert intensity.tolist() == pytest.approx([100.0, 100.0], abs=1e-3)
    assert n_total.tolist() == [9, 21]


def test_disk_overlap_counts_uses_each_disk_radius():
    centres = torch.tensor([[10.0, 10.0], [10.0, 15.0]])
    assert disk_overlap_counts(centres, 1.5).tolist() == [0, 0]
    assert disk_overlap_counts(centres, torch.tensor([1.5, 4.0])).tolist()[0] > 0


def test_radius_by_shell_follows_the_stacked_profile():
    n = torch.tensor([1.0, 8, 12, 16, 32, 28, 40, 40, 48])
    sharp = torch.tensor([100.0, 10, 0, 0, 0, 0, 0, 0, 0])
    broad = torch.tensor([30.0, 25, 15, 8, 0, 0, 0, 0, 0])
    flat = torch.zeros(9)
    dip = -broad  # noise can stack negative beyond the diffraction limit
    profiles = torch.stack([sharp, broad, flat, dip])
    counts = (n * 1e4).expand(4, 9)
    sums = profiles * counts
    squares = counts * 100.0 + sums * profiles
    radii = radius_by_shell(sums, counts, squares, base=1.5)
    # sharp spot keeps the base disk, a blurred one widens, an empty shell keeps it
    assert radii[0] == 1.5
    assert radii[1] > 2.0
    assert radii[3] == radii[2] == radii[1]


def test_integrate_rings_does_not_double_count_overlapping_disks():
    excess, var, mean = _flat_frame(0.0)
    excess[32, 32] = 60.0  # equidistant from both centres -> one owner only
    positions = torch.tensor([[32.0, 30.0], [32.0, 34.0]])
    _, intensity, _, _, _, _, *_ = integrate_rings(
        positions,
        excess,
        var,
        torch.zeros((0, 2)),
        mean=mean,
        radii=(3.0, 5.0, 7.0),
    )
    assert intensity.sum().item() == pytest.approx(60.0, abs=1e-3)


def test_integrate_rings_falls_back_to_model_variance_on_a_thin_annulus():
    # At a panel edge the annulus has too few samples to estimate its own
    # variance. Rather than zeroing sigma -- which would drop the reflection at
    # write time -- fall back to the calibrated per-pixel variance.
    excess, var, mean = _flat_frame(1.0)
    excess[0, 0] = 20.0
    _, intensity, sigma, _, _, _, *_ = integrate_rings(
        torch.tensor([[0.0, 0.0]]),
        excess,
        var,
        torch.zeros((0, 2)),
        mean=mean,
        radii=(1.0, 1.5, 2.0),
    )
    assert float(intensity[0]) != 0.0
    assert float(sigma[0]) > 0.0  # usable, so the reflection survives


def test_integrate_rings_prefers_the_annulus_variance_when_it_has_samples():
    # With a wide annulus the measured spread wins over the model variance, so a
    # noisy background raises sigma even when the model variance is tiny.
    excess, var, mean = _flat_frame(5.0)
    var = torch.full(SHAPE, 1e-6, dtype=torch.float32)
    torch.manual_seed(0)
    excess = excess + torch.randn(SHAPE) * 4.0
    _, _, sigma, _, _, _, *_ = integrate_rings(
        torch.tensor([[32.0, 32.0]]),
        excess,
        var,
        torch.zeros((0, 2)),
        mean=mean,
        radii=(3.0, 5.0, 9.0),
    )
    assert float(sigma[0]) > 1.0


def test_integrate_rings_zero_sigma_only_when_no_pixels_are_owned():
    excess, var, mean = _flat_frame(1.0)
    # a centre far off the frame owns nothing at all
    _, _, sigma, _, _, _, *_ = integrate_rings(
        torch.tensor([[-50.0, -50.0]]),
        excess,
        var,
        torch.zeros((0, 2)),
        mean=mean,
        radii=(2.0, 4.0, 6.0),
    )
    assert float(sigma[0]) == 0.0


def _result(cell, positions: torch.Tensor, keep: torch.Tensor) -> IndexResult:
    n = len(positions)
    r = IndexResult(
        frame_index=0,
        n_peaks=n,
        n_indexed=n,
        rmsd=0.0,
        A=torch.eye(3),
        U=torch.eye(3),
        B=torch.eye(3),
        cell=cell,
        indexed_mask=torch.ones(n, dtype=torch.bool),
        hkl=torch.zeros((n, 3)),
        positions=positions,
        intensities=torch.ones(n),
        sigmas=torch.ones(n),
        loss_history=torch.zeros(1),
    )
    r.predicted_hkl = torch.zeros((int(keep.sum()), 3))
    r.predicted_positions = positions[keep]
    for name in ("intensities", "sigmas", "peak", "background"):
        setattr(r, "predicted_" + name, torch.arange(float(keep.sum())))
    r._integration_positions = positions
    r._integration_valid = keep
    return r


def test_flag_overlaps_counts_shared_pixels_across_lattices_and_clears_scratch(
    geometry_dict, cell
):
    idxr = Indexer(geometry_dict, cell)
    idxr._measured_radii = (4.0, 6.0, 8.0)
    # one crystal per lattice; their single centres are 4 px apart, so the
    # radius-4 disks share pixels even though neither conflicts alone
    keep = torch.ones(1, dtype=torch.bool)
    a = _result(cell, torch.tensor([[20.0, 20.0]]), keep)
    b = _result(cell, torch.tensor([[20.0, 24.0]]), keep)
    idxr._flag_overlaps([a, b])
    for r in (a, b):
        assert len(r.predicted_positions) == 1
        assert int(r.predicted_n_overlap[0]) > 0
        assert r._integration_positions is None and r._integration_valid is None


def test_flag_overlaps_leaves_well_separated_reflections_clean(geometry_dict, cell):
    idxr = Indexer(geometry_dict, cell)
    idxr._measured_radii = (4.0, 6.0, 8.0)
    keep = torch.ones(1, dtype=torch.bool)
    a = _result(cell, torch.tensor([[20.0, 20.0]]), keep)
    b = _result(cell, torch.tensor([[80.0, 80.0]]), keep)
    idxr._flag_overlaps([a, b])
    for r in (a, b):
        assert r.predicted_n_overlap.tolist() == [0]


def test_flag_overlaps_ignores_crystals_without_circular_integration(
    geometry_dict, cell
):
    idxr = Indexer(geometry_dict, cell)
    idxr._measured_radii = (4.0, 6.0, 8.0)
    r = _result(cell, torch.tensor([[20.0, 20.0]]), torch.ones(1, dtype=torch.bool))
    r._integration_positions = r._integration_valid = None
    idxr._flag_overlaps([r])  # no scratch state -> nothing to do
    assert r.predicted_n_overlap is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(max_lattices=0), "max_lattices"),
        (dict(max_index_peaks=5), "max_index_peaks"),
        (dict(rank_sigma=0.0), "rank_sigma"),
        (dict(peel_radius=float("nan")), "peel_radius"),
    ],
)
def test_seed_config_rejects_invalid_multi_lattice_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SeedConfig(**kwargs)


def test_valid_multi_lattice_configs_are_accepted():
    seed = SeedConfig(max_lattices=3, max_index_peaks=128, rank_sigma=0.003)
    assert seed.max_lattices == 3


def test_index_lattices_recovers_two_overlaid_lattices(geometry_dict, cell):
    # superpose the peaks of two distinct orientations on one frame; peeling
    # must find both, and neither twice
    seed = SeedConfig(
        n_directions=1500,
        n_spin=60,
        top_directions=12,
        max_candidates=32,
        max_lattices=2,
    )
    idxr = Indexer(
        geometry_dict,
        cell,
        seed=seed,
        refine=RefineConfig(max_iters=150, reassign_every=10),
    )
    U1 = sim.proper_rotation(0, max_angle_deg=8.0)
    U2 = sim.proper_rotation(7, max_angle_deg=8.0)
    p1, _ = sim.lattice_peaks(geometry_dict, cell, U1)
    p2, _ = sim.lattice_peaks(geometry_dict, cell, U2)
    positions = torch.cat([p1, p2]).to(torch.float32)
    n = len(positions)
    found = idxr.index_lattices(
        {0: positions},
        intensities_by_frame={0: torch.ones(n)},
        sigmas_by_frame={0: torch.ones(n)},
        weights_by_frame={0: torch.ones(n)},
    )
    lattices = found.get(0, [])
    assert len(lattices) == 2
    a, b = (r.A for r in lattices)
    # the two recovered orientations are genuinely different lattices
    assert float(torch.minimum((a - b).norm(), (a + b).norm()) / b.norm()) > 0.02


def test_index_lattices_stops_at_one_when_max_lattices_is_one(geometry_dict, cell):
    idxr = Indexer(
        geometry_dict,
        cell,
        seed=SeedConfig(
            n_directions=1500, n_spin=60, top_directions=12, max_candidates=32
        ),
        refine=RefineConfig(max_iters=150, reassign_every=10),
    )
    positions, _ = sim.lattice_peaks(geometry_dict, cell, sim.proper_rotation(0))
    positions = positions.to(torch.float32)
    n = len(positions)
    found = idxr.index_lattices(
        {0: positions},
        intensities_by_frame={0: torch.ones(n)},
        sigmas_by_frame={0: torch.ones(n)},
        weights_by_frame={0: torch.ones(n)},
    )
    assert len(found.get(0, [])) == 1


def _planted(psf: float, centres, shape=(320, 320)):
    excess = torch.zeros(shape, dtype=torch.float32)
    yy, xx = torch.meshgrid(
        torch.arange(shape[0]).float(), torch.arange(shape[1]).float(), indexing="ij"
    )
    for r, c in centres:
        excess += 500.0 * torch.exp(
            -(((yy - r) ** 2 + (xx - c) ** 2) / (2 * psf * psf))
        )
    return excess


def _analytic_radius(psf: float, floor: float = 0.02) -> float:
    # a Gaussian's ring mean falls to `floor` of its centre at this radius
    return psf * math.sqrt(-2.0 * math.log(floor))


@pytest.mark.parametrize("psf", [1.5, 2.0, 2.5, 3.5])
def test_learned_signal_radius_matches_the_analytic_gaussian_radius(psf):
    g = torch.Generator().manual_seed(0)
    centres = (torch.rand(30, 2, generator=g) * 260 + 30).round()
    radii = radii_from_profile(radial_profile(_planted(psf, centres), centres.float()))
    assert radii is not None
    assert radii[0] == pytest.approx(_analytic_radius(psf), rel=0.1)
    assert 0 < radii[0] < radii[1] < radii[2]


def test_learned_radius_is_not_widened_by_neighbouring_reflections():
    # A dense lattice puts neighbours just outside the spot. A threshold
    # crossing would be pushed out by their flux; the fit must not be.
    psf = 2.0
    axis = torch.arange(20, 300, 15).float()
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    centres = torch.stack([gy.flatten(), gx.flatten()], dim=-1)
    radii = radii_from_profile(radial_profile(_planted(psf, centres), centres))
    assert radii is not None
    assert radii[0] == pytest.approx(_analytic_radius(psf), rel=0.1)


def test_learned_annulus_holds_the_requested_background_samples():
    g = torch.Generator().manual_seed(0)
    centres = (torch.rand(30, 2, generator=g) * 260 + 30).round()
    profile = radial_profile(_planted(2.0, centres), centres.float())
    for want in (60.0, 120.0, 300.0):
        r_sig, r_in, r_out = radii_from_profile(profile, background_pixels=want)
        assert math.pi * (r_out**2 - r_in**2) == pytest.approx(want, rel=1e-6)
        assert r_in > r_sig  # the annulus clears the signal disk


def test_radial_profile_is_empty_without_peaks():
    excess = torch.zeros((32, 32), dtype=torch.float32)
    assert float(radial_profile(excess, torch.zeros((0, 2))).abs().sum()) == 0.0


def test_radii_from_profile_returns_none_for_a_flat_profile():
    assert radii_from_profile(torch.zeros(17)) is None


def test_radii_from_profile_rejects_bad_settings():
    profile = torch.linspace(1.0, 0.0, 17)
    for kwargs in (
        dict(floor=0.0),
        dict(floor=1.0),
        dict(fit_above=0.0),
        dict(gap=0.0),
        dict(background_pixels=0.0),
    ):
        with pytest.raises(ValueError):
            radii_from_profile(profile, **kwargs)


def test_snr_aperture_shrinks_to_the_3x3_for_a_pixel_wide_spot():
    # a ~1 px spot: S/sqrt(N) peaks at the 3x3 block, well inside the 2 % radius
    g = torch.Generator().manual_seed(0)
    centres = (torch.rand(30, 2, generator=g) * 260 + 30).round()
    profile = radial_profile(_planted(0.7, centres), centres.float())
    flux = radii_from_profile(profile)
    snr = radii_from_profile(profile, snr=True)
    assert flux is not None and snr is not None
    assert snr[0] == 1.5
    assert snr[0] < flux[0]
    assert snr[1:] == flux[1:]  # annulus unchanged


def test_snr_aperture_never_exceeds_the_flux_radius():
    g = torch.Generator().manual_seed(1)
    centres = (torch.rand(30, 2, generator=g) * 260 + 30).round()
    for psf in (1.0, 2.0, 3.0):
        profile = radial_profile(_planted(psf, centres), centres.float())
        flux = radii_from_profile(profile)
        snr = radii_from_profile(profile, snr=True)
        assert snr[0] <= flux[0]
        assert snr[0] >= 1.5


def test_snr_disk_radius_matches_the_background_limited_optimum():
    # Gaussian sigma=2: S/sqrt(N) is maximised near r ~ 1.6 sigma
    g = torch.Generator().manual_seed(2)
    centres = (torch.rand(30, 2, generator=g) * 260 + 30).round()
    profile = radial_profile(_planted(2.0, centres), centres.float())
    assert snr_disk_radius(profile, 16.0) == pytest.approx(3.5, abs=1.0)


def _crystals(n_cryst=12, per=10, seed=0):
    # spots of n_cryst crystals, each on its own frame, at random positions
    g = torch.Generator().manual_seed(seed)
    centres = (torch.rand(n_cryst * per, 2, generator=g) * 260 + 30).round()
    frame = torch.arange(n_cryst).repeat_interleave(per)
    q = ((centres.float() - 160) ** 2).sum(-1).sqrt()
    return centres, frame, q / q.max()


def _ring_data(
    psf,
    centres,
    shape=(320, 320),
    noise=4.0,
    seed=0,
    background=0.0,
    extra=None,
    amplitude=2000.0,
):
    # spots of per-centre width psf and amplitude (scalar or (N,)) on Gaussian
    # noise over a background (a per-spot local offset the annulus must remove);
    # |q| is the distance from the frame centre
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.arange(shape[0]).float(), torch.arange(shape[1]).float(), indexing="ij"
    )
    excess = torch.randn(shape, generator=g) * noise + background * (xx / shape[1])
    if extra is not None:
        excess += extra
    psf = torch.as_tensor(psf).float().expand(len(centres))
    amplitude = torch.as_tensor(amplitude).float().expand(len(centres))
    for (r, c), s, a in zip(centres, psf, amplitude):
        excess += a * torch.exp(-(((yy - r) ** 2 + (xx - c) ** 2) / (2 * s * s)))
    sums, counts, sq = ring_sums(excess, centres.float(), squares=True)
    q = ((centres.float() - shape[0] / 2) ** 2).sum(-1).sqrt()
    return sums, sq, counts, q, nearest_neighbour_distance(centres.float())


SNR_RADII, FLUX_RADII = (1.5, 5.0, 7.2), (4.0, 5.0, 7.2)


def _decide(psf, centres, frame, **kw):
    sums, sq, counts, q, nn = _ring_data(psf, centres, **kw)
    return choose_aperture(
        sums, sq, counts, q, frame, SNR_RADII, FLUX_RADII, nn_dist=nn
    )


def test_ring_sums_agree_with_the_radial_profile():
    g = torch.Generator().manual_seed(0)
    centres = (torch.rand(20, 2, generator=g) * 260 + 30).round()
    excess = _planted(2.0, centres)
    total, counts, sq = ring_sums(excess, centres.float(), squares=True)
    assert total.shape == counts.shape == sq.shape == (20, 17)
    assert torch.allclose(
        (total / counts.clamp_min(1)).median(0).values,
        radial_profile(excess, centres.float()),
    )
    # rings 0..3 (offsets rounding to that distance) hold 1, 8, 12, 16 pixels
    assert counts[0, :4].tolist() == [1.0, 8.0, 12.0, 16.0]
    assert torch.all(sq >= 0)


def test_auto_aperture_keeps_snr_for_a_constant_spot_shape():
    centres, frame, _ = _crystals(seed=3)
    choice, diag = _decide(1.2, centres, frame, background=40.0)
    assert choice == "snr", diag
    assert diag["n_crystals"] >= 10 and diag["crystal_spread"] < 0.02
    # keys the CLI summary reads on the full path
    assert {
        "f_mean",
        "n",
        "residual_trend",
        "crystal_spread",
        "spread_z",
    } <= diag.keys()


def test_auto_aperture_absorbs_a_resolution_trend_common_to_all_crystals():
    # every crystal's spots broaden alike with |q|: a common B, not a bias
    centres, frame, u = _crystals(seed=4)
    choice, diag = _decide(0.9 + 0.8 * u, centres, frame)
    assert choice == "snr", diag
    assert diag["common_trend"] < -0.05  # the trend is there, and ignored


def test_auto_aperture_falls_back_to_flux_when_crystals_broaden_differently():
    # mosaicity differing between crystals: the |q| slope of the captured
    # fraction is crystal-specific and no scale or B removes it
    centres, frame, u = _crystals(seed=5)
    eta = 1.6 * (frame.float() / frame.max())
    choice, diag = _decide(0.9 + eta * u, centres, frame)
    assert choice == "flux", diag
    assert diag["crystal_spread"] >= 0.02 and diag["spread_z"] >= 3
    assert diag["noise_ratio"] < 1


def test_auto_aperture_keeps_snr_when_the_flux_disk_noise_costs_more():
    # crystal-specific broadening again, but narrow spots fading into the noise
    # at high |q|: the flux disk's extra background outweighs the bias
    centres, frame, u = _crystals(n_cryst=20, per=20, seed=2)
    eta = frame.float() / frame.max()
    amp = 3000.0 * torch.exp(-4 * u * u)
    choice, diag = _decide(
        0.8 + 0.8 * eta * u, centres, frame, noise=25.0, amplitude=amp
    )
    assert choice == "snr", diag
    assert diag["spread_z"] >= 3 and diag["noise_ratio"] > 1


def test_auto_aperture_ignores_a_diffuse_halo_common_to_all_crystals():
    centres, frame, u = _crystals(seed=6)
    yy, xx = torch.meshgrid(
        torch.arange(320).float(), torch.arange(320).float(), indexing="ij"
    )
    halo = torch.zeros(320, 320)
    for (r, c), uu in zip(centres, u):
        halo += 400.0 * uu * torch.exp(-(((yy - r) ** 2 + (xx - c) ** 2) / 72.0))
    choice, diag = _decide(1.2, centres, frame, extra=halo)
    assert choice == "snr", diag


def test_auto_aperture_needs_enough_strong_peaks_and_crystals():
    centres, frame, _ = _crystals(n_cryst=2, per=5, seed=7)
    choice, diag = _decide(1.2, centres, frame)
    assert choice == "snr" and diag["reason"] == "too few strong peaks"
    centres, frame, _ = _crystals(n_cryst=6, per=12, seed=8)
    choice, diag = _decide(1.2, centres, frame)
    assert choice == "snr" and diag["reason"].startswith("too few crystals")
    assert "f_mean" in diag and "crystal_spread" not in diag  # partial diag
    sums, sq, counts, q, nn = _ring_data(1.2, centres)
    assert (
        choose_aperture(sums, sq, counts, q, frame, FLUX_RADII, FLUX_RADII, nn_dist=nn)[
            0
        ]
        == "flux"
    )


def test_falloff_limit_needs_a_shell_above_the_target():
    q = torch.linspace(1.0, 6.0, 200)
    assert falloff_resolution_limit(q, torch.full((200,), 0.2)) is None
    isig = torch.where(q < 3.0, 2.0, 0.1)
    assert 2.5 < falloff_resolution_limit(q, isig) < 3.5


def test_same_lattice_sees_through_a_change_of_setting():
    from probixi.indexer.indexer import _same_lattice

    A = torch.tensor([[0.016, 0.001, 0.0], [0.0, 0.011, 0.002], [0.001, 0.0, 0.0066]])
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0]))  # a 2-fold of mmm
    swap = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    assert _same_lattice(A, -A)
    assert _same_lattice(A, A @ flip)
    assert _same_lattice(A, A @ swap)
    assert not _same_lattice(A, A @ torch.diag(torch.tensor([2.0, 1.0, 1.0])))
    rot = torch.linalg.matrix_exp(
        torch.tensor([[0, -0.1, 0], [0.1, 0, 0], [0, 0, 0.0]])
    )
    assert not _same_lattice(A, rot @ A)


def test_unexplained_signal_separates_chance_from_real_lattices():
    g = torch.Generator().manual_seed(0)
    sigma = torch.full((300,), 2.0)
    use = torch.ones(300, dtype=torch.bool)
    null = [
        unexplained_signal_p(2.0 * torch.randn(300, generator=g), sigma, use)
        for _ in range(500)
    ]
    assert sum(p < 1e-3 for p in null) <= 3  # calibrated: ~N(0, 1) under chance
    weak = 2.0 * (0.4 + torch.randn(300, generator=g))  # <I/sigma> = 0.4
    assert unexplained_signal_p(weak, sigma, use) < 1e-6
    assert unexplained_signal_p(weak, sigma, torch.zeros_like(use)) == 0.0


def test_frame_background_stats_bands_and_ice_score():
    from probixi.indexer.integrate import frame_background_stats, resolution_bands

    d = torch.tensor([7.0, 5.0, 3.5, 2.5, 2.0]).repeat(40, 8)  # (40, 40)
    d[:20, :20] = 3.90  # an ice window
    d[20:, :20] = 3.80  # its flank
    band, ice = resolution_bands(d)
    assert band[0, 20:25].tolist() == [0, 1, 2, 3, 4]
    assert ice[0, 0] == 1 and ice[39, 0] == 2 and ice[0, 25] == 0
    mean = band.float()  # background equal to the band index
    excess = torch.where(ice == 1, 5.0, 1.0)
    stats = frame_background_stats(mean, excess, torch.ones_like(mean), None, band, ice)
    assert stats[:5].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert stats[5].item() == pytest.approx(4.0)
    masked = frame_background_stats(mean, None, None, band == 4, band, ice)
    assert torch.isnan(masked[:4]).all() and masked[4] == 4.0 and torch.isnan(masked[5])
