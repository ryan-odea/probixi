from __future__ import annotations

import math

import pytest
import sim
import torch

from probixi.indexer import IndexResult, RefineConfig, SeedConfig
from probixi.indexer.indexer import Indexer
from probixi.indexer.integrate import (
    integrate_rings,
    keep_non_overlapping,
    radial_profile,
    radii_from_profile,
    snap_positions,
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
    positions, snapped = snap_positions(predicted, observed, radius=5.0)
    assert bool(snapped.all())
    assert torch.allclose(positions, observed)


def test_snap_positions_leaves_prediction_beyond_radius():
    predicted = torch.tensor([[10.0, 10.0]])
    observed = torch.tensor([[30.0, 10.0]])
    positions, snapped = snap_positions(predicted, observed, radius=5.0)
    assert not bool(snapped.any())
    assert torch.allclose(positions, predicted)


def test_snap_positions_handles_empty_inputs():
    empty = torch.zeros((0, 2))
    positions, snapped = snap_positions(empty, torch.tensor([[1.0, 1.0]]), radius=5.0)
    assert positions.shape == (0, 2) and snapped.shape == (0,)
    predicted = torch.tensor([[10.0, 10.0]])
    positions, snapped = snap_positions(predicted, empty, radius=5.0)
    assert not bool(snapped.any())


def test_keep_non_overlapping_drops_both_members_of_a_close_pair():
    # 1.5 * radius = 6 px cutoff; the pair is 4 px apart, the third is far
    positions = torch.tensor([[10.0, 10.0], [10.0, 14.0], [40.0, 40.0]])
    keep = keep_non_overlapping(positions, 4.0)
    assert keep.tolist() == [False, False, True]


def test_keep_non_overlapping_ignores_neighbours_on_another_panel():
    positions = torch.tensor([[10.0, 10.0], [10.0, 14.0]])
    panels = torch.tensor([0, 1])
    assert keep_non_overlapping(positions, 4.0, panels).tolist() == [True, True]


def test_keep_non_overlapping_spans_block_boundary():
    # blocks are 512 wide: a conflicting pair straddling the boundary is caught
    far = torch.arange(600, dtype=torch.float32)[:, None].repeat(1, 2) * 20.0
    positions = torch.cat([far, torch.tensor([[0.0, 3.0]])])
    keep = keep_non_overlapping(positions, 4.0)
    assert not bool(keep[0]) and not bool(keep[-1])


def test_keep_non_overlapping_rejects_bad_input():
    with pytest.raises(ValueError, match="shape"):
        keep_non_overlapping(torch.zeros(4), 4.0)
    with pytest.raises(ValueError, match="finite and positive"):
        keep_non_overlapping(torch.zeros((2, 2)), 0.0)


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
    positions, intensity, sigma, snapped, peak, background = integrate_rings(
        torch.zeros((0, 2)),
        excess,
        var,
        torch.zeros((0, 2)),
        mean=mean,
        radii=(3.0, 5.0, 7.0),
    )
    for t in (positions, intensity, sigma, snapped, peak, background):
        assert len(t) == 0


def test_integrate_rings_recovers_intensity_above_flat_background():
    excess, var, mean = _flat_frame(7.0)
    excess[32, 32] = 100.0
    _, intensity, _, _, peak, background = integrate_rings(
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


def test_integrate_rings_does_not_double_count_overlapping_disks():
    excess, var, mean = _flat_frame(0.0)
    excess[32, 32] = 60.0  # equidistant from both centres -> one owner only
    positions = torch.tensor([[32.0, 30.0], [32.0, 34.0]])
    _, intensity, _, _, _, _ = integrate_rings(
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
    _, intensity, sigma, _, _, _ = integrate_rings(
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
    _, _, sigma, _, _, _ = integrate_rings(
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
    _, _, sigma, _, _, _ = integrate_rings(
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


def test_exclude_overlaps_filters_across_lattices_and_clears_scratch(
    geometry_dict, cell
):
    idxr = Indexer(geometry_dict, cell)
    idxr._measured_radii = (4.0, 6.0, 8.0)
    # one crystal per lattice; their single centres are 4 px apart, so the
    # 1.5 * 4 = 6 px rule must drop both even though neither conflicts alone
    keep = torch.ones(1, dtype=torch.bool)
    a = _result(cell, torch.tensor([[20.0, 20.0]]), keep)
    b = _result(cell, torch.tensor([[20.0, 24.0]]), keep)
    idxr._exclude_overlaps([a, b])
    for r in (a, b):
        assert len(r.predicted_positions) == 0
        assert r._integration_positions is None and r._integration_valid is None


def test_exclude_overlaps_keeps_well_separated_reflections(geometry_dict, cell):
    idxr = Indexer(geometry_dict, cell)
    idxr._measured_radii = (4.0, 6.0, 8.0)
    keep = torch.ones(1, dtype=torch.bool)
    a = _result(cell, torch.tensor([[20.0, 20.0]]), keep)
    b = _result(cell, torch.tensor([[80.0, 80.0]]), keep)
    idxr._exclude_overlaps([a, b])
    for r in (a, b):
        assert len(r.predicted_positions) == 1


def test_exclude_overlaps_ignores_crystals_without_circular_integration(
    geometry_dict, cell
):
    idxr = Indexer(geometry_dict, cell)
    idxr._measured_radii = (4.0, 6.0, 8.0)
    r = _result(cell, torch.tensor([[20.0, 20.0]]), torch.ones(1, dtype=torch.bool))
    r._integration_positions = r._integration_valid = None
    idxr._exclude_overlaps([r])  # no scratch state -> nothing to do
    assert len(r.predicted_positions) == 1


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
