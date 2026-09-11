"""Photon counts for detected peaks, and how they reach the DB.

Only searched peaks carry these: a predicted position has no measured blob, so
there is no honest pixel set to sum a background over.
"""

from __future__ import annotations

import torch

from probixi.peakfinding.peaks.blobs import compute_blob_stats


def _maps():
    labels = torch.zeros(9, 9, dtype=torch.long)
    labels[2:4, 2:4] = 1  # 4-pixel blob
    labels[6, 6] = 2  # 1-pixel blob
    excess = torch.zeros(9, 9)
    excess[2:4, 2:4] = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    excess[6, 6] = 7.0
    mean = torch.full((9, 9), 3.0)
    return labels, excess, mean


def _stats(mean=None):
    labels, excess, m = _maps()
    return compute_blob_stats(
        labels,
        2,
        excess=excess,
        z=excess,
        log_bf=excess,
        posterior=excess,
        var=torch.ones(9, 9),
        mean=m if mean is None else mean,
    )


def test_background_sum_totals_the_noise_model_over_the_blob_pixels():
    stats = _stats()
    assert float(stats.background_sum[0]) == 12.0  # 4 px x mean 3.0
    assert float(stats.background_sum[1]) == 3.0  # 1 px x mean 3.0


def test_background_sum_pairs_with_size_not_a_bounding_box():
    # the sum is over the blob's own pixels, so it tracks `size` exactly
    stats = _stats()
    assert [int(v) for v in stats.size] == [4, 1]
    assert torch.allclose(stats.background_sum, stats.size.float() * 3.0)


def test_observed_counts_are_excess_plus_background():
    stats = _stats()
    observed = stats.intensity_sum + stats.background_sum
    assert float(observed[0]) == 112.0  # 100 excess + 12 background
    assert float(observed[1]) == 10.0  # 7 excess + 3 background


def test_background_sum_is_zero_when_no_mean_map_is_given():
    labels, excess, _ = _maps()
    stats = compute_blob_stats(
        labels,
        2,
        excess=excess,
        z=excess,
        log_bf=excess,
        posterior=excess,
        var=torch.ones(9, 9),
    )
    assert torch.all(stats.background_sum == 0.0)


def test_background_sum_survives_blob_selection():
    from probixi.peakfinding.peaks.blobs import select_blobs

    stats = _stats()
    kept = select_blobs(stats, torch.tensor([False, True]))
    assert float(kept.background_sum[0]) == 3.0


def test_peaks_only_gain_falls_back_to_geometry(tmp_path):
    from probixi.io.db import DuckDBOffloader

    off = DuckDBOffloader(tmp_path / "x.db", {"adu_per_photon": 2.0})
    assert off._gain() == 2.0
    assert off._gain(None) == 2.0


def test_indexed_gain_prefers_the_value_the_frame_was_processed_with(tmp_path):
    from probixi.io.db import DuckDBOffloader

    class R:
        adu_per_photon = 0.671

    off = DuckDBOffloader(tmp_path / "x.db", {"adu_per_photon": 12960.0})
    assert off._gain(R()) == 0.671  # measured gain wins over the geometry's
    assert off._gain() == 12960.0


def test_gain_rejects_nonsense_values(tmp_path):
    from probixi.io.db import DuckDBOffloader

    for bad in ({}, {"adu_per_photon": None}, {"adu_per_photon": 0.0}):
        off = DuckDBOffloader(tmp_path / "x.db", bad)
        assert off._gain() == 1.0


def test_peak_photometry_survives_the_index_peak_cap(geometry_dict, cell):
    # The cap keeps the brightest peaks by topk, whose indices are not in
    # position order, so the photometry has to be reindexed with them rather
    # than truncated. Truncation would pair a peak with another peak's counts.
    from probixi.indexer.indexer import Indexer, SeedConfig

    idxr = Indexer(geometry_dict, cell, seed=SeedConfig(max_index_peaks=6))
    n = 10
    positions = torch.stack([torch.arange(n).float()] * 2, dim=-1)
    # ascending, so topk picks the LAST six: its indices are not a prefix, and a
    # truncating implementation would hand peaks the wrong counts
    intensities = torch.arange(n).float()
    bg = intensities * 10.0  # tie each peak's photometry to its intensity
    npix = intensities + 100.0
    pos, inten, _, _, out_bg, out_npix = idxr._cap_peak_data(
        positions, intensities, None, None, bg, npix
    )
    assert len(pos) == 6 and len(out_bg) == 6 and len(out_npix) == 6
    # every surviving peak still carries its own photometry
    assert torch.equal(out_bg, inten * 10.0)
    assert torch.equal(out_npix, inten + 100.0)


def test_peak_photometry_follows_each_peeled_lattice(geometry_dict, cell):
    # Two lattices on one frame: peeling hands the second lattice the residue,
    # so each result's photometry must match its own surviving peaks.
    import sim

    from probixi.indexer.indexer import Indexer, RefineConfig, SeedConfig

    idxr = Indexer(
        geometry_dict,
        cell,
        seed=SeedConfig(
            n_directions=1500,
            n_spin=60,
            top_directions=12,
            max_candidates=32,
            max_lattices=2,
        ),
        refine=RefineConfig(max_iters=150, reassign_every=10),
    )
    p1, _ = sim.lattice_peaks(geometry_dict, cell, sim.proper_rotation(0, 8.0))
    p2, _ = sim.lattice_peaks(geometry_dict, cell, sim.proper_rotation(7, 8.0))
    positions = torch.cat([p1, p2]).to(torch.float32)
    n = len(positions)
    intensities = torch.arange(n).float() + 1.0
    found = idxr.index_lattices(
        {0: positions},
        intensities_by_frame={0: intensities},
        sigmas_by_frame={0: torch.ones(n)},
        weights_by_frame={0: torch.ones(n)},
        peak_bg_by_frame={0: intensities * 10.0},
        peak_npix_by_frame={0: intensities + 100.0},
    )
    lattices = found.get(0, [])
    assert len(lattices) == 2
    for r in lattices:
        assert r.peak_background_sum is not None
        assert len(r.peak_background_sum) == len(r.positions)
        assert len(r.peak_n_pixels) == len(r.positions)
        # the invariant that a truncating implementation would break
        assert torch.equal(r.peak_background_sum, r.intensities * 10.0)
        assert torch.equal(r.peak_n_pixels, r.intensities + 100.0)
