from __future__ import annotations

import numpy as np
import sim
import torch

from probixi.peakfinding.noise.calibrate import (
    calibrate_noise,
    calibrate_threshold,
    fit_photon_transfer,
)
from probixi.peakfinding.noise.model import NoiseModel

DTYPE = torch.float64


def _frames(shape, n, *, background=100.0, noise_sigma=10.0, seed=0):
    arr = sim.simulate_noise_frames(
        shape, n, background=background, noise_sigma=noise_sigma, seed=seed
    )
    return [torch.from_numpy(f).to(DTYPE) for f in arr]


def test_update_recovers_background_mean_and_noise_variance():
    shape = (96, 96)
    background, noise_sigma, n = 100.0, 10.0, 40
    frames = _frames(shape, n, background=background, noise_sigma=noise_sigma, seed=1)
    # disable robust clipping: pure background, want the raw streaming estimate
    nm = NoiseModel(shape, mode="online", robust_update=False, dtype=DTYPE)
    nm.fit(frames)
    pred = nm.predict(combination="pixel")
    mean = pred["mean"]
    var = pred["var"]
    # SE of the mean is noise_sigma/sqrt(n); allow a few SE across all pixels.
    se_mean = noise_sigma / (n**0.5)
    assert float((mean - background).abs().mean()) < 5.0 * se_mean
    assert abs(float(mean.mean()) - background) < 2.0 * se_mean
    # var ~ sigma^2; mean over pixels concentrates by 1/sqrt(n).
    assert abs(float(var.mean()) - noise_sigma**2) < 0.1 * noise_sigma**2


def test_robust_update_not_inflated_by_bright_peaks():
    shape = (96, 96)
    background, noise_sigma, n = 100.0, 10.0, 40
    clean = sim.simulate_noise_frames(
        shape, n, background=background, noise_sigma=noise_sigma, seed=2
    )
    poisoned = clean.copy()
    # inject a cluster of very bright counts at a fixed pixel, only after the
    # background has warmed up so robust clipping has a sigma to clip against
    rr, cc = 48, 48
    for i in range(10, n, 5):
        poisoned[i, rr - 1 : rr + 2, cc - 1 : cc + 2] = 50000.0

    robust = NoiseModel(shape, mode="online", robust_update=True, dtype=DTYPE)
    robust.fit([torch.from_numpy(f).to(DTYPE) for f in poisoned])
    naive = NoiseModel(shape, mode="online", robust_update=False, dtype=DTYPE)
    naive.fit([torch.from_numpy(f).to(DTYPE) for f in poisoned])

    rp = robust.predict(combination="pixel")
    npred = naive.predict(combination="pixel")
    # the naive estimate blows up at the poisoned pixel: mean and var explode
    assert float(npred["mean"][rr, cc]) > background + 100.0 * noise_sigma
    assert float(npred["var"][rr, cc]) > 1e4 * noise_sigma**2
    # robust clipping keeps the mean near truth and the var orders of magnitude
    # below the naive value (a single early hit can still leak a little spread)
    assert float(rp["mean"][rr, cc]) < background + 5.0 * noise_sigma
    assert float(rp["var"][rr, cc]) < 1e-3 * float(npred["var"][rr, cc])
    # an unpoisoned control pixel matches the clean background statistics
    assert abs(float(rp["mean"][10, 10]) - background) < 5.0 * noise_sigma
    assert abs(float(rp["var"][10, 10]) - noise_sigma**2) < 0.5 * noise_sigma**2


def test_bad_pixels_masked_after_warmup():
    shape = (96, 96)
    n = 24
    clean = sim.simulate_noise_frames(shape, n, seed=3)
    sat = (10, 20)  # forced saturated every frame -> zero variance
    dead = (70, 60)  # forced dead every frame -> zero variance
    frames = clean.copy()
    frames[:, sat[0], sat[1]] = sim.SATURATED_VALUE
    frames[:, dead[0], dead[1]] = sim.DEAD_VALUE

    nm = NoiseModel(shape, mode="online", warmup_frames=8, dtype=DTYPE)
    nm.fit([torch.from_numpy(f).to(DTYPE) for f in frames])

    mask = nm.predict(combination="pixel")["mask"]
    # saturated and dead pixels have zero spread -> masked out after warmup
    assert not bool(mask[sat[0], sat[1]])
    assert not bool(mask[dead[0], dead[1]])
    # a normal background pixel stays valid
    assert bool(mask[48, 48])


def test_calibrate_noise_whitens_background_and_is_deterministic():
    shape = (96, 96)
    frames = _frames(shape, 32, background=100.0, noise_sigma=10.0, seed=4)
    nm = NoiseModel(shape, mode="online", dtype=DTYPE)
    cal = calibrate_noise(nm, frames, rng_seed=7)
    cal.apply(nm)

    z = nm.z_score(frames[0], combination="calibrated")
    valid = nm.valid_mask
    zv = z[valid]
    assert abs(float(zv.mean())) < 0.15
    assert abs(float(zv.std()) - 1.0) < 0.15
    assert cal.kappa > 1.0
    assert 0.0 < cal.prior_peak < 1.0

    # deterministic for a fixed rng_seed
    nm2 = NoiseModel(shape, mode="online", dtype=DTYPE)
    cal2 = calibrate_noise(nm2, _frames(shape, 32, seed=4), rng_seed=7)
    assert cal2.kappa == cal.kappa
    assert cal2.prior_peak == cal.prior_peak
    assert cal2.var_scale == cal.var_scale
    assert cal2.weights == cal.weights


def test_calibrate_threshold_meets_target_and_stays_in_grid():
    shape = (96, 96)
    frames = _frames(shape, 32, background=100.0, noise_sigma=10.0, seed=5)
    nm = NoiseModel(shape, mode="online", dtype=DTYPE)
    calibrate_noise(nm, frames, rng_seed=0).apply(nm)

    target = 5.0
    grid_min, grid_max = 3.0, 8.0
    tc = calibrate_threshold(
        nm,
        frames,
        target_noise_peaks=target,
        threshold_grid_min=grid_min,
        threshold_grid_max=grid_max,
        rng_seed=0,
    )
    # signal-free frames: the achieved noise-blob rate meets the target
    assert tc.achieved_noise_peaks <= target
    assert grid_min <= tc.threshold <= grid_max
    assert tc.n_quiet_frames >= 4


def test_fit_photon_transfer_recovers_positive_gain():
    shape = (96, 96)
    # poisson-like data: mean varies across pixels and var ~ read_var + gain*mean.
    rng = np.random.Generator(np.random.PCG64(11))
    read_sigma, gain = 3.0, 2.0
    levels = rng.uniform(20.0, 400.0, size=shape).astype(np.float64)
    n = 32
    frames = []
    for i in range(n):
        sub = np.random.Generator(np.random.PCG64(100 + i))
        var = read_sigma**2 + gain * levels
        f = sub.normal(levels, np.sqrt(var)).astype(np.float64)
        frames.append(torch.from_numpy(f).to(DTYPE))

    nm = NoiseModel(shape, mode="online", robust_update=False, dtype=DTYPE)
    calibrate_noise(nm, frames, rng_seed=0).apply(nm)
    read_var, fit_gain = fit_photon_transfer(nm)
    assert read_var >= 0.0
    assert fit_gain > 0.0
    # gain is the dominant var-vs-level slope; should land in the right ballpark
    assert 0.5 * gain < fit_gain < 2.0 * gain
    assert nm.gain == fit_gain
    assert nm.read_var == read_var
