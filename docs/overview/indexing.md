# Indexing and integration

[Peakfinding](peakfinding.md) produces a list of peaks, each with a detection posterior $c_i$. Indexing finds the crystal orientation that places those peaks on reciprocal-lattice points. Prediction and integration then measure every reflection the lattice implies, whether or not it was detected. Constants and defaults are listed in the [method reference](method.md).

## Orientation matrix

A reflection with Miller indices $\mathbf h = (h,k,l)$ appears at the scattering vector

$$\mathbf q = A\,\mathbf h, \qquad A = U B,$$

where $B$ is the reciprocal basis of the unit cell and $U$ the crystal's orientation. The cell file fixes $B$, so indexing is a search over the rotation $U$, three degrees of freedom.

## Peaks in reciprocal space

Each peak centroid is mapped to the laboratory frame through its panel's position and fast-scan and slow-scan vectors. With $\hat{\mathbf s}$ the unit vector from the sample to the pixel and $\hat{\mathbf s}_0 = (0,0,1)$ the beam, the Ewald construction gives

$$\mathbf q = \frac{\hat{\mathbf s} - \hat{\mathbf s}_0}{\lambda},$$

in the crystallographic convention without $2\pi$, so $|\mathbf q|$ is the reciprocal of the resolution. A peak is explained by $A$ when $\mathbf h = \operatorname{round}(A^{-1}\mathbf q)$ predicts $\mathbf q$ back within a tolerance of a quarter of the shortest reciprocal axis. The tolerance therefore scales with the cell.

## Orientation search

A blind search over all rotations is too expensive. Following TORO, `probixi` factors the rotation into the direction of the crystal's $\mathbf a$ axis (two degrees of freedom) and a roll about it (one).

### Direction fitness

Trial directions $\hat{\mathbf d}$ are laid out on a Fibonacci sphere with both signs, $\max(6000,\;6000\,(|\mathbf a|/55\,\text{Å})^2)$ points by default so the spacing follows the fitness peak width $\sim 1/(|\mathbf a| q)$. Each is scored with the trial vector $\mathbf t = |\mathbf a|\,\hat{\mathbf d}$ by

$$F(\hat{\mathbf d}) = \frac{\sum_i c_i \cos\!\left(2\pi\,\mathbf t\cdot\mathbf q_i\right)}{\sum_i c_i}.$$

Because $\mathbf a\cdot\mathbf b^* = \mathbf a\cdot\mathbf c^* = 0$ and $\mathbf a\cdot\mathbf a^* = 1$, the product $\mathbf a\cdot\mathbf q_i$ equals the integer $h_i$ for every reflection on the lattice. Every cosine is then 1 when $\mathbf t$ is the true $\mathbf a$ axis, and $F$ reaches its upper bound. For any other direction the products are not all integers and $F < 1$. Weighting by $c_i$ lets a weak detection vote in proportion to its evidence.

### Roll search

The 64 best directions are each spun through 120 rolls. Every resulting $A = UB$ is scored by the $c$-weighted count of peaks it explains, and the 64 best go to refinement.

Seeding uses at most the 80 brightest peaks. A frame with fewer than five peaks is not indexed, and on a frame with fewer than 30 the direction and roll counts are doubled.

## Orientation refinement

All candidates of all frames in a batch are refined together on the GPU. Each candidate carries an axis-angle perturbation $\boldsymbol\omega$, applied as $A_{\text{eff}} = R(\boldsymbol\omega)A$, and Adam minimises

$$\ell(\boldsymbol\omega) = \frac{\sum_{i\in\mathcal I} c_i\,\lVert A_{\text{eff}}\mathbf h_i - \mathbf q_i\rVert^2}{\sum_{i\in\mathcal I} c_i} + \mathbf 1\big[|\mathcal I| < 6\big],$$

with the inlier set $\mathcal I$ and indices $\mathbf h_i$ recomputed every 10 steps. As the orientation tightens, more peaks fall within tolerance and join the fit. The indicator is constant in $\boldsymbol\omega$, so it marks under-constrained candidates without steering the fit.

Candidates are ranked by their weighted inlier sum. The first with at least six inliers is refined further by Levenberg-Marquardt over the rotation and the cell parameters the lattice type allows, under a Gaussian prior that keeps the cell near its starting value. In that step the residual along the scattered beam, which measures the excitation error rather than the orientation, is weighted separately from the perpendicular residual.

## Cell-match acceptance

A candidate is accepted only when its refined cell matches the target, with sorted edges within 5% and sorted angles within 3°. Sorting makes the check independent of axis labelling. A frame with no acceptable candidate is reported as unindexed. With `--max-lattices` above 1, the peaks an accepted lattice explains are removed and the remainder is searched again.

## Target-cell and camera-length correction

The cell file need only be approximate. At calibration, and again after each of the first two sets of 200 accepted lattices, `probixi` replaces the target by the component-wise median of the refined cells when that median lies within the match window of the cell file. The cell and the camera length are degenerate only at low angle. A camera-length error moves a spot radially in proportion to its radius $\rho$, a cell-scale error by $\rho/\cos 2\theta$, so `probixi` fits the camera length to that curvature in the radial residuals of the seed lattices' indexed peaks, with each crystal's scale profiled out. A shift larger than 0.1% and five standard errors is applied, and the seeds are indexed again at the new distance, up to three times. `--no-cell-calibrate` disables both.

## Spot prediction

For an accepted $A$, `probixi` enumerates every $\mathbf h$ within the detector's resolution that the lattice centring allows. With $\mathbf S = \lambda\mathbf q + \hat{\mathbf z}$, a reflection diffracts exactly when $\lVert\mathbf S\rVert = 1$, and the excitation error is $\varepsilon = \lVert\mathbf S\rVert - 1$. Finite domain size, mosaic spread $\eta$, and bandwidth thicken the Ewald sphere into a shell of half-width

$$\rho(|\mathbf q|) = r_{\text{size}} + \tfrac12\,\eta\,|\mathbf q| + \tfrac12\,\lambda\,\frac{\Delta\lambda}{\lambda}\,|\mathbf q|^2,$$

and a reflection is predicted when $|\varepsilon| < 1.5\,\lambda\,\rho$. The mosaicity is fitted per crystal. The radial offsets of its indexed reflections are absolute values of draws whose spread grows linearly with $|\mathbf q|$. For $X \sim \mathcal N(0,\varsigma^2)$, $\mathbb E|X| = \varsigma\sqrt{2/\pi}$, so the through-origin slope of the offsets against $|\mathbf q|$, scaled by $\sqrt{\pi/2}$, estimates $\eta/2$. Below eight indexed reflections a prior of 0.1° is used.

## Integration

### Integration aperture

After calibration, `probixi` stacks the radial profiles of peaks from a further random sample of frames and fits a Gaussian to the core. The full-flux radius is where that Gaussian falls to 2% of its centre. The background annulus begins one pixel beyond it and holds 120 pixels. The signal disk defaults to the smaller radius that maximises the background-limited signal-to-noise ratio of the profile. A smaller disk captures a smaller fraction of each spot. A constant fraction is absorbed by each crystal's scale factor in merging, and a trend with resolution common to all crystals by an overall $B$ factor. What scaling cannot remove is a resolution trend that differs between crystals, so `probixi` tests for that on the calibration peaks and reverts to the full-flux radius when it is significant and larger than the extra background noise of the bigger disk (`--aperture auto`).

That profile is centred on each peak's own centroid. A reflection with no found peak nearby is integrated at its predicted position instead, which adds the prediction error to the spot. The error grows with resolution, largely along the radius, and at high resolution a disk sized for the spot alone captures only a fraction of the flux, a smaller one than for the same reflection when it is strong enough to be found. `probixi` therefore stacks the excess around the predicted positions of the indexed seed lattices in twelve resolution shells and gives these reflections, shell by shell, the smallest radius within 2% of the best signal-to-noise ratio of the stacked profile. The radius never falls below the signal disk or with resolution, and the background annulus moves out with it.

### Intensity and uncertainty

Each predicted reflection is first moved onto an observed peak centroid within 5 pixels, if there is one. With $\bar b$ and $\sigma_b^2$ the mean and variance of the $n_b$ annulus pixels, and $n$ the disk pixels the reflection owns,

$$I = \sum_{j=1}^{n}(x_j - \bar b), \qquad \sigma^2(I) = n\,\sigma_b^2\left(1 + \frac{n}{n_b}\right) + g\max(I, 0).$$

The variance follows from the independence of disk and annulus. The $n$ background pixels contribute $n\sigma_b^2$, the subtracted mean of $n_b$ pixels contributes $n^2\sigma_b^2/n_b$, and the signal contributes its own shot noise with gain $g$ in detector units per photon. Where disks overlap, each pixel belongs to the nearest predicted centre, so no pixel is counted twice. Reflections closer than 1.5 disk radii to another on the same panel are dropped.

### Per-crystal resolution limit

Every reflection is integrated to the detector edge. `probixi` reports a per-crystal resolution limit, the more generous of the second-highest-resolution indexed peak (one peak that a dense high-resolution lattice indexes by chance cannot set it) and the point where the shell-averaged $I/\sigma(I)$ falls to 0.5, but leaves the cutoff to the merge. A per-crystal cutoff would bias the merged intensities upward.

## The enrichment gate

An orientation can fit a sparse peak list without being the true lattice, so `probixi` tests each predicted lattice against the image. A pixel is bright when the maximum of $z$ over its 5 by 5 neighbourhood exceeds the detection threshold, and $p$ is the fraction of valid pixels that are bright. Of the $M$ predicted positions, $n_{\text{bright}}$ are bright. Under the null that the prediction is unrelated to the frame, $n_{\text{bright}}$ is binomial with mean $Mp$, which the Poisson law approximates for small $p$, so

$$\text{enrichment} = \frac{n_{\text{bright}}/M}{p}, \qquad \texttt{enrich\_p} = P(X \ge n_{\text{bright}}),\quad X \sim \mathrm{Poisson}(Mp).$$

An enrichment near 1 means the predictions land on signal as often as chance. The bright spots a lattice was fitted to cannot prove it real, so after integration it must also carry signal at the predictions no found peak explains: the summed $I/\sigma$ over the unsnapped predictions inside the crystal's resolution limit, $z = \sum I/\sigma/\sqrt n$, is $\sim\mathcal N(0,1)$ for a chance lattice, and `enrich_p` is the larger of the two chance probabilities. With `--enrich-gate`, a lattice is kept only when `enrich_p` is at most `--enrich-alpha` (default $10^{-3}$). The gate runs on each lattice as it is found, before the next is searched and before the overlap exclusion, so a rejected lattice cannot remove reflections from one that is kept. Its peaks are still removed before the next search. The gate is off by default.
