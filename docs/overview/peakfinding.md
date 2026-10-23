# Peakfinding

A conventional peakfinder keeps whatever exceeds a threshold, in detector units or in a signal-to-noise proxy. The appropriate threshold depends on the beam, the detector, the background, and the sample, so a value tuned on one run does not transfer to the next. `probixi` replaces the threshold with a hypothesis test whose null distribution is learned from the data. Constants and defaults are listed in the [method reference](method.md).

## Stationary background and sparse signal

1. Every frame records the same detector under the same beam. The background at pixel $i$ is therefore a draw from one distribution $F_i$, common to all frames and changing only slowly within a run.
2. Bragg signal is sparse. At any pixel only a small fraction of observations carry signal.

The first property makes $F_i$ learnable from the stream. The second makes the learning robust, because almost every value that reaches the estimator is background.

Detection is then a test at every pixel. Under the null $H_0$ the value $x_i$ is a background draw with mean $\mu_i$ and variance $\sigma_i^2$. Under the alternative $H_1$ it is background plus a positive excess. `probixi` estimates $(\mu_i, \sigma_i)$ online, forms the whitened residual

$$z_i = \frac{x_i - \mu_i}{\sigma_i},$$

which is standard normal under $H_0$ once the model is calibrated, and tests $z$.

## The background model

`probixi` accumulates the mean and variance with Welford's update, which avoids the cancellation that $\sum x^2 - n\bar x^2$ suffers in single precision. A per-pixel estimate needs many frames to stabilise and ignores the spatial structure of a detector background, so three estimators of the same form run on the same stream.

- The pixel estimator keeps one estimate per pixel.
- The radial estimator pools pixels into annuli 2 pixels wide about the beam centre, which captures solvent and air scatter, symmetric about the beam.
- The frame-wide estimator pools the pixels of each geometry panel, or of the whole frame on a single-panel detector, which captures an offset common to a panel, such as a pedestal shift.

The prediction is a convex blend of the three means with one scale on the per-pixel standard deviation,

$$\mu_i = \sum_k w_k\,\mu_{k,i}, \qquad \sigma_i = s\,\sigma_{\text{pix},i}.$$

A Bragg peak that enters the update inflates the variance against which later peaks are tested. Each frame is therefore clipped from above at $\mu_i + 5\sigma_i$ before the update. The clip is one-sided, so a downward excursion updates the model as usual and a peak contributes at most $5\sigma_i$. Pixels whose variance is still zero after 16 frames are masked as dead.

## Calibration of the null and peak hypotheses

The test assumes $z \sim \mathcal N(0,1)$ on background, which an arbitrary blend does not satisfy. `probixi` therefore chooses $(w, s)$ on a set of seed frames drawn at random from the run. On a random subsample of pixels it labels the background set $\{|z| < 3\}$ using the pixel mean alone, so the set does not depend on the blend. For each $w$ on a simplex grid of spacing 0.1 it evaluates

$$\mathcal L(w) = \bar z^{\,2} + (v - 1)^2 + \tfrac12\,\mathrm{kurt}^2,$$

where $\bar z$, $v$, and $\mathrm{kurt}$ are the mean, variance, and excess kurtosis of $z$ over that set. The three terms are the departures of location, scale, and tail shape from their standard-normal values, and the loss is zero only when all three agree. The grid holds 66 points, so the search is exhaustive. At the minimiser $v$ is not in general 1, and setting $s = \sqrt v$ restores unit variance exactly.

For the alternative, `probixi` models a peak pixel as $z \sim \mathcal N(0, \kappa^2)$ with $\kappa > 1$, the weakest assumption consistent with an excursion of unknown size. On the calibrated residuals, expectation maximisation fits

$$p(z) = (1-\pi)\,\mathcal N(0,1) + \pi\,\mathcal N(0,\kappa^2)$$

with the background component fixed, which gives the peak contrast $\kappa$ and the prior $\pi$ that a pixel carries signal.

## Per-pixel evidence

With $\kappa$ and $\pi$ known, the log Bayes factor of peak against background is a function of $z$ alone,

$$\log \mathrm{BF}_i = -\log\kappa + \tfrac12 z_i^2\left(1 - \kappa^{-2}\right),$$

set to zero wherever the excess is not positive, because a downward fluctuation is not a peak. Adding the log prior odds $\log[\pi/(1-\pi)]$ gives posterior log odds. These are averaged over a small Gaussian window, which lets the pixels of one spot reinforce each other while a single bright pixel cannot, and mapped through the logistic function to a posterior $c_i \in [0,1]$. The posterior does not decide detection. It is the weight each peak carries into [indexing](indexing.md).

## Per-frame background correction

The running model is an average over the run. A given frame can sit above or below that average through a pedestal shift or a smooth gradient. `probixi` corrects the model on every frame from a square annulus around each pixel, excluding the inner 9 by 9 box and extending to 19 by 19,

$$\mu^{\text{eff}}_i = \mu_i + \operatorname{mean}_{\mathcal A_i}\!\big[\min(r, 5\sigma)\big], \qquad (\sigma^{\text{eff}}_i)^2 = \max\!\Big(\sigma_i^2,\ \operatorname{var}_{\mathcal A_i}\!\big[\min(r, 5\sigma)\big]\Big),$$

with $r = x - \mu$ the residual against the running model. A spot of radius up to 4 pixels never enters its own background estimate, and the clip limits what a neighbouring reflection can contribute. Taking the larger variance means a frame with more local scatter than the model is tested against its own scatter, and a frame with less is not given a lower bar. Detection uses $z$ formed from $\mu^{\text{eff}}$ and $\sigma^{\text{eff}}$.

## The matched filter

A Bragg spot covers several pixels, and a single-pixel test discards that extent. For a known template in white noise the optimal linear filter is the template itself. After whitening the background is white to the extent that calibration succeeded, so `probixi` correlates $z$ with Gaussian templates of width 1.0, 1.6, and 2.4 pixels and takes the maximum response.

One threshold must mean the same thing at every pixel, including beside panel gaps where part of the template falls on masked pixels. With $u_s$ the template scaled to unit energy and $m \in \{0,1\}$ the mask,

$$T_s(i) = \frac{\sum_j u_{s,j}\,m_{i+j}\,z_{i+j}}{\sqrt{\sum_j u_{s,j}^2\,m_{i+j}}}, \qquad T(i) = \max_s T_s(i).$$

Under $H_0$ the numerator has variance $\sum_j u_{s,j}^2 m_{i+j}^2 = \sum_j u_{s,j}^2 m_{i+j}$, because $m^2 = m$, which is the square of the denominator. Each $T_s$ therefore has unit variance at every pixel, whatever fraction of the template is masked.

## The threshold from a noise budget

Each $T_s$ has unit variance, but $T$ is not standard normal. The maximum over widths shifts its location upward, and the template correlates neighbouring values, which fattens the right tail. Both effects depend on the detector and the mask. `probixi` measures the null on the seed frames instead, taking $\mu_0$ as the median of the pooled $T$ and $\sigma_0$ from the values below the median reflected about it. Peak pixels lie in the right tail and enter neither estimate.

A seed frame is quiet when its largest $T$ falls below

$$T_{\text{quiet}} = \mu_0 + \sigma_0\sqrt{2\ln N},$$

the leading-order maximum of $N$ independent normal values, with $N$ the number of valid pixels. A frame whose maximum exceeds this carries something the null does not explain. On the quiet frames `probixi` sweeps candidate thresholds from 3 to 8 and counts the local maxima above each. The median count does not increase with the threshold, so the smallest threshold whose median count is at or below the budget is the most sensitive one that meets it.

Input that holds only hits, such as Cheetah's hit files, has no quiet frame, and counting on its quietest frames would count Bragg peaks as noise. `probixi` then counts the local maxima of the mirrored statistic $-\min_s T_s$ on every seed frame instead. Positive signal never reaches that tail, and near-Gaussian noise reaches it as often as the positive one.

The budget, five noise detections per signal-free frame by default (`--target-noise-peaks`), is the one operating point left in the method. It is stated in units of noise detections per frame, which mean the same thing on every detector.

## Peak extraction and shape filters

Pixels above the threshold are grouped into 8-connected components. Each component is reduced to a centroid weighted by the positive excess, a summed excess with its uncertainty, and two shape terms. A component is kept when it has at least two pixels, its second-moment eigenvalue ratio is at most 5, which rejects streaks and cosmic-ray tracks, and its maximum exceeds 1.2 times its mean, which rejects flat regions. The size limit follows from the filter. A Gaussian response of height $R$ exceeds the threshold over an area $2\pi\sigma_s^2\ln(R/T^\star)$, and a component may cover four times that area at the largest width, or 30 pixels if that is larger. A frame with more than 1000 components indicates a failure of the background model and yields no peaks.

## Masks learned from the data

Pixels that carry no measurement are excluded from every estimate and every test. The static mask combines the geometry's bad regions, panel edges, saturated pixels, and any mask stored with the data. Three more are learned.

- Dead pixels, with zero variance after 16 frames.
- Static shadows, such as a beamstop holder. A pixel is dark when its running mean lies far below the radial mean at the same radius. Dark regions are grown by 9 pixels to keep every local-background annulus clear of the edge, and the model is refitted without them.
- The beamstop halo. The seed peaks are histogrammed by $|\mathbf q|$. When the innermost bins hold at least 50 times the median peak density, the exclusion is extended outward, by at most two bins, to where the density falls to a tenth of the spike. This also sets the low-resolution limit the output reports.

Frames that received no beam, with a median level below a tenth of the run's, are withheld from the running statistics but still pass through detection (`--force-all` disables this).

## Flux-dependent variance

The running variance is measured at the run's typical fluence. With the flux model ([vignette](../vignettes/variable_flux.md)), chosen at calibration when the background level changes between shots beyond photon noise or forced with `--flux-variance`, `probixi` fits $\sigma^2 = \sigma^2_{\text{read}} + g\,\mu$ to the calibrated model and whitens each pixel against the variance predicted at that frame's own level, bounded below by 0.15 of the running variance.

## Frozen parameters and drift

The blend, the scale, $\kappa$, $\pi$, and the threshold are frozen after calibration. The running statistics continue to update and follow slow drift within the run. They average the whole run unless `--noise-decay d` is set, which makes them forget with a memory of about $1/d$ frames, for a sample or beam that changes within the run. `--recalibrate-every` instead rebuilds the calibration at fixed intervals and freezes the statistics in between, which suits a run that spans an abrupt change. Every run is whitened to the same unit-variance standard, so peak counts and indexing rates are comparable between conditions.
