# Peakfinding

Indexing, integration, and eventually merging are only as good as the peaks you hand them. The usual way to find peaks is to pick a threshold, in ADU or some signal-to-noise proxy, and keep whatever pokes above it. The catch is that the right threshold drifts with the beam, the detector, the background, and the sample, so a value tuned on one run quietly throws away half the hits on the next.

`probixi` learns what the detector looks like with no crystal in the beam, then flags whatever is an outlier under that model.

## Modeling the background

The first job is to learn, per pixel, what "quiet" looks like. That means a mean $\mu$ and a variance $\sigma^2$, both accumulated online as frames stream past with Welford's method (or an exponentially-weighted variant when you want the model to track drift).

$$\delta = x - \mu,\qquad \mu \mathrel{+}= \tfrac{\delta}{n},\qquad M_2 \mathrel{+}= \delta\,(x - \mu_{\text{new}}),\qquad \sigma^2 = \tfrac{M_2}{n-1}.$$

A single per-pixel estimate is noisy early on, and it ignores the fact that detector backgrounds have *structure*. So `probixi` keeps three views of the background and blends them.

- **per-pixel**, the slow-to-settle estimate above.
- **radial**, a rotationally-symmetric model in $\sim$2 px annuli about the beam center.
- **per-panel**, a coarse per-tile estimate for panel-to-panel offsets.

The default blend, `shrinkage`, leans on the low-variance radial prior while evidence is thin and migrates to per-pixel as it accrues.

$$w = \min\!\Big(1,\ \tfrac{n}{n_{\text{warmup}}}\Big),\qquad \mu = w\,\mu_{\text{pixel}} + (1-w)\,\mu_{\text{radial}}.$$

Two details keep the model honest. **Robust clipping** caps each frame at $\mu + 5\sigma$ before folding it into the statistics, so the Bragg peaks we are hunting cannot inflate the variance we detect them against (a peak must not poison the stats used to find it). And after `warmup_frames` (default 16), any pixel whose across-frame variance is still exactly zero is declared dead and masked out.

## Whitening

With a mean and variance in hand, every frame becomes a **z-map**, how many standard deviations each pixel sits above its own quiet level.

$$z = \frac{x - \mu}{\sigma}.$$

Under a crystal-free frame, $z$ should look like standard normal noise, $\mathcal{N}(0,1)$. This expectation anchors everything downstream.

## Calibration

"Should look like $\mathcal{N}(0,1)$" and "does" are different things, so `probixi` spends a few dozen **seed frames** making it true. A few things get learned here, and each one is learned because a fixed value cannot know your detector.

**The blend weights and a variance scale.** Over a random subsample of background pixels, `probixi` searches the convex blend $w$ over $\{$pixel, radial, panel$\}$ that makes the background $z$ most normal, minimizing

$$\mathcal{L}(w) = \underbrace{m^2}_{\text{mean}\to 0} + \underbrace{(v-1)^2}_{\text{var}\to 1} + \tfrac{1}{2}\underbrace{\operatorname{kurt}^2}_{\text{tails}\to \text{normal}},$$

where $m$, $v$, and $\operatorname{kurt}$ are the mean, variance, and excess kurtosis of the background $z$. A residual variance scale $\text{var\_scale} = \sqrt{v}$ then rescales $\sigma$ so the calibrated background has *exactly* unit variance.

**The peak contrast $\kappa$ and the peak prior $\pi$.** On the calibrated $z$, a two-component EM mixture separates noise from signal, the background pinned at unit variance.

$$p(z) = (1-\pi)\,\mathcal{N}(0, 1) + \pi\,\mathcal{N}(0, \kappa^2).$$

$\kappa > 1$ is how much wider real Bragg pixels are than noise, the detection contrast. $\pi$ is the prior probability that any given pixel is peak. Both are properties of your instrument and sample, so `probixi` reads them off the data.

## Detecting

Given the calibrated model, `probixi` scores pixels one of two ways.

**The Bayesian approach.** For each pixel, weigh the peak hypothesis ($x \sim \mathcal{N}(\mu, (\kappa\sigma)^2)$) against the null ($x \sim \mathcal{N}(\mu, \sigma^2)$). The log Bayes factor is a clean function of $z$.

$$\log \mathrm{BF} = -\log\kappa + \tfrac{1}{2}z^2\Big(1 - \tfrac{1}{\kappa^2}\Big).$$

Add the log prior odds $\log\frac{\pi}{1-\pi}$, smooth the resulting logits with a small Gaussian so pixels in a real spot reinforce their neighbours, and squash through a sigmoid for a per-pixel posterior probability of "peak." Threshold at `posterior_threshold` (default 0.5) and you have candidate pixels.

**The matched filter.** Bragg spots have a size. The matched filter correlates the $z$-map against a bank of Gaussian kernels at a few scales (`mf_scales`, default $\sigma \in \{1.0, 1.6, 2.4\}$). Each kernel is normalized to unit energy, $u = K/\lVert K\rVert_2$, and its response is divided by the mask-aware kernel energy.

$$T_s = \frac{(z\!\cdot\!\text{mask}) \star u_s}{\sqrt{\sum u_s^2\,\text{mask}}}.$$

Under $z\sim\mathcal{N}(0,1)$ each scale's $T_s$ has $\operatorname{Var}=1$ everywhere, panel edges and masked pixels included, so a single cut means the same thing across the whole detector. Detection thresholds the per-pixel scale-space max $T = \max_s T_s$.

## Calibrating the threshold

After calibration each per-pixel $z$ is $\mathcal{N}(0,1)$ and each scale's $T_s$ has unit variance, but their max $T = \max_s T_s$ is **not** $\mathcal{N}(0,1)$. Taking a max biases it upward, and spatial correlation fattens its right tail, so a fixed $\sigma$ cut on $T$ would mean different things on different detectors.

`probixi` solves for the cut from a noise budget, as part of the same `calibrate` step. `probixi` measures the real null on the seed frames, marks quiet (crystal-free) frames with the expected maximum of $N$ Gaussians,

$$T_{\text{quiet}} = \mu_0 + \sigma_0\sqrt{2\ln N},$$

sweeps the threshold $T^\star$ over a grid, counts noise blobs on each quiet frame, and takes the smallest $T^\star$ whose *median* blob count meets the budget.

`--target-noise-peaks` overrides the default operating point. The matched filter carries a fixed fallback cut (`mf_threshold`, default $5\sigma$), and the budget calibration replaces it whenever a budget is set. One is set by default, so the calibrated cut normally wins. A budget of 5 lands near $5.5\sigma$. Lower it for stricter, cleaner peaks. Raise it for more sensitivity on weak data. It is under increased testing, and you should not need to touch it.

## Local background

A frozen, run-wide background cannot follow a smooth gradient or a shot-to-shot shift. So each frame also gets a **local** background, computed in a box annulus around every pixel (inner radius 4, outer 9 px), with the inner box left out so a peak never sits in its own estimate. The effective background blends the two,

$$\mu_{\text{eff}} = \mu + \mu_{\text{local}},\qquad \sigma^2_{\text{eff}} = \max(\sigma^2,\ \sigma^2_{\text{local}}),$$

and detection runs on $z = (x - \mu_{\text{eff}})/\sigma_{\text{eff}}$. This local test keeps faint spots alive on a sloped or drifting background, and it is the sensitivity that decides how many weak frames you index (see the [poorly-diffracting](../vignettes/bad_diffraction.md) vignette).

Each surviving blob then gets summarized. An intensity $\sum(\text{excess})$, a $\sigma = \sqrt{\sum \text{var}}$, a peak $z$, plus shape terms (eccentricity, peakedness) that reject cosmic rays and streaks.

## Low flux

The running model freezes one variance, measured at the full-dose background. Whitening a genuinely dim, low-flux frame against that variance over-suppresses its real spots. Turning on `flux_variance` (in the Python API) fits a photon-transfer curve so the variance scales with each frame's own level,

$$\sigma^2 = \text{read\_var} + \text{gain}\cdot\text{level},$$

with a sub-photon floor (`flux_var_floor`, default 0.15 of the running variance) so nothing collapses to zero. It is opt-in, aimed at XFEL/SFX or jet-intensity-variable data. This may be exposed to the CLI in the future; however, more testing is still needed to determine if it makes a significant impact.

## Safeguards

Saturated, dead, and geometry-flagged pixels (anything at or above `max_adu`, plus the file's own mask) stay out of the background and out of detection from the very first frame. One hot pixel left in would poison an entire radial bin. And if a single frame produces more than `max_peaks` (default 1000) blobs, that points to a failed background model, so `probixi` emits nothing for that frame.

The result is a peak finder with essentially no thresholds to set by hand. You give it seed frames and a noise budget, and it calibrates the rest against your data. Where those peaks go next is [Indexing](indexing.md).
