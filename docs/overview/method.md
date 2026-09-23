# The method, end to end

This page derives what `probixi` computes, in the order it computes it: from a raw detector frame to the reflection list a merging program reads. Every symbol is defined where it first appears and reused afterwards; the notation table at the end collects them. Defaults are quoted with the name of the configuration field or module constant that holds them, and each quantity is marked as **fixed** (a constant in the code) or **learned** (estimated from the data at hand). File and line references point into the package.

## 0. Objects and conventions

A run is a sequence of frames $x_f \in \mathbb{R}^{H\times W}$, $f = 1, \dots, F$, in detector units (ADU; for a photon-counting detector one ADU is one photon). A pixel is indexed by $p = (i, j)$ with $i$ the slow-scan row and $j$ the fast-scan column; CrystFEL's `fs`/`ss` are $j$/$i$. A static Boolean mask $m(p) \in \{0, 1\}$ marks live pixels. Three coordinate frames appear:

- **detector** $(i, j)$ in pixels;
- **laboratory** $(x, y, z)$ in metres, $z$ along the beam, given by the geometry's per-panel origin and fast/slow basis vectors;
- **reciprocal space** $\mathbf{q} \in \mathbb{R}^3$ in $\text{Å}^{-1}$ (streams and databases use $\text{nm}^{-1}$: multiply by $10$, `A_INV_TO_NM_INV`).

Wavelength $\lambda$ (Å) and camera length $L$ (m) come from the geometry file; the detector gain $g$ (ADU per photon) is read from the geometry when declared and re-measured from the data when the flux-variance model is on (§2.6).

## 1. The background model

### 1.1 Streaming moments

For each of three "views" of the detector, `probixi` maintains a running mean $\mu$ and second central moment $M_2$ over frames with Welford's update (`NoiseStats.update`, `peakfinding/noise/model.py:40-56`). With $n$ the number of frames seen so far and $y$ the projected, clipped frame (§1.3):

$$
\delta = y - \mu, \qquad \mu \leftarrow \mu + \frac{\delta}{n}, \qquad M_2 \leftarrow M_2 + \delta\,(y - \mu_{\text{new}}), \qquad \hat\sigma^2 = \frac{M_2}{\max(n-1, 1)} .
$$

With `decay` $= \alpha < 1$ the update is exponentially weighted, $\mu \leftarrow \mu + \alpha\delta$, $M_2 \leftarrow (1-\alpha) M_2 + (1-\alpha)\alpha\,\delta^2$, and $\hat\sigma^2 = M_2$. The pipeline default is $\alpha = 1$ (plain running statistics).

### 1.2 Three views

- **Pixel** (`_pixel.py`): $\mu_{\text{pix}}(p)$, $\sigma^2_{\text{pix}}(p)$ per pixel, fed without a mask.
- **Rotational** (`_radial.py`): with $r(p)$ the lab-frame radius of pixel $p$ from the beam centre in pixels and bin width $\Delta = 2$ px (`radial_bin_width`, fixed), bin $b(p) = \lfloor r(p)/\Delta \rfloor$; each frame is projected to masked bin means $\bar y_b = \sum_{p\in b} m(p)\,y(p) / \max(\sum_{p\in b} m(p), 1)$, and the running statistics live on the bins. Per pixel, $\mu_{\text{rot}}(p) = \mu_{b(p)}$ and $\sigma^2_{\text{rot}}(p) = \hat\sigma^2_{b(p)}\,N_{b(p)}$ with $N_b$ the bin's pixel count (the variance of a bin *mean* re-inflated to a single pixel).
- **Panel** (`_panel.py`): the same construction per detector panel $\pi(p)$.

### 1.3 Robust update and dead pixels

Before a frame enters any running statistic it is clipped from above against the pixel view (`NoiseModel._robust_clip`, `model.py:252-263`), so that Bragg peaks cannot inflate the variance they are later detected against:

$$
\tilde x(p) = \min\!\big(x(p),\ \mu_{\text{pix}}(p) + k_r\,\sigma_{\text{pix}}(p)\big), \qquad k_r = 5 \ (\texttt{robust\_k}, \text{fixed}),
$$

applied once $n \ge 4$ frames (`robust_min_frames`). After `warmup_frames` (default 16) any pixel whose variance is still exactly zero is declared dead: $m(p) \leftarrow m(p) \wedge [\hat\sigma^2_{\text{pix}}(p) > 0]$ (`commit_dead_pixel_mask`). The a-priori part of $m$ comes from the geometry (bad regions, panel edges, `max_adu`) and is later ANDed with the shadow (§2.1) and beam-stop (§2.8) exclusions.

### 1.4 The blended prediction

The model's prediction for a frame is a convex blend of the three means with weights $w_k$, $k \in \{\text{pix}, \text{rot}, \text{pan}\}$, and one source's variance (`NoiseModel.predict`, `model.py:348-404`):

$$
\mu_0(p) = \sum_k w_k\,\mu_k(p), \qquad \sigma_0^2(p) = s_v^2\,\sigma^2_{\text{pix}}(p),
$$

where $s_v$ is the variance scale of §2.2. Before calibration the `shrinkage` preset uses $w_{\text{pix}} = \min(1, n/\texttt{warmup\_frames})$, $w_{\text{rot}} = 1 - w_{\text{pix}}$, $s_v = 1$; after calibration the learned weights are used.

## 2. Calibration on the seed frames

Calibration runs once on $F_s$ randomly drawn **seed frames** (`--seed-frames`, 1000 in the benchmarks) after they have warmed the model (`Probixi.calibrate`, `probixi.py`). Everything in this section is learned unless marked fixed.

### 2.1 Static shadows

A beam-stop arm or mount leaves a region far below the radial model at its own radius. With the pixel and rotational means from the warm-up (`shadow_mask`, `probixi.py`):

$$
\rho(p) = \frac{\mu_{\text{pix}}(p)}{\mu_{\text{rot}}(p)}, \qquad \ell(p) = \log\rho(p), \qquad \text{cut} = \min\!\big(\operatorname{med}(\ell) - 6\,\text{MAD}(\ell),\ \log\tfrac12\big),
$$

with $\text{MAD}$ the median absolute deviation scaled by $1.4826$ (a robust $\sigma$; the constants 6 and $\tfrac12$ are fixed). Dark pixels are $\ell(p) < \text{cut}$; a shadow *core* is a dark pixel whose $(2r_{\text{out}}+1)^2$ window is at least half dark, with $r_{\text{out}}$ the local-background radius of §3.1; the mask is the core grown by $r_{\text{out}}$ so that no background annulus can straddle the edge. When a shadow is found the model is reset and re-warmed with it excluded, so no view ever averages over it. Without a static gradient this masks nothing.

### 2.2 Blend weights and variance scale

Over a random subsample of $S = 40\,000$ live pixels and all seed frames, with $\sigma(p) = \sigma_{\text{pix}}(p)$, the *background set* is $\mathcal{B} = \{|x - \mu_{\text{pix}}| < 3\sigma\}$. For every weight triple on a step-$0.1$ simplex the calibrated residual $z = (x - \sum_k w_k\mu_k)/\sigma$ restricted to $\mathcal{B}$ is summarised by its mean $\bar z$, population variance $v$ and excess kurtosis $\text{kurt}$, and (`calibrate_noise`, `noise/calibrate.py:90-200`)

$$
w^\star = \arg\min_w \Big[\bar z^2 + (v-1)^2 + \tfrac12\,\text{kurt}^2\Big], \qquad s_v = \sqrt{\max(v_{w^\star},\ 10^{-6})} .
$$

$s_v$ is the factor by which the pixel variance must be scaled so the background residual has unit variance; it multiplies $\sigma^2_{\text{pix}}$ in every later prediction.

### 2.3 The peak prior: $\kappa$ and $\pi$

On the calibrated residuals $z = (x - \mu_0)/(s_v\sigma)$ of all seed pixels, a two-component mixture $(1-\pi)\,\mathcal N(0,1) + \pi\,\mathcal N(0,\kappa^2)$ is fitted by 50 EM iterations from $\kappa_0 = 10$, $\pi_0 = 10^{-3}$, with $\pi \in [10^{-6}, \tfrac12]$ and $\kappa^2 \ge 1.001$:

$$
r_i = \frac{\pi\,\kappa^{-1} e^{-z_i^2/2\kappa^2}}{(1-\pi)\,e^{-z_i^2/2} + \pi\,\kappa^{-1} e^{-z_i^2/2\kappa^2}}, \qquad \pi \leftarrow \bar r, \qquad \kappa^2 \leftarrow \frac{\sum_i r_i z_i^2}{\sum_i r_i}.
$$

$\pi$ is the prior probability that a pixel carries signal and $\kappa$ the width inflation of the signal hypothesis; both enter the Bayes factor of §3.3.

### 2.4 Photon-transfer (flux) variance

With `--flux-variance`, the per-pixel variance is modelled as a linear function of the level, which lets the noise follow frame-to-frame changes of the background rather than a frozen per-pixel estimate (`fit_photon_transfer`, `calibrate.py:203-271`). Live pixels are binned into 24 quantile bins of $\mu_0$; bins with $\ge 500$ pixels contribute medians $(\bar\mu_b, \bar\sigma^2_b)$ and an ordinary least-squares line gives

$$
\sigma^2 = v_r + g\,\mu, \qquad v_r = \max(a, 0)\ [\text{ADU}^2], \quad g = \max(b, 10^{-6})\ [\text{ADU/photon}],
$$

falling back to $v_r = 0$, $g = \operatorname{med}_b(\bar\sigma^2_b/\bar\mu_b)$ if the fit is unphysical. $g$ is the *measured* gain (9.9 ADU/photon on the Jungfrau b2AR data whose geometry declares 1) and is what the stream header reports as `probixi/adu_per_photon`.

### 2.5 Eigen-background (optional)

With `eigen_modes` $= r > 0$, the two-sided clipped residuals $R_f = \text{clip}(x_f - \mu_{\text{pix}}, \pm 4\sigma_{\text{pix}})\,m$ over the seed frames are column-centred and the top-$r$ eigenvectors $U_\rho$ of $R R^\top$ are lifted to pixel space and unit-normalised (`fit_eigen_background`). At run time the model mean is corrected per frame by the projection $\mu_0 \leftarrow \mu_0 + \sum_\rho \langle U_\rho, \min(x - \mu_0, 5\sigma_0)\,m\rangle\,U_\rho$. Off by default.

### 2.6 The detection threshold

The detection statistic $T$ of §3.4 is not $\mathcal N(0,1)$ under background even when $z$ is: the maximum over scales biases its body and spatial correlation fattens its tail. Both are measured (`calibrate_threshold`, `calibrate.py:354-555`). Pooling up to $10^6$ background values of $T$ over the seed frames,

$$
\mu_T = \operatorname{med}(T), \qquad \sigma_T = \operatorname{std}\big(\{T < \mu_T\} \cup \{2\mu_T - T : T < \mu_T\}\big),
$$

a body fit that ignores the signal-bearing right tail. A seed frame is *quiet* when its maximum $T$ does not exceed $\mu_T + \sigma_T\sqrt{2\ln N_{\text{valid}}}$, the expected extreme of $N_{\text{valid}}$ null draws (at least four, else the lower half by $\max T$). For each candidate $T^\star$ on the grid $3.0, 3.1, \dots, 8.0$ the median count of local maxima of $T > T^\star$ with at least `size_min` pixels over the quiet frames is computed, and

$$
\tau = \min\{T^\star : \operatorname{med}_{\text{quiet}} \#\text{blobs}(T^\star) \le N_{\text{noise}}\}, \qquad N_{\text{noise}} = 5 \ (\texttt{--target-noise-peaks}),
$$

i.e. the lowest threshold at which a signal-free frame yields at most $N_{\text{noise}}$ spurious blobs. If no grid value achieves this, $\tau$ pins at the grid maximum 8.0, which is the symptom that revealed the shadow problem of §2.1.

### 2.7 Frame level and the blank-shot screen

The *level* of a frame is the median of every 97th pixel, $\text{lvl}(x) = \operatorname{med}(x[0::97])$; the run's reference level is the upper median of the seed levels. A frame with $\text{lvl}(x) < 0.1\,\text{ref}$ (`frame_screen_frac`) is a blank shot: it is still scored but never folded into the running statistics, which would otherwise drift down towards empty frames.

### 2.8 Beam-stop exclusion

After the threshold is set, the calibrated finder runs on up to 200 seed frames and the $|\mathbf q|$ of every peak is histogrammed in 40 bins against the number of live pixels per bin. A peak density in the innermost populated bins exceeding 50 times the median density identifies the beam-stop halo; $q_{\min}$ is set at the outer edge of that spike (extending while the density stays above 10 % of the peak) provided at least 5 % of the peaks fall inside, and pixels with $|\mathbf q| < q_{\min}$ are removed from $m$ (`_infer_beamstop_qmin`, `_beamstop_qmin_from_histogram`).

### 2.9 Per-frame scale

For every frame a relative fluence is estimated by weighted least squares of the frame against the model on a fixed random subset $I$ of $40\,000$ live pixels, weights $1/\sigma_0^2$, after discarding pixels above $\mu_0 + 5\sigma_0$ (`ScaleReference.estimate`, `noise/scale.py`):

$$
x(p) \approx a + s\,\mu_0(p), \quad p \in I, \qquad \sigma_s = \sqrt{S_w / D},
$$

with $S_w = \sum_I 1/\sigma_0^2$ and $D$ the WLS determinant. $s$ and $\sigma_s$ are written as `probixi/scale` on every crystal, an initialisation for downstream per-pattern scaling.

## 3. Detecting peaks in one frame

### 3.1 Local background and the effective model

The blended mean is corrected per frame by a local annulus, so slow structure the running model does not track (jet scatter, lipid rings) does not read as excess (`PeakFinder._effective_background`, `peaks/peakfinder.py:397-434`; `local_mean_var`, `peaks/neighborhood.py:109-142`). With the square annulus $A_p = \{q : r_{\text{in}} < \|q - p\|_\infty \le r_{\text{out}}\}$, $r_{\text{in}} = 4$, $r_{\text{out}} = 9$ px (fixed; $280$ nominal pixels), the one-sided clipped residual $\tilde\rho = \min(x - \mu_0,\ 5\sigma_0)$ and $C_p = \max(\sum_{A_p} m, 1)$:

$$
\ell_\mu(p) = \frac{1}{C_p}\sum_{q\in A_p} m\,\tilde\rho, \qquad \ell_{\sigma^2}(p) = \max\!\Big(\frac{1}{C_p}\sum_{q\in A_p} m\,\tilde\rho^2 - \ell_\mu^2,\ 0\Big),
$$

$$
\mu_{\text{eff}} = \mu_0 + \ell_\mu, \qquad
\sigma^2_{\text{eff}} = \begin{cases}
\max(\sigma_0^2,\ \ell_{\sigma^2}) & \text{frozen variance}\\[2pt]
\max\!\big(v_r + g\,\max(\mu_{\text{eff}}, 0),\ \phi\,\sigma_0^2,\ \ell_{\sigma^2}\big) & \text{flux variance}, \ \phi = 0.15
\end{cases}
$$

($\phi$ = `flux_var_floor`, fixed). The clip at $5\sigma_0$ (`LOCAL_BG_CLIP_K`) keeps Bragg pixels inside the annulus from lifting the local mean.

### 3.2 Excess and whitening

$$
E(p) = x(p) - \mu_{\text{eff}}(p)\ [\text{ADU}], \qquad z(p) = \frac{E(p)}{\sigma_{\text{eff}}(p)} .
$$

Under the model, background pixels have $z \sim \mathcal N(0,1)$; every later statistic is built on $z$ and on $E$.

### 3.3 Bayes factor and posterior

Per pixel, $H_0: x \sim \mathcal N(\mu_{\text{eff}}, \sigma^2_{\text{eff}})$ against $H_1: x \sim \mathcal N(\mu_{\text{eff}}, \kappa^2\sigma^2_{\text{eff}})$ with the learned $\kappa$ (§2.3), gated to positive excursions:

$$
\log\text{BF}(p) = \big[m(p) \wedge E(p) > 0\big]\Big(-\log\kappa + \tfrac12 z(p)^2\big(1 - \kappa^{-2}\big)\Big), \qquad
L(p) = \log\text{BF}(p) + \log\frac{\pi}{1-\pi} .
$$

The logits are smoothed with a mask-weighted $5\times5$ Gaussian ($\sigma = 1$ px) and squashed to a posterior $P(p) = m(p)\,\sigma(\tilde L(p))$, kept in the stream as a per-peak `posterior_mean`; the pipeline itself detects on the matched filter below.

### 3.4 Matched filter

For a bank of unit-energy Gaussian kernels $u_k$ with $\sigma_k \in \{1.0, 1.6, 2.4\}$ px (`mf_scales`, fixed; sizes $2\lceil 3\sigma_k\rceil + 1$) the masked correlation, renormalised by the energy of the kernel that fell on live pixels, is exactly $\mathcal N(0,1)$ under the null even at edges (`matched_filter_z`, `neighborhood.py:146-165`):

$$
\text{MF}_k(p) = \frac{\sum_q u_k(q-p)\,m(q)\,z(q)}{\sqrt{\sum_q u_k(q-p)^2\,m(q)}}, \qquad T(p) = \max_k \text{MF}_k(p) .
$$

A pixel is a detection when $T(p) > \tau$ with the learned $\tau$ of §2.6. Detections are grouped into 8-connected blobs (`label_connected_components`); a frame with more than 1000 blobs is treated as empty.

### 3.5 Blob statistics and filters

For a blob $B$ with weights $w_p = \max(E(p), 0)$ (`compute_blob_stats`, `peaks/blobs.py:178-350`):

| quantity | definition |
|---|---|
| size $n_B$ | $|B|$ (px) |
| centroid $(\bar i, \bar j)$ | $\sum_B w_p\,p / \sum_B w_p$ (px) |
| intensity $I_B$ | $\sum_B E(p)$ (ADU, signed) |
| $\sigma(I_B)$ | $\sqrt{\sum_B \sigma^2_{\text{eff}}(p)}$ |
| background sum | $\sum_B \mu_{\text{eff}}(p)$ (ADU; the noise-model background under the blob) |
| response $R_B$ | $\max_B T(p)$ |
| eccentricity | $\lambda_{\max}/\lambda_{\min}$ of the $w$-weighted second-moment matrix ($\infty$ for collinear blobs) |
| peakedness | $\max_B E / (I_B/n_B)$ |

A blob is kept when $n_B \ge 2$, eccentricity $\le 5$, peakedness $\ge 1.2$ (fixed) and $n_B \le \text{cap}_B$ with the response-aware footprint cap

$$
\text{cap}_B = \max\!\Big(\texttt{size\_max},\ 4 \cdot 2\pi\,\sigma_{\max}^2\,\ln\max(R_B/\tau, 1)\Big), \qquad \sigma_{\max} = \max_k\sigma_k,
$$

four times the area a Gaussian of width $\sigma_{\max}$ and peak response $R_B$ has above $\tau$, so that a bright spot is allowed the footprint its own response implies while `size_max` $= 30$ binds only dim blobs. The kept centroids, $I_B$, $\sigma(I_B)$, background sums and pixel counts are the *peaks* handed to indexing and written to the `Peaks from peak search` table.
## 4. From pixels to reciprocal space

### 4.1 The forward model

A detector pixel $(i, j)$ maps to laboratory coordinates through its panel's origin $(c_x, c_y)$ and basis vectors $\mathbf f = (f_x, f_y)$, $\mathbf s = (s_x, s_y)$ (in pixel units, from the geometry file; `_lab_xy_pixels`, `indexer/forward.py:82-110`):

$$
x_{\text{px}} = c_x + (j - j_{\min})\,f_x + (i - i_{\min})\,s_x, \qquad
y_{\text{px}} = c_y + (j - j_{\min})\,f_y + (i - i_{\min})\,s_y,
$$

and, with pixel size $p$ (Å), camera length $L$ (Å) and the incident direction $\mathbf s_0 = (0,0,1)$ (`detector_to_q`, `forward.py:113-144`):

$$
\mathbf r = (x_{\text{px}}\,p,\ y_{\text{px}}\,p,\ L), \qquad
\mathbf q = \frac{1}{\lambda}\Big(\frac{\mathbf r}{|\mathbf r|} - \mathbf s_0\Big) \quad [\text{Å}^{-1}].
$$

Every observed $\mathbf q$ lies on the Ewald sphere of radius $1/\lambda$ centred at $-\mathbf s_0/\lambda$; $|\mathbf q| = 1/d$ (crystallographic convention, no $2\pi$). The inverse map (`q_to_detector`) intersects the ray $\lambda\mathbf q + \hat{\mathbf z}$ with the detector plane and solves the $2\times2$ panel system for $(i, j)$; a predicted reflection that lands on no panel is dropped.

### 4.2 Cells and orientation

A cell $(a, b, c, \alpha, \beta, \gamma)$ defines the direct matrix $\mathbf M = [\mathbf a\ \mathbf b\ \mathbf c]$ with $\mathbf a \parallel x$ and $\mathbf b$ in the $xy$ plane, and the reciprocal basis $\mathbf B = \mathbf M^{-\top}$, whose columns are $\mathbf a^*, \mathbf b^*, \mathbf c^*$ (`cell_to_B`, `indexer/lattice.py:14-37`). A crystal's orientation matrix is

$$
\mathbf A = \mathbf U\,\mathbf B, \qquad \mathbf q = \mathbf A\,\mathbf h,
$$

with $\mathbf U$ a rotation and $\mathbf h = (h, k, l)$ integer. Given $\mathbf A$, the cell is recovered from $\mathbf M = \mathbf A^{-\top}$ (`B_to_cell`) and $\mathbf U = \mathbf A\,\mathbf B^{-1}$ with $\mathbf B$ re-canonicalised.

### 4.3 The reciprocal tolerance

One tolerance governs seeding, refinement and hkl assignment (`Indexer.__init__`, `indexer/indexer.py:650-656`):

$$
q_{\text{tol}} = f_q\,\min\big(|\mathbf a^*|, |\mathbf b^*|, |\mathbf c^*|\big), \qquad f_q = 0.25 \ (\texttt{q\_tolerance\_fraction}, \text{fixed}),
$$

a quarter of the shortest reciprocal axis of the target cell, so it scales with the cell rather than the detector. A peak with $\mathbf h = \operatorname{round}(\mathbf A^{-1}\mathbf q)$ is *indexed* by $\mathbf A$ when $|\mathbf A\mathbf h - \mathbf q| < q_{\text{tol}}$.

## 5. Indexing

### 5.1 Seeding with a known cell

The cell is known; only $\mathbf U$ is sought (`sphere_seed_candidates`, `indexer/seed.py:91-148`). Peaks are capped to the 80 brightest (`max_seed_peaks`) for seeding. With $\hat{\mathbf a}$ the direct $a$-axis of the target cell and $L_a = |\mathbf a|$:

1. **Directions.** $2n_d$ directions $\pm\hat{\mathbf d}_i$ from a Fibonacci hemisphere, $n_d = 6000$ (`n_directions`).
2. **Integer-projection fitness.** Placing $\mathbf a$ along $\hat{\mathbf d}$, every peak's projection $L_a\,\hat{\mathbf d}\cdot\mathbf q_i$ must be near an integer if the direction is right:
   $$F(\hat{\mathbf d}) = \frac{\sum_i w_i \cos\!\big(2\pi L_a\,\hat{\mathbf d}\cdot\mathbf q_i\big)}{\sum_i w_i},$$
   with $w_i$ the peaks' posterior means (§3.3). The 64 best directions are kept (`2 * top_directions`).
3. **Roll.** For each kept direction, the shortest-arc rotation $\mathbf R_0: \hat{\mathbf a} \to \hat{\mathbf d}$ is combined with $n_s = 120$ rolls about $\hat{\mathbf d}$ (`n_spin`), $\mathbf U = \mathbf R(\theta_s\hat{\mathbf d})\,\mathbf R_0$, giving $64 \times 120$ candidate $\mathbf A = \mathbf U\mathbf B$.
4. **Score.** $S(\mathbf A) = \sum_i w_i\,\mathbb 1\big[|\mathbf A\,\operatorname{round}(\mathbf A^{-1}\mathbf q_i) - \mathbf q_i| < q_{\text{tol}}\big]$; the top 64 (`max_candidates`) go to refinement.

Frames with fewer than 5 peaks are not attempted; below 30 peaks the direction and roll grids are doubled (`adaptive_sparse`).

### 5.2 Orientation refinement

For every candidate an axis-angle perturbation $\boldsymbol\omega \in \mathbb R^3$ is optimised with the cell fixed, $\mathbf A(\boldsymbol\omega) = \mathbf R(\boldsymbol\omega)\,\mathbf A^{\text{seed}}$ with $\mathbf R$ the Rodrigues rotation (`refine_multiframe_known_B`, `indexer/refine.py:268-427`, Triton twin in `kernels/refine.py`). With inlier indicators $m_i$ re-assigned every 10 steps and residuals $\mathbf r_i = \mathbf A\mathbf h_i - \mathbf q_i$:

$$
\mathcal L(\boldsymbol\omega) = \frac{\sum_i w_i m_i\,|\mathbf r_i|^2}{\max(\sum_i w_i m_i, 1)},
$$

minimised by 200 Adam steps at learning rate $10^{-3}$ (fixed). Each candidate ends with $n_{\text{idx}} = \sum_i m_i$ and $\text{rmsd} = \sqrt{\langle |\mathbf r_i|^2\rangle_{m_i = 1}}$ (Å$^{-1}$, full 3-D residual).

### 5.3 Ranking and cell check

Candidates are ranked by $\text{score}_k = \sum_i w_i m_{ki} - \text{rmsd}_k / (10^3 \max_k \text{rmsd}_k)$ (inlier weight first, rmsd as a tie-breaker) and taken in order; a candidate needs $n_{\text{idx}} \ge 6$ (`min_indexed`) and, after the cell refinement below, a cell within the target's tolerance (`_cell_matches_target`): every sorted edge within 5 % and every sorted angle within $3^\circ$ (`CellMatchConfig`, fixed). The first candidate that passes is the frame's lattice.

### 5.4 Cell refinement

Holding every crystal at the target cell puts its predictions systematically off (0.5 % of the radius on b2AR), so orientation and the lattice-type-allowed cell parameters are refined jointly (`refine_cell`, `indexer/refine.py:117-233`). With $\mathbf p_0$ the candidate's own cell parameters and $\mathbf T \in \{0,1\}^{6\times m_c}$ the map from the $m_c$ free parameters $\boldsymbol\rho$ to the six cell values (ties and fixed angles per lattice type: monoclinic frees $a, b, c$ and the unique angle; hexagonal ties $a = b$ and frees $c$; cubic ties all three edges, etc.),

$$
\mathbf A(\boldsymbol\theta) = \mathbf R(\boldsymbol\omega)\,\mathbf U_0\,\mathbf B(\mathbf p_0 + \mathbf T\boldsymbol\rho), \qquad \boldsymbol\theta = (\boldsymbol\omega, \boldsymbol\rho).
$$

A detector position constrains a reflection's $\mathbf q$ only perpendicular to its scattered ray $\hat{\mathbf k}_i = (\mathbf q_i + \mathbf s_0/\lambda)/|\cdot|$; along the ray the residual is the Ewald excitation error, which the lift onto the sphere sets to zero for observed peaks. The residual is therefore split and whitened per component using scales from the starting fit:

$$
\mathbf d_i = \mathbf A\mathbf h_i - \mathbf q_i, \quad d_{\parallel,i} = \mathbf d_i\cdot\hat{\mathbf k}_i, \quad \mathbf d_{\perp,i} = \mathbf d_i - d_{\parallel,i}\hat{\mathbf k}_i,
\qquad
s_\parallel = \sqrt{\langle d_\parallel^2\rangle}, \quad s_\perp = \max\!\Big(\sqrt{\tfrac12\langle|\mathbf d_\perp|^2\rangle},\ s_\parallel/100\Big),
$$

$$
\mathbf f(\boldsymbol\theta) = \Big[\ \frac{\mathbf d_{\perp,i}}{s_\perp},\ \frac{d_{\parallel,i}}{s_\parallel}\ \Big]_{i \in \text{inliers}} \oplus \frac{\mathbf T\boldsymbol\rho}{\boldsymbol\sigma_{\text{prior}}}, \qquad
\boldsymbol\sigma_{\text{prior}} = \tfrac13\,(0.05\,a_0,\ 0.05\,b_0,\ 0.05\,c_0,\ 3^\circ, 3^\circ, 3^\circ).
$$

The prior is a unit-variance Tikhonov term of width one third of the cell-match tolerance, centred on the starting cell; it is what stops a cell axis that lies along the beam (unconstrained by the data) from drifting. $\mathcal L = |\mathbf f|^2$ is minimised by Levenberg–Marquardt, $\delta\boldsymbol\theta = -(\mathbf H + \lambda_{\text{LM}}\operatorname{diag}\mathbf H)^{-1}\mathbf J^\top\mathbf f$ with $\mathbf J$ from autograd, 8 iterations per round, two rounds with hkl re-assignment in between, solved on the host in float64. The result replaces the orientation-only solution when the whitened data residual has not increased, $n_{\text{idx}} \ge 6$ and the refined cell still matches the target.

### 5.5 Several lattices per frame

With `max_lattices` $> 1$ the peaks explained by an accepted lattice, $|\mathbf A\mathbf h_i - \mathbf q_i| < 0.006\ \text{Å}^{-1}$ (`peel_radius`), are removed and the residue re-seeded while at least 6 remain; a new $\mathbf A$ within 2 % (Frobenius, up to sign) of an existing one is a duplicate and is skipped.

## 6. Predicting the full reflection set

For an accepted $\mathbf A$ every $\mathbf h$ with $|\mathbf A\mathbf h| \le q_{\max}$ (the largest $|\mathbf q|$ at the detector corners) and the cell's centering condition is enumerated (`predict_reflections`, `indexer/predict.py:64-140`). Its excitation error is the distance of $\mathbf q$ from the Ewald sphere,

$$
\varepsilon(\mathbf h) = |\lambda\,\mathbf A\mathbf h + \hat{\mathbf z}| - 1 ,
$$

and a reflection is predicted to diffract when $|\varepsilon| < \tau(q)$ with a rocking half-width that grows with resolution (`rocking_radius`):

$$
R(q) = r_{\text{size}} + \tfrac12\,\eta\,q + \tfrac12\,\lambda\,\beta_{bw}\,q^2, \qquad
\tau(q) = \max\!\big(\lambda\cdot 1.5\cdot R(q),\ 5\times10^{-4}\big),
$$

where $r_{\text{size}} = 5\times10^{-5}\ \text{Å}^{-1}$ is the domain-size term (`domain_size_recip`, fixed), $\beta_{bw} = \Delta\lambda/\lambda$ the bandwidth (default 0), the factor 1.5 is `predict_sigma` and $5\times10^{-4}$ the `partiality_threshold` floor. The mosaicity $\eta$ is **learned per crystal** from its indexed peaks (`estimate_mosaicity`): with $r_{s,i} = |\varepsilon_i|/\lambda - \tfrac12\lambda\beta_{bw}q_i^2$ the through-origin fit $r_s = \tfrac12\eta\,q$ gives $\eta = 2\,\sqrt{\pi/2}\,\sum_i q_i r_{s,i} / \sum_i q_i^2$ (half-normal mean to $\sigma$), clamped to $[0.02^\circ, 0.8^\circ]$ and falling back to the prior $0.1^\circ$ below 8 peaks. Predicted positions come from `q_to_detector`; those off the detector are dropped. The rocking radius at the crystal's resolution limit is what the stream reports as `profile_radius` (§8).
## 7. Integration

### 7.1 Learning the aperture

Integration radii are not configured; they are read off the data once, after calibration (`_learn_integration_radii`, `probixi.py`). Peaks from a seeded random sample of frames (until at least 200 peaks on more than 32 frames, at most 512 frames) are stacked into a radial profile of the excess (`radial_profile`, `indexer/integrate.py:52-88`): for ring $b = \operatorname{round}\sqrt{\delta r^2 + \delta c^2} \le 16$ around each peak centroid, $\bar E_i(b)$ is the mean excess on that ring and

$$
P(b) = \operatorname{med}_i\,\bar E_i(b), \qquad p(b) = P(b) - \min_b P(b), \qquad b = 0, \dots, 16 .
$$

**Flux rule** (`radii_from_profile`, `integrate.py:110-156`). A Gaussian is fitted to the core, $\ln p(b) = \text{const} + s\,b^2$ over the rings before $p$ first drops below $10\,\%$ of $p(0)$, giving $\sigma_{\text{spot}} = \sqrt{-1/2s}$ and the radius at which the spot falls to the `floor` of 2 %:

$$
r_{\text{sig}} = \sigma_{\text{spot}}\sqrt{-2\ln 0.02} \approx 2.8\,\sigma_{\text{spot}}, \qquad
r_{\text{in}} = r_{\text{sig}} + 1, \qquad
r_{\text{out}} = \sqrt{r_{\text{in}}^2 + 120/\pi},
$$

so the background annulus always holds about 120 pixels (`background_pixels`, fixed) and starts one pixel outside the spot.

**SNR rule** (default, `--aperture snr`; `snr_disk_radius`, `integrate.py:98-107`). With $n(b)$ the number of integer offsets on ring $b$, the disk radius is the one maximising the background-limited signal-to-noise of the *profile*:

$$
S(k) = \sum_{b\le k} p(b)\,n(b), \qquad N(k) = \sum_{b\le k} n(b), \qquad
k^\star = \arg\max_{1\le k\le r_{\text{sig}}}\frac{S(k)}{\sqrt{N(k)}}, \qquad r_{\text{sig}} \leftarrow \min(r_{\text{sig}},\ k^\star + 0.5),
$$

with $r_{\text{in}}, r_{\text{out}}$ left at the flux-rule values so the annulus still clears the 2 % radius. On 1 px spots this gives a $3\times3$ disk ($r_{\text{sig}} = 1.5$) where the flux rule gave 5 px; on broad spots both rules agree. If too few peaks are found the indexer falls back to $(3.0, 4.0, 7.36)$ px (`FALLBACK_RADII`).

### 7.2 Ring integration of a predicted reflection

For each predicted position (`integrate_rings`, `integrate.py:159-250`), first snapped onto an observed peak centroid within 5 px (`snap_radius`) when one exists, the raw frame is $R = E + \mu_{\text{eff}}$ and pixel offsets $(\delta r, \delta c)$ about the rounded centre have $\rho^2 = \delta r^2 + \delta c^2$:

- **annulus** $\mathcal A_i = \{r_{\text{in}}^2 \le \rho^2 \le r_{\text{out}}^2\}$ on live in-frame pixels, $n_{b,i} = |\mathcal A_i|$,
  $$B_i = \frac{1}{n_{b,i}}\sum_{\mathcal A_i} R_p \ [\text{ADU/px}], \qquad s^2_{b,i} = \frac{1}{n_{b,i}-1}\sum_{\mathcal A_i}(R_p - B_i)^2 ;$$
- **disk** $\mathcal D_i = \{\rho^2 \le r_{\text{sig}}^2\}$, *deblended*: a pixel claimed by several disks belongs to the reflection whose (unrounded) centre is nearest, ties to the lower index; the owned set is $\mathcal U_i$ with $n_i = |\mathcal U_i|$ pixels.

$$
I_i = \sum_{p\in\mathcal U_i}(R_p - B_i)\ [\text{ADU}], \qquad
\text{peak}_i = \max_{p\in\mathcal U_i}(R_p - B_i) .
$$

The uncertainty combines the annulus scatter (propagated through the subtraction, hence the $1 + n_i/n_{b,i}$ factor) with the signal's own shot noise in ADU$^2$, $I/g$ photons times $g^2$:

$$
\sigma_i = \sqrt{\,n_i\,s^2_{b,i}\Big(1 + \frac{n_i}{n_{b,i}}\Big) + g\,\max(I_i, 0)\,} ,
$$

and when the annulus has fewer than 10 live pixels (`MIN_ANNULUS_PIXELS`) the annulus term is replaced by the noise model's, $\sum_{\mathcal U_i} V_p\,(1 + n_i/n_{\text{bg}})$, with $V = \sigma^2_{\text{eff}}$ and $n_{\text{bg}} = 280$ the local-background annulus of §3.1. Reflections with $\sigma_i \le 0$ are dropped; then, across all lattices of the frame, any two reflections on the same panel closer than $1.5\,r_{\text{sig}}$ are both removed (`keep_non_overlapping`).

### 7.3 The model background under the disk

Alongside the annulus estimate, each reflection carries what the noise model itself expects under its pixels and how uncertain that is:

$$
\text{bg\_model}_i = \sum_{p\in\mathcal U_i}\mu_{\text{eff}}(p)\ [\text{ADU}], \qquad
\text{bg\_model\_var}_i = \frac{n_i}{n_{\text{bg}}}\sum_{p\in\mathcal U_i} V_p\ [\text{ADU}^2].
$$

The variance follows from §3.1: the only frame-dependent part of $\mu_{\text{eff}}$ under a disk is the single local-annulus mean $\ell_\mu$ shared by all $n_i$ pixels, whose variance is $\bar V/n_{\text{bg}}$, so the summed background has variance $n_i^2\,\bar V/n_{\text{bg}}$. Compared with the ring estimate's $c\,B_i n_i$ with $c = n_i/n_{b,i}$, this is smaller by roughly $n_{b,i}/n_{\text{bg}}$ and carries no Poisson-count bias; it is written as the extra columns `n_pix bg_model bg_model_var` for a merging program that models the background explicitly (torchsx `--background model`).

### 7.4 Enrichment: is this lattice real?

A lattice that explains a handful of noise blobs must not survive. Before integrating, the predicted positions are tested against the image (`spot_enrichment`, `integrate.py:310-364`). With $z = E/\sqrt{V}$, a pixel is *bright* when the $5\times5$ maximum of $z$ around it exceeds the detection threshold $\tau$; the background bright rate is $p_0 = \#\{\text{bright live pixels}\}/\#\{\text{live pixels}\}$. Of the $n_{\text{keep}}$ predicted centres on live pixels, $n_{\text{bright}}$ land on bright pixels, and

$$
\text{enrichment} = \frac{n_{\text{bright}}/n_{\text{keep}}}{p_0}, \qquad
\texttt{enrich\_p} = \Pr\big[X \ge n_{\text{bright}}\big],\ X \sim \text{Poisson}(n_{\text{keep}}\,p_0) .
$$

An enrichment near 1 is a lattice indistinguishable from chance; `--enrich-gate` drops crystals with $\texttt{enrich\_p} > \alpha$, $\alpha = 10^{-3}$ (`--enrich-alpha`). Both numbers are written per crystal.

### 7.5 Resolution limit and profile radius

The crystal's diffraction limit is where its shell-averaged $I/\sigma$ crosses 1 (`falloff_resolution_limit`): predicted reflections are sorted by $|\mathbf q|$ into 10 equal-count shells and the crossing is interpolated linearly between the two shells bracketing $\langle I/\sigma\rangle = 1$ (`resolution_isigma`); with fewer than 40 reflections the 0.90 quantile of the observed peaks' $|\mathbf q|$ is used instead. It is reported (`diffraction_resolution_limit`, nm$^{-1}$) but never used to truncate the reflection list. The `profile_radius` is the rocking half-width of §6 evaluated at that limit, $R(q_{\text{lim}})$, converted to nm$^{-1}$ and clamped to $[5\times10^{-4}, 2\times10^{-2}]$.

## 8. What is written

Per frame: the peak list (fs, ss, $10|\mathbf q|$ in nm$^{-1}$, $I_B$), `hit = [n_{\text{peaks}} \ge 5]`. Per crystal: the cell recovered from $\mathbf A$ (nm, degrees), $\mathbf a^*, \mathbf b^*, \mathbf c^*$ as the columns of $10\,\mathbf A$ (nm$^{-1}$), `profile_radius`, `probixi/rmsd` (Å$^{-1}$), `probixi/mosaicity` (degrees), `probixi/scale` $= s \pm \sigma_s$ (§2.9), `probixi/enrichment`, `probixi/enrich_p`, the diffraction limit, and the reflection table:

| column | quantity | units |
|---|---|---|
| `h k l` | predicted Miller indices | |
| `I` | $I_i$ | ADU |
| `sigma(I)` | $\sigma_i$ | ADU |
| `peak` | $\max_{\mathcal U_i}(R_p - B_i)$ | ADU |
| `background` | $B_i$ | ADU/px |
| `fs ss panel` | snapped position and panel | px |
| `n_pix` | $n_i$ | px |
| `bg_model` | $\sum_{\mathcal U_i}\mu_{\text{eff}}$ | ADU |
| `bg_model_var` | $\tfrac{n_i}{n_{\text{bg}}}\sum_{\mathcal U_i}V_p$ | ADU$^2$ |

The header records the recipe the run actually used: `probixi/int_radius` $= (r_{\text{sig}}, r_{\text{in}}, r_{\text{out}})$, `probixi/adu_per_photon` $= g$ (the measured gain when the flux model is on, else the geometry's), `probixi/bg_annulus_pixels` $= n_{\text{bg}}$, `probixi/aperture` and `probixi/reflection_columns`. The three trailing columns are ignored by CrystFEL's readers. The DuckDB output holds the same quantities in the `integration`, `frames`, `crystals`, `reflections` and `peaks` tables, the last with `photons` $= (I_B + \text{bg})/g$ and `background_photons` $= \text{bg}/g$ for every searched peak.

## 9. Handing over to merging

A merging program that models detection as $n \sim \text{Poisson}(S + b)$ needs the raw disk count and an estimate of $b$ with its uncertainty. From the columns above, in photons,

$$
n_i = \frac{I_i + B_i\,n_i^{\text{px}}}{g}, \qquad
b_i^{\text{ring}} = \frac{B_i\,n_i^{\text{px}}}{g}\ \text{with}\ \operatorname{Var} = \frac{n_i^{\text{px}}}{n_{b,i}}\,b_i^{\text{ring}}, \qquad
b_i^{\text{model}} = \frac{\text{bg\_model}_i}{g}\ \text{with}\ \operatorname{Var} = \frac{\text{bg\_model\_var}_i}{g^2} .
$$

The ring pair is the CrystFEL convention (torchsx `--background annulus`); the model pair replaces the annulus count by the noise model's expectation, whose variance is measured rather than assumed Poisson (`--background model`). On b2AR the model background's standard deviation is about a third of the ring estimate's.

## 10. Notation

| symbol | meaning | units | where |
|---|---|---|---|
| $x_f, x(p)$ | raw frame, pixel value | ADU | §0 |
| $m(p)$ | live-pixel mask (geometry, dead, shadow, beam stop) | | §1.3, §2.1, §2.8 |
| $\mu_k, \sigma^2_k$ | running mean and variance of view $k \in \{\text{pix}, \text{rot}, \text{pan}\}$ | ADU, ADU$^2$ | §1.1–1.2 |
| $w_k$, $s_v$ | learned blend weights and variance scale | | §2.2 |
| $\mu_0, \sigma_0^2$ | blended model prediction | ADU, ADU$^2$ | §1.4 |
| $\kappa, \pi$ | peak-hypothesis width inflation and prior | | §2.3 |
| $v_r, g$ | read variance and gain of the photon-transfer law | ADU$^2$, ADU/photon | §2.4 |
| $r_{\text{in}}, r_{\text{out}}, n_{\text{bg}}$ | local-background annulus radii (4, 9 px) and pixel count (280) | px | §3.1 |
| $\ell_\mu, \ell_{\sigma^2}$ | local annulus mean and variance of the residual | ADU, ADU$^2$ | §3.1 |
| $\mu_{\text{eff}}, \sigma^2_{\text{eff}} (= V)$ | effective per-frame background model | ADU, ADU$^2$ | §3.1 |
| $E, z$ | excess and whitened excess | ADU, | §3.2 |
| $\text{MF}_k, T, \tau$ | matched-filter response at scale $k$, its maximum, learned threshold | $\sigma$ | §3.4, §2.6 |
| $\mu_T, \sigma_T$ | null body of $T$ | | §2.6 |
| $I_B, R_B, n_B$ | blob excess sum, response, size | ADU, $\sigma$, px | §3.5 |
| $s, \sigma_s$ | per-frame scale and its uncertainty | | §2.9 |
| $\mathbf q, \lambda, L, p$ | scattering vector, wavelength, camera length, pixel size | Å$^{-1}$, Å, Å, Å | §4.1 |
| $\mathbf A = \mathbf U\mathbf B$, $\mathbf h$ | orientation matrix, reciprocal basis, Miller indices | Å$^{-1}$ | §4.2 |
| $q_{\text{tol}}$ | reciprocal tolerance, $0.25\min|\mathbf a^*, \mathbf b^*, \mathbf c^*|$ | Å$^{-1}$ | §4.3 |
| $m_i, n_{\text{idx}}, \text{rmsd}$ | inlier indicator, count, residual | Å$^{-1}$ | §5.2 |
| $\hat{\mathbf k}_i, d_\parallel, \mathbf d_\perp, s_\parallel, s_\perp$ | scattered ray, residual components, whitening scales | Å$^{-1}$ | §5.4 |
| $\boldsymbol\sigma_{\text{prior}}$ | cell prior width, tolerance/3 | Å, rad | §5.4 |
| $\varepsilon, R(q), \eta, r_{\text{size}}, \beta_{bw}$ | excitation error, rocking half-width, mosaicity, domain term, bandwidth | , Å$^{-1}$, rad, Å$^{-1}$, | §6 |
| $r_{\text{sig}}, r_{\text{in}}, r_{\text{out}}$ (integration) | learned disk and annulus radii | px | §7.1 |
| $B_i, s^2_{b,i}, n_{b,i}$ | annulus mean, variance, pixel count | ADU/px, ADU$^2$, px | §7.2 |
| $\mathcal U_i, n_i, I_i, \sigma_i$ | owned disk pixels, count, intensity, uncertainty | px, ADU | §7.2 |
| $\text{bg\_model}_i, \text{bg\_model\_var}_i$ | model background under the disk and its variance | ADU, ADU$^2$ | §7.3 |
| $p_0, n_{\text{bright}}, \texttt{enrich\_p}$ | bright rate, bright predictions, chance probability | | §7.4 |
