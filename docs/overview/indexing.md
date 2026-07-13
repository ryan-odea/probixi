# Indexing

[Peakfinding](peakfinding.md) hands over a list of spots on the detector. Indexing is the job of explaining them, finding the crystal orientation and unit cell so that every spot sits on a reciprocal-lattice point.

The target is a $3\times3$ matrix $A$ that turns integer Miller indices into the scattering vectors we observe,

$$q = A\,(h,k,l)^\top,\qquad A = U\,B,$$

where $B$ encodes the known unit cell and $U$ is the unknown orientation. Because $B$ is fixed by your `.cell` file, indexing is really a search over rotations $U$, three degrees of freedom.

## Into Reciprocal Space

First `probixi` needs the observed $q$ for each peak. It builds them straight from the geometry with the Ewald construction, beam along $+z$. A pixel at lab position $(x,y,z)$ (with $z$ the sample-to-detector distance `clen`) scatters along the unit vector $\hat{s}$, and

$$q = \frac{\hat{s} - \hat{s}_0}{\lambda},\qquad \hat{s} = \frac{(x,y,z)}{\lVert(x,y,z)\rVert},\quad \hat{s}_0 = (0,0,1).$$

`probixi` uses the crystallographic convention with no $2\pi$, so $a\cdot q = h$ is an integer and $q$ comes out in Å$^{-1}$ (converted to nm$^{-1}$, $\times 10$, only at output time).

The cell matrix itself is the inverse-transpose of the direct lattice, $B = M^{-\top}$, where $M$'s columns are the real-space vectors $a, b, c$ built from your cell edges and angles in the standard orientation ($a$ along $x$, $b$ in the $xy$-plane). Going the other way, `B_to_cell` recovers edges and angles from any $B$, which is how the cell is read back out of a solution.

## Seeding

Searching all of rotation space blindly is hopeless, so `probixi` uses a TORO-style trick that factors the 3 DOF into a direction (2 DOF) plus a roll (1 DOF), in three steps.

**Sample directions.** Lay $n_{\text{directions}}$ points (default 6000) on a Fibonacci hemisphere, a spiral that tiles the sphere near-uniformly with no clustering at the poles.

$$z_i = \frac{i + \tfrac12}{n},\quad r_i = \sqrt{1 - z_i^2},\quad \varphi_i = i\,\pi(3 - \sqrt5),\quad \text{dir}_i = (r_i\cos\varphi_i,\ r_i\sin\varphi_i,\ z_i).$$

Only a hemisphere, because a lattice direction and its antipode are equivalent.

**Score each direction** by how well the observed $q$ project onto integer multiples of that trial $a$-axis. With $t = L_a\,\text{dir}$ (the trial real-space $a$-vector, $L_a = \lVert a\rVert$),

$$\text{fitness}(\text{dir}) = \frac{1}{N}\sum_i \cos\!\big(2\pi\, t\cdot q_i\big).$$

If $t$ is a true lattice vector then $t\cdot q_i$ is an integer for every real reflection, so every cosine hits $+1$. Directions that align with a real crystal axis light up, and the rest average toward zero.

**Spin and confirm.** Take the best `top_directions` (32), roll each through `n_spin` (120) in-plane angles to pin down the last DOF, and form the full $A = U B$ for each. Score it by counting genuine inliers. Assign $\text{hkl} = \operatorname{round}(A^{-1} q)$, and count a peak matched when

$$\lVert A\,\text{hkl} - q\rVert^2 < q_{\text{tol}}^2.$$

That match tolerance is derived from your cell, a quarter of the smallest reciprocal spacing, $q_{\text{tol}} = 0.25 \times \min$-spacing, so it self-scales from small-molecule to large-unit-cell data. On peak-starved frames (fewer than ~30 seeding peaks) the search widens automatically (at the cost of some extra computation), with more directions and more spins, because a sparse pattern needs a denser net.

## Refinement

The best seeds are close but discrete. `probixi` then refines every surviving candidate at once with Adam, optimizing a small axis-angle perturbation $\omega$ applied as $A_{\text{eff}} = R(\omega)\,A$. The loss is the mean squared $q$-residual over the currently-assigned inliers,

$$\mathcal{L} = \frac{\sum_i \lVert A_{\text{eff}}\,\text{hkl}_i - q_i\rVert^2\, w_i}{\sum_i w_i} + \text{(penalty if fewer than }6\text{ inliers)},$$

weighted by each peak's detection confidence $w_i$ carried over from peakfinding. Every `reassign_every` (10) steps the peak-to-hkl assignment is recomputed, and as the orientation tightens more peaks snap onto lattice points and join the fit. The degeneracy penalty quietly pushes under-constrained candidates out of contention.

## Accepting a solution

A refined orientation is accepted only if its recovered cell matches the target. `probixi` compares the *sorted* edges and angles, so the check is invariant to how axes happen to be labelled.

$$\frac{|a_i - a_i^{\text{tgt}}|}{a_i^{\text{tgt}}} \le 0.05\quad\text{and}\quad |\theta_i - \theta_i^{\text{tgt}}| \le 3^\circ.$$

Candidates are ranked by inlier evidence, with RMSD as a tie-break, and the first one that clears both the minimum-inlier bar and the cell match wins. If none do, the frame goes unindexed, which beats committing a wrong lattice.

## Predicting and integrating

An accepted orientation lets `probixi` predict **every** reflection that should diffract, whether or not the peak finder flagged it, and integrate them all on a common footing. That is what makes the output merge-ready.

A reflection diffracts when its scattered vector lands on the Ewald sphere. With $S = \lambda q + \hat{z}$, the excitation error is $\varepsilon = \lVert S\rVert - 1$, and a perfect crystal would want $\varepsilon = 0$. Real crystals have mosaicity and finite domains, and the beam has bandwidth, all of which thicken the sphere into a shell. `probixi` models that shell width as a rocking radius

$$R(|q|) = r_{\text{size}} + \tfrac12\,\eta\,|q| + \tfrac12\,\lambda\,\text{(bandwidth)}\,|q|^2,$$

with a constant domain-size term, a mosaicity term $\eta$ linear in $|q|$, and a bandwidth term quadratic in $|q|$. A reflection is kept when $|\varepsilon| < \lambda \cdot \text{predict\_sigma} \cdot R(|q|)$. The mosaicity $\eta$ can be fit per crystal from the radial spread of the indexed reflections.

Each predicted spot is then box-integrated on the background-subtracted (excess) image. The background is already removed, so the intensity is just a sum, and the variance is the summed background noise plus the signal's own shot noise,

$$I = \sum_{\text{box}} \text{excess},\qquad \sigma = \sqrt{\underbrace{\textstyle\sum_{\text{box}} \text{var}}_{\text{background}} + \underbrace{\max(I,0)\cdot\text{gain}}_{\text{shot noise}}}.$$

Overlapping boxes assign each pixel to its nearest predicted center so intensity is never double-counted, and a predicted spot that lands near an observed peak is snapped onto that peak's centroid before integrating. A per-crystal resolution limit is *reported* (from where the shell-averaged $I/\sigma$ crosses 1), but reflections are integrated all the way to the detector edge regardless. The merge, not any single frame, decides where the data runs out.

## The enrichment gate

One last statistical check guards against a plausible-looking but wrong orientation. Once the full lattice is predicted, `probixi` asks the image directly. Of the predicted spot positions, how many land on above-threshold signal ($n_{\text{bright}}$)? Compare that to the rate $p$ at which *any* pixel is bright by chance, and you get an enrichment ratio and a p-value,

$$\text{enrichment} = \frac{n_{\text{bright}}/M}{p},\qquad \text{enrich\_p} = P\big(X \ge n_{\text{bright}}\big),\ \ X\sim\text{Poisson}(Mp).$$

An enrichment near 1 means the predicted spots hit signal about as often as random, a noise indexing. A value $\gg 1$ is a genuine lattice. The p-value turns the enrichment into a false-discovery decision. The [well-diffracting vignette](../vignettes/good_diffraction.md) covers when to lean on this gate, and the [poorly-diffracting vignette](../vignettes/bad_diffraction.md) covers when to ease off, since a weak lattice honestly produces few bright spots.

Every indexed frame carries these numbers, its cell, its orientation, and its integrated reflections into the output. See [Input/Output](io.md).
