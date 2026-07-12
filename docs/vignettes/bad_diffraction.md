# Poorly Diffracting Samples

You may notice that the [call from the well-diffracting walkthrough](good_diffraction.md) is exactly the same here, minus the `--enrich-gate`: on weak data a genuine lattice may put only a handful of predicted spots on measurable signal, so a gate tuned for strong shots would throw away real hits along with the noise. Even in poorly diffracting environments, `probixi` aims to let you get the most from your data, without a speed penalty from low signal-to-noise. Here, we'll work through an example of C1C2 crystals at the synchrotron.

```bash
probixi \
  -i c1c2.lst \
  -g synchrotron.geom \
  -p c1c2.cell \
  -o c1c2.stream \
  --device cuda \
  --gif noise_model.gif
```

## Turning up sensitivity

Because weak signal leaves little margin, a clean background estimate matters most. `--gif noise_model.gif` is the quickest way to confirm the noise model has converged, and calibrating on more frames with `--seed-frames` steadies it if a hit rate looks lower than expected. The auto-calibrated detection threshold usually handles the rest, but if you do need to push sensitivity harder, `--target-noise-peaks` raises it by tolerating a few more noise blobs per frame.

> NOTE: `--target-noise-peaks` is still being tested to ensure you don't merge more noise downstream

## Sparse patterns index themselves

With few peaks, seeding an orientation is the hard part, and it is handled for you. The Fibonacci-sphere seeder adapts per frame: on peak-starved frames it automatically widens the orientation search (at the cost of some computation), more sample directions and in-plane angles, so a sparse pattern still gets a fair shot, then narrows back on richer frames. A candidate must still explain a minimum handful of peaks (six by default) to be accepted, so an information-starved frame simply goes unindexed rather than indexing wrongly.

Since the gate is off, nothing is dropped for weak prediction, but every indexed frame still records its `enrichment`, `n_bright`, and `enrich_p` in the output (see the `frames` table). If overprediction does turn out to be a concern on a given dataset, you can inspect that distribution and threshold it after the fact.
