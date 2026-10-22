# Strong diffraction

On strongly diffracting crystals the Bragg peaks lie well above the background. The main risk is an incorrect lattice: a frame with many peaks offers enough constraints for a wrong orientation to fit them.

The example is bacteriorhodopsin, collected at a synchrotron on an Eiger detector.

```bash
probixi \
  -i br.lst \
  -g synchrotron.geom \
  -p br.cell \
  -o br.stream \
  --gif noise_model.gif \
  --enrich-gate
```

![image](../assets/br_sync.gif)

## The enrichment gate

After a frame is indexed, `probixi` predicts the full lattice and counts the predicted positions that land on above-threshold signal (`n_bright`). Against the rate at which any valid pixel clears the same threshold, this gives two numbers per crystal: the `enrichment`, near 1 for a chance lattice and well above 1 for a real one, and `enrich_p`, the probability that a chance lattice agrees with the image as well ([derivation](../overview/indexing.md#the-enrichment-gate)).

With `--enrich-gate`, a lattice is kept only when `enrich_p` is at most `--enrich-alpha` (default 0.001). The parameter is a significance level on `enrich_p`, not a cutoff on the enrichment, and therefore means the same on every dataset. Every crystal records its `enrichment`, `n_bright`, and `enrich_p` whether or not the gate is on.

> NOTE: The gate is off by default while it is tested as a candidate default.

## Peak export to `indexamajig`

`--peaks-only` writes the peaks in CXI format with a list file and a companion geometry, which replace peakfinder8's peaks in an existing `indexamajig` workflow.

```bash
probixi -i br.lst -g synchrotron.geom -o peaks/ --peaks-only
indexamajig -i peaks/peaks.lst -g peaks/synchrotron_cxi.geom --peaks=cxi --no-revalidate ...
```

A conventional indexer treats the exported list as certain and does not use the detection posteriors ([Weak diffraction](bad_diffraction.md#posterior-weighted-indexing)).
