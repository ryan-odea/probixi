# Strong diffraction

On strongly diffracting crystals the Bragg peaks lie well above the background. The main risk is an incorrect lattice, because a frame with many peaks offers enough constraints for a wrong orientation to fit them.

## The run

This example uses bacteriorhodopsin collected at a synchrotron on an Eiger detector.

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

The enrichment gate ([derivation](../overview/indexing.md#the-enrichment-gate)) tests each lattice against the image. After a frame is indexed, `probixi` predicts the full lattice and counts how many predicted positions land on above-threshold signal (`n_bright`). It compares that count with the rate at which any valid pixel clears the same threshold, which gives

- an **enrichment**, the bright rate of predicted spots over the background bright rate, near 1 for a chance lattice and well above 1 for a real one, and
- a Poisson tail probability, `enrich_p`, of observing at least `n_bright` bright spots by chance.

With `--enrich-gate`, a lattice is kept only when `enrich_p` is at or below `--enrich-alpha` (default 0.001). The parameter is a false-discovery level, not an enrichment cutoff, and has the same meaning on every dataset. Every crystal records its `enrichment`, `n_bright`, and `enrich_p` whether or not the gate is on.

> NOTE: The gate is off by default. It is still being tested to determine whether it becomes the default.

## Using another indexer

`--peaks-only` writes the peaks in CXI format with a list file and a companion geometry, and they replace peakfinder8's peaks in an existing `indexamajig` workflow.

```bash
probixi -i br.lst -g synchrotron.geom -o peaks/ --peaks-only
indexamajig -i peaks/peaks.lst -g peaks/synchrotron_cxi.geom --peaks=cxi --no-revalidate ...
```

A conventional indexer treats the exported list as certain and does not use the detection posteriors ([Weak diffraction](bad_diffraction.md#peakfinding-and-indexing-together)).
