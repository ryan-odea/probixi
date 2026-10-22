# Weak diffraction

A threshold-based peakfinder needs a gap between Bragg signal and background in which to place its threshold. On weakly diffracting samples the gap narrows, and on many frames no single threshold admits the real peaks without also admitting noise. `probixi` instead fixes the false-positive rate against a calibrated null, and that rate does not depend on the strength of the signal.

The example is the chimeric channelrhodopsin C1C2, collected at a synchrotron on an Eiger detector at 200 Hz.

```bash
probixi \
  -i c1c2.lst \
  -g synchrotron.geom \
  -p c1c2.cell \
  -o c1c2.stream \
  --gpus 4 \
  --gif noise_model.gif
```

![image](../assets/c1c2_sync.gif)

Apart from `--gpus 4`, the call differs from the strong-diffraction call only in omitting `--enrich-gate`. A real but weak lattice puts few of its predicted spots on measurable signal, and a strict gate would discard real lattices along with false ones. The recorded `enrichment` and `enrich_p` still allow false lattices to be assessed afterwards.

## Posterior-weighted indexing

On weak data the built-in indexer is preferable to exporting the peaks with `--peaks-only`. A conventional indexer receives a thresholded list and treats every entry as certain. The built-in indexer weights every peak by its detection posterior, so a faint detection contributes in proportion to its evidence.

On frames with fewer than 30 peaks the orientation search doubles the number of trial directions and rolls. A candidate still requires six inliers and a cell that matches the target, and a frame that offers too little is reported as unindexed.

## Calibration variance and the noise budget

- `--gif` shows whether the background model has settled, and a larger `--seed-frames` reduces the variance of the calibration.
- A larger `--target-noise-peaks` raises sensitivity by allowing more noise detections per signal-free frame ([noise budget](../overview/peakfinding.md#the-threshold-from-a-noise-budget)), and admits more noise into the peak list.
