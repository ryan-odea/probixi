# Weak diffraction

A threshold-based peakfinder requires a gap between Bragg signal and background in which to place its threshold. On weakly diffracting samples that gap narrows, and on many frames no single threshold admits the real peaks without also admitting noise. `probixi` instead fixes the false-positive rate against a calibrated null, and that rate does not depend on the strength of the signal.

## The run

This example uses the chimeric channelrhodopsin C1C2 at a synchrotron, recorded on an Eiger detector at 200 Hz.

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

The call differs from the strong-diffraction call only in omitting `--enrich-gate`. A real but weak lattice puts few of its predicted spots on measurable signal, and a gate at a strict level would discard real lattices along with the noise. Overprediction can still be assessed afterwards from the recorded `enrichment` and `enrich_p`.

## Peakfinding and indexing together

On weak data the built-in indexer is preferable to exporting the peaks with `--peaks-only`. A conventional indexer receives a thresholded list and treats every entry as certain. The built-in indexer weights every peak by its detection posterior, and a faint detection contributes in proportion to its evidence rather than being included or excluded outright.

On frames with fewer than 30 peaks the orientation search doubles the number of trial directions and rolls. A candidate still requires six inliers and a cell that matches the target, and a frame with too little information is reported as unindexed rather than assigned an incorrect lattice.

## Sensitivity

- `--gif` shows whether the background model has settled, and a larger `--seed-frames` reduces the variance of the calibration.
- `--target-noise-peaks` raises sensitivity by allowing more noise detections per signal-free frame ([noise budget](../overview/peakfinding.md#the-threshold-from-a-noise-budget)). Higher values also admit more noise into the peak list.
