# Variable flux

At a free-electron laser (FEL) the background is not stationary. Pulse energy and beam position change from shot to shot, and one frame's background can be several times brighter than the next. Hit rates are low, so most frames are blank. Multi-panel detectors such as the Jungfrau add per-panel gain and offset structure.

## The run

This example uses the β<sub>2</sub>-adrenergic receptor collected at an FEL on a Jungfrau 4M detector.

```bash
probixi \
  -i b2ar.lst \
  -g jungfrau4m.geom \
  -p b2ar.cell \
  -o b2ar.stream \
  --flux-variance
```

`--flux-variance` is the only change from a synchrotron run.

![image](../assets/br_fel.gif)

The animation shows the noise model settling on bacteriorhodopsin at an FEL.

## What `--flux-variance` does

By default `probixi` whitens each frame against one running variance, measured at the run's typical fluence. Photon noise grows with signal, and this single variance therefore understates the noise on a bright shot and overstates it on a dim one. On a dim shot the real spots are then suppressed.

With `--flux-variance`, calibration fits a photon-transfer curve across the seed frames,

$$\sigma^2 = \sigma^2_{\text{read}} + g\,\mu ,$$

where the intercept is the read variance and the slope $g$ is the detector gain in ADU per photon. At detection time each pixel is whitened against the variance the curve predicts at that frame's own background level. A bright shot and a dim one are then tested on the same footing. `--flux-var-floor` (default 0.15) bounds the predicted variance below by that fraction of the running variance. The fitted gain also goes to integration, so the shot-noise term in each σ(I) is in the right units.

It is opt-in because it only helps when flux varies. On steady synchrotron exposures the running variance already describes the noise. It has so far been tested on one FEL dataset.
