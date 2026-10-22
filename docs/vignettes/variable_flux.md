# Variable flux

At a free-electron laser (FEL) the background is not stationary. Pulse energy and beam position change from shot to shot, and one frame's background can be several times brighter than the next. Hit rates are low, so most frames are blank. Multi-panel detectors such as the Jungfrau add per-panel gain and offset structure.

## The run

This example uses the β<sub>2</sub>-adrenergic receptor collected at an FEL on a Jungfrau 4M detector.

```bash
probixi \
  -i b2ar.lst \
  -g jungfrau4m.geom \
  -p b2ar.cell \
  -o b2ar.stream
```

The command is the same as for a synchrotron run. Calibration sees that the background level changes from shot to shot and switches the flux model on, and the calibration line reports `flux_variance=on` and, in this case, `(background fluctuation 7.94)`.

![image](../assets/br_fel.gif)

The animation shows the noise model for bacteriorhodopsin at an FEL.

## What `--flux-variance` does

By default `probixi` whitens each frame against one running variance, measured at the run's typical fluence. Photon noise grows with signal, and this single variance therefore understates the noise on a bright shot and overstates it on a dim one. On a dim shot the real spots are then suppressed.

With `--flux-variance`, calibration fits a photon-transfer curve across the seed frames,

$$\sigma^2 = \sigma^2_{\text{read}} + g\,\mu ,$$

where the intercept is the read variance and the slope $g$ is the detector gain in ADU per photon. At detection time each pixel is whitened against the variance the curve predicts at that frame's own background level. `--flux-var-floor` (default 0.15) bounds the predicted variance below by that fraction of the running variance. The fitted gain also sets the shot-noise term of each σ(I) in integration.

## When it switches on

The running variance is measured across frames, so it also holds every shot-to-shot change of the background level, which the per-frame local background then removes again. Calibration compares the two on the brightest quarter of the background in up to 16 seed frames: the median ratio of the running variance to the local variance of the same frame is about 1 for a steady exposure and $1 + \operatorname{var}(s)\,N_\gamma$ when the level scales by $s$ between shots on a background of $N_\gamma$ photons. Above 1.5 the flux model is used. `--flux-variance` and `--no-flux-variance` override the choice.

| dataset | detector, source | ratio | flux model |
|---|---|---|---|
| β<sub>2</sub>AR | Jungfrau 4M, SwissFEL | 7.9 | on |
| rhodopsin | MPCCD, SACLA | 4.2 | on |
| run0037 | Jungfrau, SwissFEL Cristallina | 2.2 | on |
| C1C2 | Eiger 4M, SLS | 0.95 | off |
| bacteriorhodopsin | Eiger 4M, SLS | 0.86 | off |

On steady synchrotron exposures the running variance already describes the noise, and the two models agree.
