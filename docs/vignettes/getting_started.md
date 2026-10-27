# Getting started

A run requires a list of files, the detector geometry, and an approximate unit cell. The noise model, the detection threshold, and the integration aperture are learned from the data.

## Command-line interface

```bash
probixi -i files.lst -g my_geometry.geom -p my_unit.cell -o output.db
```

- `-i` is a CrystFEL list file of HDF5 or CXI inputs.
- `-g` is a CrystFEL geometry file.
- `-p` is a CrystFEL cell file. It only has to be approximate, because `probixi` re-centres it on the refined cells of the first lattices.
- `-o` sets the output, and its extension sets the format: `.stream` writes a CrystFEL stream, and `.db` or `.duckdb` a DuckDB database.
- `--mask` names the HDF5 file that holds the bad-pixel mask and overrides the geometry's `mask_file`. Use it when the list names the `_data_*.h5` files of an Eiger run and the mask is in the master file, or when the geometry's mask path no longer resolves ([Inputs and outputs](../overview/io.md)).

The device defaults to `auto`, which picks CUDA, then Apple MPS, then CPU.

### Calibration

| Option | Default | Effect |
| --- | --- | --- |
| `--seed-frames` | 32 | Frames drawn at random from the run to calibrate the noise model and threshold. More frames steady the estimate. |
| `--target-noise-peaks` | 5 | Noise budget: spurious detections allowed on a signal-free frame. Lower is stricter, higher more sensitive. |
| `--warmup-frames` | 16 | Frames seen before pixels with zero variance are masked as dead. |
| `--noise-decay` | 1 | Weight of the running background statistics. 1 averages the whole run, and d < 1 remembers about 1/d frames, for a sample or beam that changes within the run. |
| `--recalibrate-every` | off | Rebuild the calibration every N input frames and freeze the statistics in between, for a run that spans an abrupt change such as a jet restart. 0 freezes the initial estimate for the whole run. |
| `--noise-mode` | `online` | `online` keeps updating the background after calibration; `per_frame` rebuilds it for each frame. |
| `--flux-variance` | auto | Scale the noise with each frame's own level. Calibration turns it on when the background level changes between shots beyond photon noise, as at free-electron lasers. `--no-flux-variance` forces it off ([Variable flux](variable_flux.md)). |
| `--flux-var-floor` | 0.15 | Lower bound on the flux-scaled variance, as a fraction of the running variance. |
| `--force-all` | off | Disable the blank-shot screen so every frame trains the background. |
| `--random-seed` | 1988 | Seed for the random draw of calibration and aperture frames. Every other draw has a fixed seed, so a run is fully specified by its inputs, the `probixi` version, and this seed. |

### Indexing and integration

| Option | Default | Effect |
| --- | --- | --- |
| `--enrich-gate` | off | Drop lattices whose predicted spots land on signal no more often than chance. Recommended on strong data ([Strong diffraction](good_diffraction.md)). |
| `--enrich-alpha` | 0.001 | Largest chance probability a lattice may have under `--enrich-gate`. |
| `--aperture` | `auto` | Integration disk. `snr` maximises the background-limited signal-to-noise ratio, `flux` captures the whole spot, and `auto` keeps `snr` unless it biases intensities differently between crystals by more than the extra background noise of the `flux` disk. |
| `--max-lattices` | 1 | Lattices to search per frame. Indexed peaks are removed between passes. |
| `--cell-calibrate` / `--no-cell-calibrate` | on | Re-centre the target cell on the median refined cell at calibration and at most twice more during the run. |
| `--cell-calibrate-after` | 200 | Accepted lattices pooled before each re-centring during the run. |
| `--no-refine-cell` | off | Hold every crystal at the target cell and refine the orientation only. |
| `--batch-size` | 8 | Frames per batched refinement pass. |

### Run control and output

| Option | Effect |
| --- | --- |
| `--peaks-only` | Peakfinding only, with no cell. `-o` is then a directory of CXI peak files for `indexamajig --peaks=cxi`, or a DuckDB file. |
| `--start`, `--stop` | Frame range, start inclusive and stop exclusive. |
| `--device` | Torch device, for example `cuda`, `cuda:1`, `mps`, or `cpu`. |
| `--gpus N`, `--devices` | Split the run into contiguous blocks across N GPUs, or an explicit device list, and merge the outputs in order. |
| `--threads-per-worker` | CPU threads per GPU worker. |
| `--panel` | Panel name for peaks that fall outside every geometry panel. Default `0`. |
| `--gif` | Write an animation of the noise model over the first `--seed-frames` frames of the run. |
| `--render`, `--render-out` | Draw one frame (an index, `filename//event`, or `filename tag-N//`) with its peaks and predicted reflections. |
| `-q`, `--quiet` | Suppress the progress line. |

## Run diagnostics

- `--gif noise.gif` animates the running mean, its radial profile, and the per-batch drift over the first frames of the run. Structure that persists in the drift panel is not absorbed by the model.
- After calibration `probixi` prints one line with the calibrated quantities. A `var_scale` far from 1 means the raw per-pixel variance did not match the background scatter. An `mf_threshold` of 8.0, the top of its search range, means no seed frame looked signal-free; the usual cause is structure the model does not capture.
- `--render 1234 --render-out frame.png` overlays detected peaks (open circles) and predicted reflections (crosses) on one frame. It runs directly after calibration, and without `-o` the run stops there.
- Every indexed crystal carries its `enrichment` and `enrich_p` in the output, gated or not. A distribution centred near 1 means the lattices are no better than chance.

## Output formats

A `.stream` is a standard CrystFEL stream, read by `partialator` and `process_hkl`.

> NOTE: The stream output may be deprecated in favour of the database. It exists to compare `probixi` one-to-one with CrystFEL pipelines.

A `.db` or `.duckdb` output is a [DuckDB](https://duckdb.org/) database with one row per file-event in `frames`, one per lattice in `crystals`, and the integrated `reflections` and searched `peaks` ([schema](../overview/io.md#database)). The indexing rate is `SELECT AVG(indexed::INT) FROM frames`.

## Python interface

The CLI wraps `probixi.Probixi`. A run constructs the pipeline, calibrates it, and drains the stream.

```python
from probixi import Probixi

pipe = Probixi(
    list_file="files.lst",
    geometry_file="my_geometry.geom",
    cell_file="my_unit.cell",
    device="cuda",
)

pipe.calibrate(n_seed=32, target_noise_peaks=5.0)

stream = pipe.index_stream(pipe.frames(), batch_size=8)
n = stream.to_db(
    "output.db",
    geometry=pipe.geometry,
    cell=pipe.target_cell,
    geometry_file="my_geometry.geom",
    files=pipe.metadata.files,
)
print(f"indexed {n} frame(s)")
```

`index_stream` returns a lazy `IndexStream`: nothing is read or indexed until it is drained with `to_db`, `to_stream`, `collect`, or a loop. Each `IndexResult`, and every tensor it carries, stays on the configured device, and only the writers move data off it. A stream can be drained once.
