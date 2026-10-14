# Getting started

A run requires a list of files, the detector geometry, and an approximate unit cell. The noise model, the detection threshold, and the integration aperture are learned from the data.

## Running from the CLI

```bash
probixi -i files.lst -g my_geometry.geom -p my_unit.cell -o output.db
```

- `-i` is a CrystFEL list file of HDF5 or CXI inputs.
- `-g` is a CrystFEL geometry file.
- `-p` is a CrystFEL cell file. It only has to be approximate, because `probixi` re-centres it on the refined cells of the first lattices.
- `-o` sets the output, and its extension sets the format. `.stream` writes a CrystFEL stream, and `.db` or `.duckdb` writes a DuckDB database.

The device defaults to `auto`, which picks CUDA, then Apple MPS, then CPU.

### Calibration

| Option | Default | What it does |
| --- | --- | --- |
| `--seed-frames` | 32 | Frames drawn at random from the run to calibrate the noise model and threshold. More frames steady the estimate. |
| `--target-noise-peaks` | 5 | Noise budget, the number of spurious detections allowed on a signal-free frame. Lower is stricter, higher is more sensitive. |
| `--warmup-frames` | 16 | Frames seen before pixels with zero variance are masked as dead. |
| `--recalibrate-every` | off | Rebuild the calibration every N input frames and freeze the statistics in between. Use it for a run that spans an abrupt change, such as a jet restart. A value of 0 freezes the initial estimate for the whole run. |
| `--noise-mode` | `online` | `online` keeps updating the background after calibration. `per_frame` rebuilds it for each frame. |
| `--flux-variance` | off | Scale the noise with each frame's own level. For free-electron laser data with shot-to-shot flux changes. See [Variable flux](variable_flux.md). |
| `--flux-var-floor` | 0.15 | Lower bound on the flux-scaled variance, as a fraction of the running variance. |
| `--force-all` | off | Disable the blank-shot screen and index every frame, however dark. |
| `--random-seed` | 1988 | Seed for every random draw. A run is fully specified by its inputs, version, and this seed. |

### Indexing and integration

| Option | Default | What it does |
| --- | --- | --- |
| `--enrich-gate` | off | Drop lattices whose predicted spots land on signal no more often than chance. Recommended on strong data and still being tested as a possible default. See [Strong diffraction](good_diffraction.md). |
| `--enrich-alpha` | 0.001 | Largest chance probability a lattice may have under `--enrich-gate`. |
| `--aperture` | `auto` | Integration disk. `snr` maximises background-limited signal-to-noise, `flux` captures the whole spot, and `auto` keeps `snr` unless it would bias intensities differently between crystals. |
| `--max-lattices` | 1 | Lattices to search per frame. Indexed peaks are removed between passes. |
| `--cell-calibrate` / `--no-cell-calibrate` | on | Re-centre the target cell on the median refined cell, at most twice. |
| `--cell-calibrate-after` | 200 | Accepted lattices pooled before each re-centring. |
| `--no-refine-cell` | off | Hold every crystal at the target cell and refine orientation only. |
| `--batch-size` | 8 | Frames per batched refinement pass. |

### Run control and output

| Option | What it does |
| --- | --- |
| `--peaks-only` | Peakfinding only. `-o` is then a directory of CXI peak files for `indexamajig --peaks=cxi`, or a DuckDB file. No cell needed. |
| `--start`, `--stop` | Frame range, start inclusive and stop exclusive. |
| `--device` | Torch device, for example `cuda`, `cuda:1`, `mps`, or `cpu`. |
| `--gpus N`, `--devices` | Split the run into contiguous blocks across N GPUs, or an explicit device list, and merge the outputs in order. |
| `--threads-per-worker` | CPU threads per GPU worker. |
| `--panel` | Panel name for peaks that fall outside every geometry panel. Default `0`. |
| `--gif` | Write an animation of the noise model over the seed frames. |
| `--render`, `--render-out` | Draw one frame (an index or `filename//event`) with its peaks and predicted reflections. |
| `-q`, `--quiet` | Suppress the progress line. |

## Checking a run

- **Background.** `--gif noise.gif` animates the running mean, its radial profile, and the per-batch drift over the seed frames. Structure that persists in the drift panel has not been absorbed by the model.
- **Calibration.** After calibration `probixi` prints one line with the peak contrast (`kappa`), the peak prior (`prior_peak`), the variance scale (`var_scale`), the detection threshold (`mf_threshold`), any learned beamstop limit, blank-frame floor, and shadow fraction, and the integration radii with the aperture it chose and why. A `var_scale` far from 1 means the raw per-pixel variance did not match the background scatter. An `mf_threshold` pinned at 8.0, the top of its search range, means no seed frame looked signal-free, and the usual cause is structure the model does not capture.
- **Detections.** `--render 1234 --render-out frame.png` overlays detected peaks (open circles) and predicted reflections (crosses) on one frame. It runs directly after calibration, and without `-o` the run stops there.
- **Lattices.** Every indexed crystal carries its `enrichment` and `enrich_p` in the output, gated or not. A distribution centred near 1 means the lattices are no better than chance.

## Recovering results

A `.stream` is a standard CrystFEL stream, read by `partialator` and `process_hkl`.

> NOTE: The stream output may be deprecated in favour of the database. It exists to compare `probixi` one-to-one with CrystFEL pipelines.

### Database

The `.db` (or `.duckdb`) output is a [DuckDB](https://duckdb.org/) file. Three **metadata** tables are written once, at the start of the run.

| Table | Rows | What it stores |
| --- | --- | --- |
| `geometry` | one | The resolved detector geometry: `beam_center_row`/`beam_center_col`, `clen`, `pixel_size`, `wavelength`, `photon_energy_eV`, `adu_per_photon`, `n_panels`, and the full text of the geometry file in `geometry_file`. |
| `panels` | one per panel | Each panel's `name` and fast/slow-scan pixel bounds (`min_fs`, `max_fs`, `min_ss`, `max_ss`). |
| `cell` | one | The target unit cell: edges `a_A`/`b_A`/`c_A`, angles `alpha_deg`/`beta_deg`/`gamma_deg`, `volume_A3`, and the symmetry labels `lattice_type`, `centering`, `unique_axis`. |

Four **result** tables are linked by `frame_id`, the first 16 hex digits of the SHA-1 of `filename//event`, and per lattice by `crystal_id`.

| Table | Rows | What it stores |
| --- | --- | --- |
| `frames` | one per file-event | Provenance (`frame_index`, `filename`, `event`, `serial`), an `indexed` flag, the peak search (`n_peaks`, `peak_resolution_nm_inv`), the per-frame fluence scale (`scale`, `scale_sigma`), and `num_reflections` over every lattice on the frame. |
| `crystals` | one per accepted lattice | Keyed by `crystal_id` and joined to its frame by `frame_id`. `lattice_index` orders the lattices on one frame. Counts (`n_indexed`, `num_reflections`), fit quality (`rmsd`, `mosaicity_deg`, `profile_radius_nm_inv`), enrichment (`enrichment`, `n_bright`, `enrich_p`), resolution (`diffraction_limit_nm_inv`), the refined cell (`cell_*`), and the reciprocal basis (`astar_*`, `bstar_*`, `cstar_*`, in nm⁻¹). |
| `reflections` | one per integrated reflection | Keyed by `crystal_id` and `frame_id`. Miller indices `h`/`k`/`l`, `intensity`, `sigma`, `peak`, `background`, position (`fs`, `ss`, `panel`), `resolution_nm_inv`, and the disk bookkeeping: `n_pixels` summed, `n_pixels_total` in the nominal disk, of which `n_masked` were lost to the mask or frame edge and `n_overlap` are shared with another predicted disk. A disk with any masked or shared pixel, or no usable pixel at all, is kept here but left out of the stream. |
| `peaks` | one per searched peak | Keyed by `frame_id`. Position (`fs`, `ss`, `panel`), `intensity`, `resolution_nm_inv`, `n_pixels`, and `n_masked`, the blob pixels bordering the mask or frame edge; a peak with any is left out of the stream's peak list. |

- Frames that never indexed still get a `frames` row with `indexed = FALSE`, so the indexing rate is `SELECT AVG(indexed::INT) FROM frames`. Under `--start`/`--stop` or multi-GPU runs, each writer fills only the frame range it owns.
- A frame keeps one `frames` row however many lattices it yields. Count lattices with `SELECT COUNT(*) FROM crystals` and frames with `SELECT COUNT(*) FROM frames WHERE indexed`. Per-lattice statistics live on `crystals`.
- `--peaks-only` fills `peaks`, not `reflections`. Those frames have `indexed = FALSE` and empty `crystals` and `reflections` tables.

## Using `probixi` with Python

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

`index_stream` returns a **lazy** `IndexStream`. Nothing is read or indexed until you drain it with `to_db`, `to_stream`, `collect`, or a loop. Each `IndexResult`, and every tensor it carries, stays on the configured device, and only the writers move data off it. A stream is single-use.
