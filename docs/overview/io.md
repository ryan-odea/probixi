# Inputs and outputs

`probixi` reads CrystFEL inputs and writes a CrystFEL stream or a DuckDB database.

## Inputs

1. A CrystFEL geometry file for the detector.
2. A CrystFEL cell file. It only has to be approximate, because `probixi` re-centres it on the data.
3. A list file of the HDF5 or CXI files to process.

Any pixel mask stored with the data is read and combined with the masks `probixi` learns. The geometry's `mask` and `mask_file` say where the mask is stored; without `mask_file`, it is read from the first data file. `probixi` also reads the CrystFEL 0.10 names of the first mask, `mask0_data`, `mask0_file`, `mask0_goodbits` and `mask0_badbits`; it ignores `mask1_*` and beyond with a warning. A geometry that sets `mask_good` or `mask_bad` without a `mask` path, as Cheetah's SACLA geometries do, reads the NeXus pixel mask or Cheetah's `/metadata/pixelmask` from that file.

`photon_energy` takes a number or an HDF5 path, each with an optional CrystFEL unit: `photon_energy = 9.5 keV` and `photon_energy = /LCLS/photon_energy_eV eV` both work. A value without a unit is in eV.

When the path gives one photon energy per frame, each frame is indexed, predicted, and integrated at its own wavelength, and the stream and database record it per frame. Two layouts do this: a per-event path (`photon_energy = /%/photon_energy_ev`), and a dataset with one entry per frame of a stacked file (`photon_energy = /LCLS/photon_energy_eV`). A frame without a finite reading takes the mean of the file. A dataset of any other length gives that mean to every frame.

The pixel size comes from `res`. A geometry may set `res` at the top level, per panel, or both; a panel's own value overrides the top-level one. `probixi` takes one pixel size for the whole detector and stops when the panels differ.

`--mask FILE` (or `Probixi(mask_file=...)`) reads the mask from `FILE` instead, for a mask that lives in an HDF5 master file while the list names the data files, or a `mask_file` that has moved. `probixi` reads the geometry's `mask` dataset if `FILE` has it, and otherwise the NeXus `/entry/instrument/detector/pixel_mask` or the same mask under `detectorSpecific/`. A NeXus pixel is bad when any flag bit is set, unless the geometry sets `mask_good` or `mask_bad`. An explicit `--mask` that cannot be read stops the run; an unreadable geometry mask gives a warning, and the run continues without it.

## Frame reading and decompression

Frames are read lazily on a background thread, a few batches ahead of the GPU. On CUDA with the `gpu` extra, bitshuffle-LZ4 datasets chunked one frame per chunk are decompressed on the GPU, and the frames never pass through host memory. Other codecs, and every codec without the extra, are decompressed by CPU threads.

## Outputs

### CrystFEL stream

An output ending in `.stream` is the CrystFEL text format read by `partialator` and `process_hkl`. It carries only what CrystFEL reads. A reflection whose signal disk touches the mask, the frame edge, or another predicted disk is left out of the stream, as is a peak bordering the mask; the database keeps both, flagged by pixel counts.

The stream header states one `--int-radius`, and a reader that rebuilds the counting variance from it assumes every reflection has that disk. `probixi` therefore integrates every reflection of a stream with one disk, the calibrated one or the one given by `--int-radius R_DISK,R_IN,R_OUT`, and warns that it does so. The disk that grows with resolution ([Integration aperture](indexing.md#integration-aperture)) goes to the database, which records the pixel count of each disk. `--disk shell` forces the growing disk into a stream; the header then states the largest disk. Give every part of a split run the same `--int-radius`, because each part otherwise calibrates its own radii and the joined stream keeps the header of the first part. For TorchSX, write a database.

> NOTE: The stream output may be deprecated in favour of the database.

### Database

An output ending in `.db` or `.duckdb` is a [DuckDB](https://duckdb.org/) database. Every file-event has a row in `frames`, indexed or not, and lattices, reflections, and peaks link to it by `frame_id`, the first 16 hex digits of the SHA-1 of `filename//event`. The key is deterministic, so a peaks-only pass, an indexed pass, and separate multi-GPU shards agree on which row is which frame. Under `--start`/`--stop` or a multi-GPU run, each writer fills only the frame range it owns.

Four metadata tables are written once, at the start of the run.

| Table | Rows | Contents |
| --- | --- | --- |
| `geometry` | one | The resolved detector geometry: `beam_center_row`/`beam_center_col`, `clen`, `pixel_size`, `wavelength`, `photon_energy_eV`, `adu_per_photon`, `n_panels`, and the full text of the geometry file in `geometry_file`. |
| `panels` | one per panel | Each panel's `name` and fast-scan and slow-scan pixel bounds (`min_fs`, `max_fs`, `min_ss`, `max_ss`). |
| `cell` | one | The target unit cell: edges `a_A`/`b_A`/`c_A`, angles `alpha_deg`/`beta_deg`/`gamma_deg`, `volume_A3`, and the symmetry labels `lattice_type`, `centering`, `unique_axis`. |
| `integration` | one | The integration recipe the run applied: disk radius `r_peak` and annulus radii `r_gap`/`r_bg` (px), the gain `adu_per_photon` (measured when the flux model is on, otherwise the geometry's), the local-background annulus size `bg_annulus_pixels`, and the `aperture` rule. The five `bg_model_var_*` columns describe the scale fitted to the model background variance at the end of the run, as set out below the result tables. |
| `calibration` | one per rank | What the calibration settled: the probixi version and database `schema_version`, the gain and read variance, whether the flux model is on, the detection threshold with its noise budget and tail ratio, the peak mixture (`kappa`, `prior_peak`, `var_scale`), the aperture, the camera length, and the cell in use. A multi-GPU run calibrates on every rank, and `rank` identifies the worker. |

Four result tables are joined on `frame_id`, and per lattice on `crystal_id`.

| Table | Rows | Contents |
| --- | --- | --- |
| `frames` | one per file-event | Provenance (`frame_index`, `filename`, `event`, `serial`), the `indexed` flag, the peak search (`n_peaks`, `peak_resolution_nm_inv`), the per-frame fluence scale (`scale`, `scale_sigma`), and `num_reflections` over every lattice on the frame. Per-event (`/%/`) data adds `event_name`. `photon_energy_eV` is set when the geometry reads the photon energy per frame. `rank` is the worker that processed the frame, so `frames JOIN calibration USING (rank)` gives each frame its calibration. |
| `crystals` | one per accepted lattice | `lattice_index` orders the lattices on one frame. Counts (`n_indexed`, `num_reflections`), fit quality (`rmsd`, `mosaicity_deg`, `profile_radius_nm_inv`), enrichment (`enrichment`, `n_bright`, `enrich_p`), resolution (`diffraction_limit_nm_inv`), the refined cell (`cell_*`), and the reciprocal basis (`astar_*`, `bstar_*`, `cstar_*`, in nm⁻¹). |
| `reflections` | one per integrated reflection | Miller indices `h`/`k`/`l`, `intensity`, `sigma`, `peak`, `background`, position (`fs`, `ss`, `panel`), and `resolution_nm_inv`. Of the `n_pixels_total` pixels in the nominal disk, `n_pixels` were summed, `n_masked` fell on the mask or frame edge, and `n_overlap` are shared with another predicted disk. `background_model` and `background_model_var` are the noise model's background under the summed pixels and its variance ([method](method.md#73-the-model-background-under-the-disk)), the variance already multiplied by `integration.bg_model_var_scale`. With `n_bg = integration.bg_annulus_pixels` and gain g, the uncertainty of the model-subtracted intensity `I_model = intensity + n_pixels * background - background_model` is `sigma^2 = background_model_var * (1 + n_bg / n_pixels) + g * max(I_model, 0)`. |
| `peaks` | one per searched peak | Position (`fs`, `ss`, `panel`), `intensity`, `resolution_nm_inv`, `n_pixels`, and `n_masked`, the blob pixels bordering the mask or frame edge. `photons` and `background_photons` are the blob's total and background counts in photons. |

At the end of the run, probixi multiplies `reflections.background_model_var` by a factor k fitted to the measured scatter of `I_model`. The reference reflections lie past their own crystal's diffraction limit, where they carry little signal, but no finer than the run's resolution, the 10th percentile in d of every lattice's limit and TorchSX's default `--dmin`. They have no masked or overlapping pixels, and they belong to lattices with `enrich_p <= 0.001`, the default of `--enrich-alpha`. This threshold is fixed and does not depend on `--enrich-gate` or `--enrich-alpha`, which decide only which lattices are written. The database records it as `integration.bg_model_var_alpha`, and TorchSX reads the database through the same gate by default. Lattices at chance level scatter far less than the model predicts past their limits and would pull k down. On one b2ar dataset, k fell from 0.588 to 0.201 when they were included, while every threshold from 1e-6 to 0.2 gave k within 0.002 of 0.588. With fewer than 1000 reference reflections on passing lattices, or with no `enrich_p` recorded, the fit uses every lattice and issues a warning. The `integration` table records k in `bg_model_var_scale`, the resolution cut in `bg_model_var_dmin_A` (Å), the reference set in `bg_model_var_ref` (`enrich_pass` or `all`), the threshold of a gated fit in `bg_model_var_alpha`, and the number of reflections fitted, at most 2,000,000, in `bg_model_var_n_ref`. All five are NULL when too few reflections lie in the window. A multi-GPU run fits k once, over the merged database. For a database written before the fit used only passing lattices, `probixi.io.db.recalibrate_model_variance` divides out the recorded scale and fits it again.

Frames that never indexed have `indexed = FALSE`, so the indexing rate is `SELECT AVG(indexed::INT) FROM frames`. A frame has one row however many lattices it yields: count lattices with `SELECT COUNT(*) FROM crystals` and indexed frames with `SELECT COUNT(*) FROM frames WHERE indexed`. `--peaks-only` fills `peaks` and leaves `crystals` and `reflections` empty.

TorchSX reads the database directly (`torchsx.train run.duckdb --spacegroup ...`).

The database records `probixi.__version__`, the version of the source that wrote it. The two packages name each other's version. `probixi.SCHEMA_VERSION` (2) is the layout above, and `probixi.TORCHSX_VERSION` is the first TorchSX release that reads it. TorchSX holds the counterpart in `torchsx.SUPPORTED_DB_SCHEMA` and `torchsx.PROBIXI_VERSION`. Neither package depends on the other. Where both are installed, probixi warns before it writes a database that the installed TorchSX cannot read. Schema 1 is the layout without the `rank` columns and the `bg_model_var_*` columns.

### CXI peak export

With `--peaks-only` and a directory as `-o`, `probixi` writes one `.cxi` per input file with the raw frames externally linked, a `peaks.lst`, and a companion `<geometry>_cxi.geom`, ready for `indexamajig --peaks=cxi`. With a `.db` file as `-o`, the peaks go into the database instead.

## Indexing ambiguity

When the lattice symmetry is higher than the point group, each crystal can be indexed in more than one way. `probixi-resolve` implements the method of Brehm and Diederichs, as in CrystFEL's `ambigator`, on a DuckDB output. It clusters the crystals into the two indexing choices, reindexes those on the wrong side by transforming `h k l`, the reciprocal axes, and the cell, and records each assignment in an `ambiguity` table.

```bash
probixi-resolve -i run.duckdb -y 3 -w 321 -o resolved.duckdb
```

## Frame rendering

`render_frame` (or `--render` on the CLI) draws a frame with detected peaks as open circles and predicted reflections as crosses. The display is scaled to percentiles of the valid pixels only, so masked and dead pixels do not set the contrast.
