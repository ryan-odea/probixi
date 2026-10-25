# Inputs and outputs

`probixi` reads CrystFEL inputs and writes a CrystFEL stream or a DuckDB database.

## Inputs

1. A CrystFEL geometry file for the detector.
2. A CrystFEL cell file. It only has to be approximate, because `probixi` re-centres it on the data.
3. A list file of the HDF5 or CXI files to process.

Any pixel mask stored with the data is read and combined with the masks `probixi` learns. The geometry's `mask` and `mask_file` say where the mask is stored; without `mask_file`, it is read from the first data file. A geometry that sets `mask_good` or `mask_bad` without a `mask` path, as Cheetah's SACLA geometries do, reads the NeXus pixel mask or Cheetah's `/metadata/pixelmask` from that file.

When the geometry reads the photon energy per event (`photon_energy = /%/photon_energy_ev`), each frame is indexed, predicted, and integrated at its own wavelength, and the stream and database record it per frame.

`--mask FILE` (or `Probixi(mask_file=...)`) reads the mask from `FILE` instead, for a mask that lives in an HDF5 master file while the list names the data files, or a `mask_file` that has moved. `probixi` reads the geometry's `mask` dataset if `FILE` has it, and otherwise the NeXus `/entry/instrument/detector/pixel_mask` or the same mask under `detectorSpecific/`. A NeXus pixel is bad when any flag bit is set, unless the geometry sets `mask_good` or `mask_bad`. An explicit `--mask` that cannot be read stops the run; an unreadable geometry mask gives a warning, and the run continues without it.

## Frame reading and decompression

Frames are read lazily on a background thread, a few batches ahead of the GPU. On CUDA with the `gpu` extra, bitshuffle-LZ4 datasets chunked one frame per chunk are decompressed on the GPU, and the frames never pass through host memory. Other codecs, and every codec without the extra, are decompressed by CPU threads.

## Outputs

### CrystFEL stream

An output ending in `.stream` is the CrystFEL text format read by `partialator` and `process_hkl`. It carries only what CrystFEL reads. A reflection whose signal disk touches the mask, the frame edge, or another predicted disk is left out of the stream, as is a peak bordering the mask; the database keeps both, flagged by pixel counts.

> NOTE: The stream output may be deprecated in favour of the database.

### Database

An output ending in `.db` or `.duckdb` is a [DuckDB](https://duckdb.org/) database. Every file-event has a row in `frames`, indexed or not, and lattices, reflections, and peaks link to it by `frame_id`, the first 16 hex digits of the SHA-1 of `filename//event`. The key is deterministic, so a peaks-only pass, an indexed pass, and separate multi-GPU shards agree on which row is which frame. Under `--start`/`--stop` or a multi-GPU run, each writer fills only the frame range it owns.

Four metadata tables are written once, at the start of the run.

| Table | Rows | Contents |
| --- | --- | --- |
| `geometry` | one | The resolved detector geometry: `beam_center_row`/`beam_center_col`, `clen`, `pixel_size`, `wavelength`, `photon_energy_eV`, `adu_per_photon`, `n_panels`, and the full text of the geometry file in `geometry_file`. |
| `panels` | one per panel | Each panel's `name` and fast-scan and slow-scan pixel bounds (`min_fs`, `max_fs`, `min_ss`, `max_ss`). |
| `cell` | one | The target unit cell: edges `a_A`/`b_A`/`c_A`, angles `alpha_deg`/`beta_deg`/`gamma_deg`, `volume_A3`, and the symmetry labels `lattice_type`, `centering`, `unique_axis`. |
| `integration` | one | The integration recipe the run applied: disk radius `r_peak` and annulus radii `r_gap`/`r_bg` (px), the gain `adu_per_photon` (measured when the flux model is on, otherwise the geometry's), the local-background annulus size `bg_annulus_pixels`, and the `aperture` rule. |
| `calibration` | one | What the calibration settled: the probixi version and database `schema_version`, the gain and read variance, whether the flux model is on, the detection threshold with its noise budget and tail ratio, the peak mixture (`kappa`, `prior_peak`, `var_scale`), the aperture, the camera length, and the cell in use. |

Four result tables are joined on `frame_id`, and per lattice on `crystal_id`.

| Table | Rows | Contents |
| --- | --- | --- |
| `frames` | one per file-event | Provenance (`frame_index`, `filename`, `event`, `serial`), the `indexed` flag, the peak search (`n_peaks`, `peak_resolution_nm_inv`), the per-frame fluence scale (`scale`, `scale_sigma`), and `num_reflections` over every lattice on the frame. Per-event (`/%/`) data adds `event_name` and `photon_energy_eV`. |
| `crystals` | one per accepted lattice | `lattice_index` orders the lattices on one frame. Counts (`n_indexed`, `num_reflections`), fit quality (`rmsd`, `mosaicity_deg`, `profile_radius_nm_inv`), enrichment (`enrichment`, `n_bright`, `enrich_p`), resolution (`diffraction_limit_nm_inv`), the refined cell (`cell_*`), and the reciprocal basis (`astar_*`, `bstar_*`, `cstar_*`, in nm⁻¹). |
| `reflections` | one per integrated reflection | Miller indices `h`/`k`/`l`, `intensity`, `sigma`, `peak`, `background`, position (`fs`, `ss`, `panel`), and `resolution_nm_inv`. Of the `n_pixels_total` pixels in the nominal disk, `n_pixels` were summed, `n_masked` fell on the mask or frame edge, and `n_overlap` are shared with another predicted disk. `background_model` and `background_model_var` are the noise model's background under the summed pixels and its variance ([method](method.md#73-the-model-background-under-the-disk)). |
| `peaks` | one per searched peak | Position (`fs`, `ss`, `panel`), `intensity`, `resolution_nm_inv`, `n_pixels`, and `n_masked`, the blob pixels bordering the mask or frame edge. `photons` and `background_photons` are the blob's total and background counts in photons. |

Frames that never indexed have `indexed = FALSE`, so the indexing rate is `SELECT AVG(indexed::INT) FROM frames`. A frame has one row however many lattices it yields: count lattices with `SELECT COUNT(*) FROM crystals` and indexed frames with `SELECT COUNT(*) FROM frames WHERE indexed`. `--peaks-only` fills `peaks` and leaves `crystals` and `reflections` empty.

TorchSX reads the database directly (`torchsx.train run.duckdb --spacegroup ...`).

### CXI peak export

With `--peaks-only` and a directory as `-o`, `probixi` writes one `.cxi` per input file with the raw frames externally linked, a `peaks.lst`, and a companion `<geometry>_cxi.geom`, ready for `indexamajig --peaks=cxi`. With a `.db` file as `-o`, the peaks go into the database instead.

## Indexing ambiguity

When the lattice symmetry is higher than the point group, each crystal can be indexed in more than one way. `probixi-resolve` implements the method of Brehm and Diederichs, as in CrystFEL's `ambigator`, on a DuckDB output. It clusters the crystals into the two indexing choices, reindexes those on the wrong side by transforming `h k l`, the reciprocal axes, and the cell, and records each assignment in an `ambiguity` table.

```bash
probixi-resolve -i run.duckdb -y 3 -w 321 -o resolved.duckdb
```

## Frame rendering

`render_frame` (or `--render` on the CLI) draws a frame with detected peaks as open circles and predicted reflections as crosses. The display is scaled to percentiles of the valid pixels only, so masked and dead pixels do not set the contrast.
