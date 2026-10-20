# Inputs and outputs

`probixi` reads CrystFEL inputs and writes a CrystFEL stream or a DuckDB database.

## Inputs

1. A CrystFEL geometry file for the detector.
2. A CrystFEL cell file. It only has to be approximate, because `probixi` re-centres it on the data.
3. A list file of the HDF5 or CXI files to process.

Any pixel mask stored with the data is read and combined with the masks `probixi` learns. The geometry's `mask` and `mask_file` say where the mask is stored. Without `mask_file`, the mask is read from the first data file.

`--mask FILE` (or `Probixi(mask_file=...)`) reads the mask from `FILE` instead. Use it when the mask lives in an HDF5 master file but the list names the data files, or when `mask_file` has moved. `probixi` reads the geometry's `mask` dataset if `FILE` has it. Otherwise it reads the NeXus `/entry/instrument/detector/pixel_mask`, or the same mask under `detectorSpecific/`. For the NeXus mask, a pixel is bad if any flag bit is set, unless the geometry sets `mask_good` or `mask_bad`. If the mask cannot be read from an explicit `--mask`, the run stops. An unreadable geometry mask only gives a warning, and the run continues without it.

## Reading frames

Frames are read lazily. A background thread reads slices into a bounded queue a few batches ahead, which overlaps disk reads with GPU compute. For bitshuffle-LZ4 datasets chunked one frame per chunk, `probixi` reads the raw compressed chunks and decompresses them on the GPU, and uncompressed frames never pass through host memory. Deflate (gzip) chunks, and LZF chunks HDF5 stored uncompressed, are read and decompressed by CPU threads and unshuffled on the GPU. Each frame or batch is moved to the chosen device and data type as it is handed out.

## Outputs

**CrystFEL stream (`.stream`).** The per-frame, per-reflection text format, readable by `partialator` and `process_hkl`. It carries only what CrystFEL reads; probixi's own quantities (the integration recipe, the model background under each disk, pixel counts, per-crystal diagnostics) are in the database. A reflection whose signal disk touches the detector mask, the frame edge or another predicted disk is left out of the stream, as is a peak bordering the mask; the database keeps them, flagged by pixel counts.

**DuckDB (`.db` or `.duckdb`).** Run metadata goes into small `geometry`, `panels`, and `cell` tables ([schema](../vignettes/getting_started.md#database)). Every file-event becomes a row in `frames`, indexed or not, and lattices, integrated reflections, and searched peaks are linked to it by `frame_id`. The key is deterministic, the first 16 hex digits of the SHA-1 of `filename//event`, so a peaks-only pass, an indexed pass, and separate multi-GPU shards agree on which row is which frame. Under `--start`/`--stop` or multi-GPU runs, each writer fills only the frame range it owns.

**Peaks only (`--peaks-only`).** With a directory as `-o`, `probixi` writes one `.cxi` per input file with the raw frames externally linked, a `peaks.lst`, and a companion `<geometry>_cxi.geom`, ready for `indexamajig --peaks=cxi`. With a `.db` file as `-o`, the peaks go into the database instead.

> NOTE: The stream output may be deprecated in favour of the database.

## Resolving the indexing ambiguity

When the lattice symmetry is higher than the point group, each crystal can be indexed in more than one way. `probixi-resolve` implements the Brehm and Diederichs method, as in CrystFEL's `ambigator`, on a DuckDB output. It clusters the crystals into the two indexing choices and reindexes those on the wrong side, transforming `h k l`, the reciprocal axes, and the cell, and records each assignment in an `ambiguity` table.

```bash
probixi-resolve -i run.duckdb -y 3 -w 321 -o resolved.duckdb
```

## Viewing a frame

`render_frame` (or `--render` on the CLI) draws a frame with detected peaks as open circles and predicted reflections as crosses. The display is scaled to percentiles of the valid pixels only, so masked and dead pixels do not set the contrast.
