# Input/Output

`probixi` is built to slot into an existing serial-crystallography workflow, so it speaks CrystFEL. The same `.geom`, `.cell`, and `.lst` files go in, and a CrystFEL `.stream` (or a DuckDB database) comes out. This page follows the data in and back out.

> NOTE: .stream format may be deprecated in the future and left 'as-is'

## Input

`probixi` requires minimal user inputs:

1. A crystfel style geometry file from your beamline
2. Your crystal unit-cell
3. A list file containing all the files you would like indexed.


## Streaming frames

Frames are pulled off disk lazily by `iter_frames`. A background worker thread reads slices into a bounded queue, up to `prefetch` batches ahead, so (hopefully) disk reads overlap with GPU compute. For the common case of bitshuffle-LZ4-compressed, per-frame-chunked datasets, it reads the raw compressed chunks and decodes them across a thread pool. Everything is moved to your chosen device and dtype as it is handed out, one `(H, W)` frame or a `(B, H, W)` batch at a time.


## Output

`probixi` writes results three ways, each for a different purpose.

**CXI peaks (`--peaks-only`).** This exports the peak search in CrystFEL's CXI layout. With one `.cxi` per input file (the raw image stack is external-linked, so nothing is duplicated), a `peaks.lst`, and a companion `.geom` annotated so the directory is drop-in for `indexamajig --peaks=cxi`.

**CrystFEL stream (`.stream`).** The familiar per-frame, per-reflection text stream, written in CrystFEL 2.3 format so existing merging tools (`partialator`, `process_hkl`) read it directly.

**DuckDB (`.db`/`.duckdb`).** Run metadata lands in small `geometry`/`panels`/`cell` tables, every file-event becomes a row in `frames` (indexed or not, with its per-frame statistics), and the integrated `reflections` and searched `peaks` hang off each frame by a `frame_id`. That key is stable and deterministic, the first 16 hex digits of `sha1("filename//event")`, so a peaks-only pass, an indexed pass, and separate multi-GPU shards all agree on which row is which frame.

Under `--start`/`--stop` or multi-GPU, each writer backfills only the frame range it owns, so a shard never declares the rest of the dataset unindexed. The [getting-started guide](../vignettes/getting_started.md) documents the full schema.

## Seeing a frame

`render_frame` draws a frame with its overlays, detected peaks as open lime circles and indexed reflections as red crosses, scaling the display to percentiles of the valid pixels only so masked and dead pixels do not wreck the contrast. It is the quickest way to eyeball whether the peaks and the predicted lattice agree with the image.
