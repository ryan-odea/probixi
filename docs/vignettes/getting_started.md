# Getting Started with `probixi`

Getting started with `probixi` is hopefully quite easy. The primary ethos of this package is to have statistics handle the work for you; therefore, all you need to provide is your list of files, detector geometry, and your unit cell.

Let's move through a basic tutorial using the CLI, or skip ahead to [Using `probixi` with Python](#using-probixi-with-python).

## `probixi` via the CLI

Going through the required options:

- -i provides the list of files (hdf5/cxi)
- -g provides the geometry
- -p provides the unit cell
- -o provides the output (and inferred format). Use `.stream` to generate a crystfel style stream or `.db` to generate a duckdb.

```bash
probixi -i files.lst -g my_geometry.geom -p my_unit.cell -o output.db --device cuda
```

And that's it! There are a few more advanced parameters provided in detail below:

| Parameter | Accepts | Description |
| --- | --- | --- |
| --batch-size | int | Frames per batched refinement pass |
| --device | str | Torch device (e.g. cuda, cpu, mps) |
| --devices | str | Comma-separated device list for multi-gpu indexing |
| --enrich-alpha | float | Max probability to accept a frame under --enrich-gate |
| --enrich-gate | NA (flag) | Drop indexed frames whose predicted spots are not backed by image signal beyond chance - better for high fluence and good diffraction |
| --gif | str/Path | Returns a gif of the noise model evolution over the seed frames |
| --gpus | int | Multi-gpu indexing across first N devices (the same as --devices cuda:0,cuda:1...) |
| --noise-mode | str | `online` (continuously updated) or `per_frame` built noise models |
| --panel | str | Fallback panel name for peaks that fall **outside** every geometry panel (inter-panel gaps, or a geometry with no named panels). Peaks inside a defined panel already carry that panel's name; this only labels the rest. Default `0`. |
| --peaks-only | NA (flag) | Returns crystfel readable .cxi peaks, or a duckdb, depending on `-o` |
| -q/--quiet | NA (flag) | Suppress logger lines |
| --render | str/int | Recall a frame and write a peaks/index overlay. Either index or 'image_filename//event' |
| --render-out | str/Path | `--render` destination |
| --seed-frames | int | Number of frames used to calibrate the noise model and detection threshold |
| --start | int | First frame index (inclusive) |
| --stop | int | Stop frame index (exclusive) |
| --target-noise-peaks | float | Auto-calibrate the peak-detection threshold so a signal-free frame yields at most this many spurious noise blobs (default 5). Lower &rarr; stricter/cleaner peaks; higher &rarr; more sensitive. Acts as an override |
| --threads-per-worker | int | Threads per GPU worker |
| --warmup-frames | int | Number of frames observed before learned dead-pixel mask is built |

### Recovering Results

Given your output options, a `.stream` or `.db` file has been generated. The stream is analagous to any other crystfel based stream.

> NOTE: As this package evolves, the stream option may be dropped or deprecated in favor of database or straight-to-`.hkl` methods. The crystfel stream option exists currently to compare with the method library crystfel contains.

#### Database

Output to a database allows for faster parsing to other libraries, as well as semi-archival storage of all data used in the process. The `.db` (or `.duckdb`) output is a [DuckDB](https://duckdb.org/) file written by `IndexStream.to_db` (via `DuckDBOffloader`), and it is a relational alternative to the CrystFEL `.stream`: the run's metadata, every frame's per-frame statistics, and the integrated reflections and searched peaks all live in one queryable file.

It is organized into six tables. Three small **metadata** tables are written once, up front:

| Table | Rows | What it stores |
| --- | --- | --- |
| `geometry` | one | The resolved detector geometry: `beam_center_row`/`beam_center_col`, `clen`, `pixel_size`, `wavelength`, `photon_energy_eV`, `adu_per_photon`, `n_panels`, and the full text of the geometry file in `geometry_file`. |
| `panels` | one per panel | Each panel's `name` and fast/slow-scan pixel bounds (`min_fs`, `max_fs`, `min_ss`, `max_ss`). |
| `cell` | one | The target unit cell: edges `a_A`/`b_A`/`c_A`, angles `alpha_deg`/`beta_deg`/`gamma_deg`, `volume_A3`, and the symmetry labels `lattice_type`, `centering`, `unique_axis`. |

The remaining three **per-frame** tables hold the results, linked by `frame_id` — a 16-character SHA-1 hash of `filename//event` that uniquely identifies each frame:

| Table | Rows | What it stores |
| --- | --- | --- |
| `frames` | one per file-event | The row for every frame, keyed by `frame_id` (primary key). Provenance (`frame_index`, `filename`, `event`, `serial`) and an `indexed` boolean, plus per-frame statistics when indexed: peak/reflection counts (`n_peaks`, `n_indexed`, `num_reflections`), fit quality (`rmsd`, `mosaicity_deg`, `profile_radius_nm_inv`), scale (`scale`, `scale_sigma`), enrichment gate values (`enrichment`, `n_bright`, `enrich_p`), resolution (`diffraction_limit_nm_inv`, `peak_resolution_nm_inv`), the recovered cell (`cell_*`), and the reciprocal basis vectors (`astar_*`, `bstar_*`, `cstar_*`, in nm⁻¹). |
| `reflections` | one per integrated reflection | The indexed, box-integrated reflections keyed by `frame_id`: Miller indices `h`/`k`/`l`, `intensity`, `sigma`, `peak`, `background`, detector position (`fs`, `ss`, `panel`), and `resolution_nm_inv`. |
| `peaks` | one per searched peak | The peaks found by the peak-search keyed by `frame_id`: detector position (`fs`, `ss`, `panel`), `intensity`, and `resolution_nm_inv`. |

A few things worth knowing when querying:

- Frames that never indexed still get a `frames` row with `indexed = FALSE` and null statistics, so the indexed rate is simply `SELECT AVG(indexed::INT) FROM frames`. (Under `--start`/`--stop` or multi-GPU runs, each writer only backfills the frame range it owns.)
- **`--peaks-only` populates `peaks`, not `reflections`.** Those frames are written with `indexed = FALSE` and their `n_peaks`/`peak_resolution_nm_inv` set; the `reflections` table stays empty.

## Using `probixi` with Python

The CLI is a thin wrapper over the `probixi.Probixi` pipeline. Getting an indexed stream out takes three steps: construct, calibrate, drain:

```python
from probixi import Probixi

pipe = Probixi(
    list_file="files.lst",
    geometry_file="my_geometry.geom",
    cell_file="my_unit.cell",
    device="cuda",
)

# Observe seed frames to learn the noise model and detection threshold.
pipe.calibrate(n_seed=32, target_noise_peaks=5.0)

# Open a lazy stream of solutions and drain it into a DuckDB database
# (the schema described above) or a CrystFEL `.stream`.
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

> `index_stream` returns a **lazy** `IndexStream`: no frames are read and nothing is indexed until you drain it. A final `to_db`, `to_stream`, `collect`, or iteration pulls frames through the pipeline one batch at a time. Because it is lazy, each `IndexResult` (and the tensors it carries) lives on the configured torch `device`, so you can still maintain downstream torch-torch connections, and only the writers move data off-device as they serialize. A stream is single-use, re-iterating an already-drained stream yields nothing.
