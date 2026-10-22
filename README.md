# probixi - Self-Calibrating (PROB)ab(I)listic Peak Detection for Serial (X)-Ray Crystallograph(I)c Data <a href="https://github.com/ryan-odea/probixi">
<img src="https://raw.githubusercontent.com/ryan-odea/Probixi/main/docs/assets/sticker.png" align="right" height="138" alt="Probixi logo with link to website." /></a>
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![PyPI version](https://badge.fury.io/py/probixi.svg)](https://pypi.org/project/probixi) 
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/probixi)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![codecov](https://codecov.io/gh/ryan-odea/Probixi/graph/badge.svg?token=DMOVJJUWXP)](https://codecov.io/gh/ryan-odea/Probixi)
[![CUDA](https://img.shields.io/badge/CUDA-supported-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![Apple Silicon MPS](https://img.shields.io/badge/Apple%20Silicon-MPS-000000.svg?logo=apple)](https://developer.apple.com/metal/pytorch/)
[![Downloads](https://static.pepy.tech/badge/probixi)](https://pepy.tech/project/probixi)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/probixi/badge/?version=latest)](https://probixi.readthedocs.io)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`probixi` treats peakfinding as anomaly detection. Bragg signal is sparse, and the stream of frames is therefore mostly a sample of the detector's background. `probixi` learns that background at every pixel as frames arrive, whitens each frame against it, and reports the pixels the background does not explain. The peak contrast, peak prior, background blend, and detection threshold are calibrated on the data, and no parameter requires tuning. Each frame is then indexed, predicted, and integrated, with every peak weighted by its detection posterior.

![image](https://raw.githubusercontent.com/ryan-odea/Probixi/main/docs/assets/br_fel.gif)

## Installation

```bash
pip install probixi
```

On Linux with CUDA, the `gpu` extra adds the Triton kernels and nvCOMP decompression. Without it, `probixi` runs its reference implementations on the same device.

```bash
pip install "probixi[gpu]"
```

The development version:

```bash
pip install git+https://github.com/ryan-odea/probixi.git
```

## Usage

The command line goes from raw frames to integrated reflections in one call. The Python API returns lazy streams whose results stay on the GPU until collected, and any stage can feed further torch processing. `probixi` calibrates on a set of seed frames before detection, and `--gif` writes an animation of the noise model over the first frames of the run.

```bash
probixi -i files.lst -g myGeometry.geom -p myCell.cell -o stream.stream --device cuda --gif myNoiseModel.gif
```

The same run in Python:

```python
import torch

from probixi import Probixi, DataOffloader

pipeline = Probixi(
    list_file="files.lst",
    geometry_file="myGeometry.geom",
    cell_file="myCell.cell",
    device=torch.device("cuda"),
)

pipeline.noise_diagnostics("myNoiseModel.gif", stop=32)
cal = pipeline.calibrate(n_seed=1636)
print(f"kappa={cal.kappa:.2f}  prior_peak={cal.prior_peak:.4f}  "
      f"threshold={pipeline.threshold_calibration.threshold:.2f}")

with DataOffloader(
    "stream.stream",
    geometry=pipeline.geometry,
    cell=pipeline.target_cell,
    geometry_file="myGeometry.geom",
    files=pipeline.metadata.files,
) as off:
    for result in pipeline.index_stream(pipeline.frames(), batch_size=8):
        off.write(result)  # or: pipeline.index_stream(...).to_stream(off)
        for crystal in result.crystals:
            print(f"frame {result.frame_index}: "
                  f"{crystal.n_indexed}/{crystal.n_peaks} indexed (rmsd {crystal.rmsd:.4f})")
```

### DuckDB output

A `.stream` file suits `partialator`, but querying it means parsing a large text file. An output ending in `.duckdb` or `.db` is written as a [DuckDB](https://duckdb.org) database instead, which may become the default.

```bash
probixi -i files.lst -g myGeometry.geom -p myCell.cell -o run.duckdb --device cuda
```

```python
stream = pipeline.index_stream(pipeline.frames(), batch_size=8)
stream.to_db(
    "run.duckdb",
    geometry=pipeline.geometry,
    cell=pipeline.target_cell,
    geometry_file="myGeometry.geom",
    files=pipeline.metadata.files,
)
```

The database holds the run metadata (`geometry`, `panels`, `cell`, `integration`) and four result tables joined on `frame_id`: `frames`, `crystals`, `reflections`, and `peaks`. The [schema](https://probixi.readthedocs.io/en/latest/overview/io.html#database) lists their columns.

### Indexing ambiguity

When the lattice symmetry is higher than the structure's point group, each crystal can be indexed in more than one way. `probixi-resolve` clusters the crystals in a database into the two choices and reindexes those on the wrong side. It is a torch and Triton implementation of CrystFEL's `ambigator`, by Thomas White.

```bash
probixi-resolve -i run.duckdb -y 3 -w 321 -o detwinned.duckdb
```

Without `-o` the database is rewritten in place. Reindexing transforms `h k l` in `reflections` together with the reciprocal axes and cell of each crystal in `crystals`, and records every assignment in an `ambiguity` table. In Python:

```python
from probixi import Ambigator

amb = Ambigator("run.duckdb", symmetry="3", apparent="321")
result = amb.resolve()
print(f"{result.n_reindexed}/{len(result)} crystals reindexed")
amb.reindex(result, output="detwinned.duckdb")
```

### Peak export for other indexers

To use another indexer, run `probixi` as a peakfinder with the CLI's `--peaks-only` flag or the Python API's `peak_stream`. The output directory receives one CXI file per input file, a `peaks.lst`, and a geometry for `indexamajig --peaks=cxi`. The exported peaks also improve conventional indexers on strong data (see the [vignettes](https://probixi.readthedocs.io)).

```bash
probixi -i files.lst -g myGeometry.geom -o peaks/ --peaks-only --device cuda
```

```python
import torch

from probixi import Probixi, PeakOffloader

pipeline = Probixi(
    list_file="files.lst",
    geometry_file="myGeometry.geom",
    device=torch.device("cuda"),
)
pipeline.calibrate(n_seed=1636)

peaks = pipeline.peak_stream(pipeline.frames(), estimate_scale=False)
with PeakOffloader(
    "peaks/",
    geometry_file="myGeometry.geom",
    files=pipeline.metadata.files,
) as off:
    for result in peaks:
        if len(result):  # frames without peaks are not exported
            off.write(result)
```

## Dependencies

- python >= 3.10
  - click
  - deflate
  - duckdb
  - h5py
  - hdf5plugin
  - matplotlib
  - numpy
  - pillow
  - torch
- optional (`gpu`): triton >= 3.5, nvidia-libnvcomp-cu12

## Contributing

Report bugs and request features on the [issue tracker](https://github.com/ryan-odea/probixi/issues). A pull request should meet the following conventions:

1. Code is formatted with `isort`, then `black`, and passes `ruff check`. The same checks run on every PR.
2. Only user-facing objects carry docstrings, in [`numpy` style](https://numpydoc.readthedocs.io/en/latest/format.html).
3. Comments are single lines that state what the code does not show: a reason, a unit, a convention, or a source. The explanation of a bug fix belongs in the PR, not in the code.
