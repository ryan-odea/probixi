# probixi - Self-Calibrating (PROB)ab(I)listic Peak Detection for Serial (X)-Ray Crystallograph(I)c Data

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![PyPI version](https://badge.fury.io/py/probixi.svg)](https://pypi.org/project/probixi) 
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/probixi)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-supported-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![Apple Silicon MPS](https://img.shields.io/badge/Apple%20Silicon-MPS-000000.svg?logo=apple)](https://developer.apple.com/metal/pytorch/)
[![Downloads](https://static.pepy.tech/badge/probixi)](https://pepy.tech/project/probixi)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/probixi/badge/?version=latest)](https://probixi.readthedocs.io)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

TODO: add codecov


`probixi` proposes that bragg peaks can be found/recovered from a detector image by observing the background noise distributional shape over time, per pixel, and collecting peak candidates from an outlier set. Since this noise model is determined in an unsupervised fashion, the user does not need to tune hyperparameters for finding peaks. We are still testing robustness to different types of data collection (synchrotron, FEL) and random fluence changes, results will be included in this README as they arrive.

## Installing the Package

You can install via Pypi with pip:

```bash
pip install probixi
```

Or the latest development version with

```bash
pip install git+https://github.com/ryan-odea/probixi.git
```

## Using `Probixi`

`probixi` can be interacted with either via the command line interface, or through the python API. In it's current implementation, via python, the `Probixi` API returns iterables, which remain on a GPU tensor via pytorch up until collection - meaning that you can further pass information for any downstream processing. Through the CLI, this is currently a one-stop-shop for peakfinding and indexing. **This may change in the future**

`probixi` also has a 'burn-in' phase, where the noise model reaches some stable point, this can be further interrogated with a handy gif.

Via the CLI:

```bash
probixi -i files.lst -g myGeometry.geom -p myCell.cell -o stream.stream --device cuda --gif myNoiseModel.gif
```

Or with python:
#TODO a good example goes here

## Dependencies

- python >= 3.9
  - click
  - h5py
  - hdf5plugin
  - numpy
  - torch
  - matplotlib
  - pillow

## Contributing

There are many different ways to contribute to further development of this tool. If you experience a bug or would like an additional feature, please open up a [ticket](https://github.com/ryan-odea/probixi/issues). 

If you would like to contribute actively by merging code, please open a PR with the following:

1. Code is formatted with `isort`, then `black`, followed by a `ruff --check`. This will initiate on PR, so it might be best to check beforehand.
2. Docstrings are minimally on user-facing functions in [`numpy` style](https://numpydoc.readthedocs.io/en/latest/format.html). 
3. Comments, or some explanation (in PR) for the additions, limited to the scope of the project. If fixing a bug, comments should be included in the PR rather than the code itself.

