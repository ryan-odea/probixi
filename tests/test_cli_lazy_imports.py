"""The CLI's --help and `import probixi` must not import torch or the pipeline."""

from __future__ import annotations

import subprocess
import sys

# Blocking a module in sys.modules with None makes any `import <name>` raise
# ImportError, so the check fails loudly if a top-level import creeps back in.
_BLOCK = "import sys; sys.modules.update({m: None for m in %r}); "
_HEAVY = ["torch", "numpy", "h5py", "duckdb", "probixi.probixi", "probixi.indexer"]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK % _HEAVY + code],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_help_renders_without_torch():
    r = _run(
        "from click.testing import CliRunner; from probixi.cli import main; "
        "res = CliRunner().invoke(main, ['--help']); "
        "assert res.exit_code == 0, res.output; print(res.output)"
    )
    assert r.returncode == 0, r.stderr
    assert "--aperture" in r.stdout and "Usage:" in r.stdout


def test_resolve_help_renders_without_torch():
    r = _run(
        "from click.testing import CliRunner; from probixi.ambigator_cli import main; "
        "res = CliRunner().invoke(main, ['--help']); "
        "assert res.exit_code == 0, res.output; print(res.output)"
    )
    assert r.returncode == 0, r.stderr
    assert "--apparent" in r.stdout


def test_package_import_is_lazy():
    r = _run(
        "import probixi; assert 'torch' not in sys.modules or sys.modules['torch'] is None"
    )
    assert r.returncode == 0, r.stderr


def test_public_names_still_resolve():
    # no blocking here: the attribute access must import the submodule on demand
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import probixi, sys; assert 'torch' not in sys.modules; "
            "from probixi import Probixi, IntegrateConfig, __citation__; "
            "assert 'torch' in sys.modules; assert 'Probixi' in dir(probixi)",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr
