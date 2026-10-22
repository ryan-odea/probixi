from __future__ import annotations

import inspect
import types

import click
import pytest
import torch

import probixi.cli as cli
import probixi.probixi as probixi_mod
from probixi.probixi import auto_device
from probixi.tools.multigpu import resolve_devices


def _fake_mps(available: bool):
    return types.SimpleNamespace(is_available=lambda: available)


def _fake_usable(monkeypatch, usable: bool):
    # auto_device reads the probe as a module global; patching it bypasses the lru_cache
    monkeypatch.setattr(probixi_mod, "mps_is_usable", lambda: usable)


def test_auto_device_returns_a_concrete_device():
    dev = auto_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "mps", "cpu"}


def test_auto_device_prefers_cuda(monkeypatch):
    _fake_usable(monkeypatch, True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", _fake_mps(True), raising=False)
    assert auto_device().type == "cuda"


def test_auto_device_falls_back_to_mps_without_cuda(monkeypatch):
    _fake_usable(monkeypatch, True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _fake_mps(True), raising=False)
    assert auto_device().type == "mps"


def test_auto_device_falls_back_to_cpu_without_an_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _fake_mps(False), raising=False)
    assert auto_device().type == "cpu"


def test_auto_device_tolerates_a_torch_build_without_mps(monkeypatch):
    # torch.backends.__getattr__ re-imports a deleted mps, so replace the namespace
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "backends", types.SimpleNamespace())
    assert auto_device().type == "cpu"


def test_resolve_devices_uses_auto_when_no_cuda(monkeypatch):
    _fake_usable(monkeypatch, True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _fake_mps(True), raising=False)
    assert [d.type for d in resolve_devices(None)] == ["mps"]


@pytest.mark.parametrize("spec", [None, "auto", "AUTO", " auto "])
def test_cli_auto_leaves_the_single_device_path(spec):
    # None means "resolve in Probixi"; an explicit "auto" must mean the same
    assert cli._resolve_cli_devices(spec, None, None) is None


def test_cli_auto_does_not_conflict_with_multi_gpu_flags():
    got = cli._resolve_cli_devices("auto", "cuda:0,cuda:1", None)
    assert [str(d) for d in got] == ["cuda:0", "cuda:1"]


def test_cli_an_explicit_device_still_conflicts_with_multi_gpu_flags():
    with pytest.raises(click.UsageError):
        cli._resolve_cli_devices("cuda", "cuda:0,cuda:1", None)


def test_cli_named_device_is_honoured():
    assert [str(d) for d in cli._resolve_cli_devices("cpu", None, None)] == ["cpu"]


def test_auto_device_skips_mps_when_its_shaders_do_not_compile(monkeypatch):
    # Some Metal builds report MPS available but cannot compile torch.linalg shaders
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _fake_mps(True), raising=False)
    _fake_usable(monkeypatch, False)
    assert auto_device().type == "cpu"


def test_mps_is_usable_reports_false_when_the_probe_raises(monkeypatch):
    monkeypatch.setattr(
        torch.linalg, "inv", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    probixi_mod.mps_is_usable.cache_clear()
    try:
        assert probixi_mod.mps_is_usable() is False
    finally:
        probixi_mod.mps_is_usable.cache_clear()


def test_cli_multi_gpu_forwards_every_option(tmp_path, monkeypatch):
    from click.testing import CliRunner

    import probixi.tools.multigpu as mg

    sig = inspect.signature(mg.run_data_parallel)
    seen = {}
    monkeypatch.setattr(
        mg,
        "run_data_parallel",
        lambda *a, **kw: seen.update(sig.bind(*a, **kw).arguments),
    )
    for name in ("f.lst", "f.geom", "f.cell"):
        (tmp_path / name).write_text("")
    res = CliRunner().invoke(
        cli.main,
        [
            "-i",
            str(tmp_path / "f.lst"),
            "-g",
            str(tmp_path / "f.geom"),
            "-p",
            str(tmp_path / "f.cell"),
            "-o",
            str(tmp_path / "o.stream"),
            "--devices",
            "cpu,cpu",
            "--force-all",
        ],
    )
    assert res.exit_code == 0, res.output
    assert seen["frame_screen_frac"] == 0.0
