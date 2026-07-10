from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal, Optional, cast

import click
import torch

from probixi.io import DataOffloader, DuckDBOffloader, PeakOffloader, is_duckdb_path
from probixi.probixi import Probixi

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
_PROGRESS_INTERVAL_S = 60.0


def _pct(num: int, denom: int) -> str:
    return f"{(100.0 * num / denom) if denom else 0.0:.1f}%"


def _resolve_cli_devices(
    device: Optional[str], devices: Optional[str], gpus: Optional[int]
) -> Optional[list]:
    # Translate the --device / --devices / --gpus flags into a device list, or
    # None to keep the single-device path. --devices/--gpus imply multi-GPU.
    picked = [f for f in (bool(device), bool(devices), gpus) if f]
    if len(picked) > 1:
        raise click.UsageError("pass only one of --device / --devices / --gpus")
    if devices:
        return [torch.device(d.strip()) for d in devices.split(",") if d.strip()]
    if gpus is not None:
        if gpus < 1:
            raise click.UsageError("--gpus must be >= 1")
        return [torch.device(f"cuda:{i}") for i in range(gpus)]
    if device:
        return [torch.device(device)]
    return None


def _run_multi_gpu(device_list: list, **kw) -> None:
    from probixi.multigpu import run_data_parallel

    if kw["peaks_only"] or kw["render"] or kw["gif"]:
        raise click.UsageError(
            "--devices/--gpus supports the indexing path only "
            "(not --peaks-only, --render, or --gif)"
        )
    if kw["cell_file"] is None:
        raise click.UsageError("a unit cell (-p/--cell) is required for multi-GPU")
    if kw["output"] is None:
        raise click.UsageError("-o/--output is required for multi-GPU indexing")
    run_data_parallel(
        kw["list_file"],
        kw["geometry_file"],
        kw["cell_file"],
        kw["output"],
        devices=device_list,
        start=kw["start"],
        stop=kw["stop"],
        batch_size=kw["batch_size"],
        seed_frames=kw["seed_frames"],
        target_noise_peaks=kw["target_noise_peaks"],
        noise_mode=kw["noise_mode"],
        warmup_frames=kw["warmup_frames"],
        panel=kw["panel"],
        enrich_gate=kw["enrich_gate"],
        enrich_alpha=kw["enrich_alpha"],
        threads_per_worker=kw["threads_per_worker"],
        quiet=kw["quiet"],
    )


@click.command()
@click.option(
    "-i",
    "--input",
    "list_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CrystFEL list file (.lst) of HDF5 inputs.",
)
@click.option(
    "-g",
    "--geometry",
    "geometry_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CrystFEL geometry file (.geom).",
)
@click.option(
    "-p",
    "--cell",
    "cell_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="CrystFEL unit-cell file (.cell). Required unless --peaks-only.",
)
@click.option(
    "-o",
    "--output",
    "output",
    required=False,
    default=None,
    type=click.Path(writable=True),
    help="Output file. A .stream writes a CrystFEL stream; a .duckdb/.db writes "
    "a DuckDB database (frames/reflections/peaks + geometry/cell/panels tables). "
    "With --peaks-only it is instead an output directory for the CXI peak set. "
    "Optional when only --render is used.",
)
@click.option(
    "--peaks-only",
    is_flag=True,
    help="Only run peak finding and export a CXI peak set (one .cxi per input "
    "file with external-linked images, plus peaks.lst and a companion .geom) "
    "for 'indexamajig --peaks=cxi'. -o is an output directory. No cell needed.",
)
@click.option(
    "--gif",
    "gif",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Also write a noise-model diagnostic GIF over the seed frames.",
)
@click.option("--start", type=int, default=None, help="First frame index (inclusive).")
@click.option("--stop", type=int, default=None, help="Stop frame index (exclusive).")
@click.option(
    "--batch-size",
    type=int,
    default=8,
    show_default=True,
    help="Frames per batched refinement pass.",
)
@click.option("--device", default=None, help="Torch device (default: auto).")
@click.option(
    "--devices",
    default=None,
    help="Comma-separated device list for multi-GPU data-parallel indexing "
    "(e.g. 'cuda:0,cuda:1'). Splits frames into blocks across devices and merges "
    "the per-device streams. Indexing path only.",
)
@click.option(
    "--gpus",
    type=int,
    default=None,
    help="Multi-GPU data-parallel indexing across the first N CUDA devices "
    "(shorthand for --devices cuda:0,...,cuda:N-1).",
)
@click.option(
    "--threads-per-worker",
    type=int,
    default=None,
    help="Torch CPU intra-op threads per multi-GPU worker (default: "
    "cpu_count // n_workers, to avoid oversubscribing cores).",
)
@click.option(
    "--noise-mode",
    type=click.Choice(["online", "per_frame"]),
    default="online",
    show_default=True,
    help="Noise-model update mode.",
)
@click.option(
    "--warmup-frames",
    type=int,
    default=16,
    show_default=True,
    help="Frames observed before the dead-pixel mask is committed.",
)
@click.option(
    "--seed-frames",
    type=int,
    default=32,
    show_default=True,
    help="Frames used to calibrate the noise model and detection threshold.",
)
@click.option(
    "--target-noise-peaks",
    type=float,
    default=5.0,
    show_default=True,
    help="Calibrate the detection threshold so a signal-free frame "
    "yields at most this many noise blobs.",
)
@click.option(
    "--panel",
    default="0",
    show_default=True,
    help="Fallback panel name for peaks outside all geometry panels.",
)
@click.option(
    "--enrich-gate",
    is_flag=True,
    help="Drop indexed frames whose predicted spots are not backed by image signal beyond chance.",
)
@click.option(
    "--enrich-alpha",
    type=float,
    default=1e-3,
    show_default=True,
    help="Max chance probability to accept a frame under --enrich-gate.",
)
@click.option(
    "--render",
    "render",
    multiple=True,
    metavar="FRAME",
    help="Recall a frame and write a peaks/index overlay image. An absolute "
    "index or 'image_filename//event'. Repeatable; renders before any run.",
)
@click.option(
    "--render-out",
    "render_out",
    default=None,
    type=click.Path(),
    help="Render destination: an image file (single --render) or a directory.",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress progress: the periodic frames/hits/indexed rate line",
)
def main(
    list_file: str,
    geometry_file: str,
    cell_file: Optional[str],
    output: str,
    peaks_only: bool,
    gif: Optional[str],
    start: Optional[int],
    stop: Optional[int],
    batch_size: int,
    device: Optional[str],
    devices: Optional[str],
    gpus: Optional[int],
    threads_per_worker: Optional[int],
    noise_mode: str,
    warmup_frames: int,
    seed_frames: int,
    target_noise_peaks: float,
    panel: str,
    enrich_gate: bool,
    enrich_alpha: float,
    render: tuple,
    render_out: Optional[str],
    quiet: bool,
) -> None:
    """Run the probixi pipeline and write indexed frames to a CrystFEL stream.

    With --peaks-only, peak finding runs but indexing does not, and a CXI peak
    set (readable by 'indexamajig --peaks=cxi') is written to the -o directory
    instead.
    """
    if not peaks_only and not render and cell_file is None:
        raise click.UsageError(
            "a unit cell (-p/--cell) is required unless --peaks-only or --render"
        )
    if output is None and not render:
        raise click.UsageError("-o/--output is required unless only --render is used")

    device_list = _resolve_cli_devices(device, devices, gpus)
    if device_list is not None and len(device_list) > 1:
        _run_multi_gpu(
            device_list,
            list_file=list_file,
            geometry_file=geometry_file,
            cell_file=cell_file,
            output=output,
            peaks_only=peaks_only,
            gif=gif,
            render=render,
            start=start,
            stop=stop,
            batch_size=batch_size,
            seed_frames=seed_frames,
            target_noise_peaks=target_noise_peaks,
            noise_mode=noise_mode,
            warmup_frames=warmup_frames,
            panel=panel,
            enrich_gate=enrich_gate,
            enrich_alpha=enrich_alpha,
            threads_per_worker=threads_per_worker,
            quiet=quiet,
        )
        return

    dev = device_list[0] if device_list else (torch.device(device) if device else None)
    probixi = Probixi(
        list_file=list_file,
        geometry_file=geometry_file,
        cell_file=cell_file,
        noise_mode=cast("Literal['online', 'per_frame']", noise_mode),
        warmup_frames=warmup_frames,
        device=dev,
    )

    meta = probixi.metadata
    if not quiet:
        click.echo(f"Loaded {meta.n_frames} frames from {meta.n_files} file(s).")

    if gif:
        probixi.noise_diagnostics(gif, stop=seed_frames, batch_size=max(1, batch_size))
        if not quiet:
            click.echo(f"Wrote noise diagnostic GIF to {gif}")

    cal = probixi.calibrate(n_seed=seed_frames, target_noise_peaks=target_noise_peaks)
    if not quiet:
        tc = probixi.threshold_calibration
        msg = (
            f"Calibrated on {seed_frames} frames: kappa={cal.kappa:.2f} "
            f"prior_peak={cal.prior_peak:.4f} var_scale={cal.var_scale:.3f}"
        )
        if tc is not None:
            msg += f" mf_threshold={tc.threshold:.2f}"
        bmr = probixi.beamstop_min_res
        if bmr is not None:
            msg += f" beamstop_min_res={bmr:.1f}A (learned)"
        click.echo(msg)

    if render:
        out = Path(render_out) if render_out else Path(".")
        as_file = len(render) == 1 and out.suffix.lower() in _IMAGE_SUFFIXES
        if not as_file:
            out.mkdir(parents=True, exist_ok=True)
        for spec in render:
            frame_id = int(spec) if spec.lstrip("-").isdigit() else spec
            dest = out if as_file else out / f"render_{spec.replace('/', '_')}.png"
            probixi.show_frame(frame_id, path=dest)
            if not quiet:
                click.echo(f"Wrote {dest}")
        if output is None:
            return

    frames = probixi.frames(start=start, stop=stop)

    if peaks_only:
        peaks = probixi.peak_stream(
            frames, start_index=start or 0, estimate_scale=False
        )
        if is_duckdb_path(output):
            off_ctx = DuckDBOffloader(
                output,
                geometry=probixi.geometry,
                geometry_file=geometry_file,
                files=meta.files,
                frame_range=(start or 0, stop if stop is not None else meta.n_frames),
                panel=panel,
            )
        else:
            off_ctx = PeakOffloader(
                output,
                geometry_file=geometry_file,
                files=meta.files,
            )
        with off_ctx as off:
            # DuckDBOffloader records peaks via write_peaks; PeakOffloader via write
            write: Any = getattr(off, "write_peaks", None) or off.write
            n = 0
            for result in peaks:
                if len(result) == 0:
                    continue
                write(result)
                n += 1
                if not quiet:
                    click.echo(f"  frame {result.frame_index}: {len(result)} peaks")
        click.echo(f"Wrote peaks for {n} frame(s) to {output}")
        return

    stream = probixi.index_stream(frames, batch_size=batch_size, start_index=start or 0)
    if enrich_gate:
        stream = stream.enrich_gate(enrich_alpha)
    stats = stream.stats
    last_log = time.monotonic() - _PROGRESS_INTERVAL_S
    offload_kwargs: dict[str, Any] = dict(
        geometry=probixi.geometry,
        cell=probixi.target_cell,
        geometry_file=geometry_file,
        files=meta.files,
        panel=panel,
    )
    if is_duckdb_path(output):
        offloader = DuckDBOffloader
        offload_kwargs["frame_range"] = (
            start or 0,
            stop if stop is not None else meta.n_frames,
        )
    else:
        offloader = DataOffloader
    with offloader(output, **offload_kwargs) as off:
        n = 0
        for result in stream:
            off.write(result)
            n += 1
            now = time.monotonic()
            if not quiet and now - last_log >= _PROGRESS_INTERVAL_S:
                last_log = now
                click.echo(
                    f"  {stats.frames} frames | "
                    f"{stats.hits} hits ({_pct(stats.hits, stats.frames)}) | "
                    f"{n} indexed ({_pct(n, stats.frames)})"
                )

    if not quiet:
        click.echo(
            f"Completed {stats.frames} frame(s): "
            f"{stats.hits} hits ({_pct(stats.hits, stats.frames)}), "
            f"{n} indexed ({_pct(n, stats.frames)}, "
            f"{_pct(n, stats.hits)} of hits)"
        )
    click.echo(f"Wrote {n} indexed frame(s) to {output}")


if __name__ == "__main__":
    main()
