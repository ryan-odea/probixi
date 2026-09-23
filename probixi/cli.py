from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Literal, Optional, cast

import click
import torch

from probixi.indexer import IntegrateConfig, RefineConfig, SeedConfig
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
    explicit = bool(device) and device.strip().lower() != "auto"
    picked = [f for f in (explicit, bool(devices), gpus) if f]
    if len(picked) > 1:
        raise click.UsageError("pass only one of --device / --devices / --gpus")
    if devices:
        return [torch.device(d.strip()) for d in devices.split(",") if d.strip()]
    if gpus is not None:
        if gpus < 1:
            raise click.UsageError("--gpus must be >= 1")
        return [torch.device(f"cuda:{i}") for i in range(gpus)]
    if device and device.strip().lower() != "auto":
        return [torch.device(device)]
    return None


def format_cell_calibration(cc) -> str:
    c = cc.cell
    if not cc.applied:
        return (
            f"Cell NOT re-centred: median of {cc.n_lattices} lattices "
            f"a={c.a:.3f} b={c.b:.3f} c={c.c:.3f} lies outside the cell file's match window"
        )
    return (
        f"Cell re-centred on {cc.n_lattices} lattices: "
        f"a={c.a:.3f} b={c.b:.3f} c={c.c:.3f} "
        f"al={math.degrees(c.alpha):.2f} be={math.degrees(c.beta):.2f} "
        f"ga={math.degrees(c.gamma):.2f} "
        f"(target moved {100 * cc.edge_shift:.2f}% on edges, "
        f"{math.degrees(cc.angle_shift):.3f} deg on angles)"
    )


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
        random_seed=kw["random_seed"],
        target_noise_peaks=kw["target_noise_peaks"],
        noise_mode=kw["noise_mode"],
        warmup_frames=kw["warmup_frames"],
        flux_variance=kw["flux_variance"],
        flux_var_floor=kw["flux_var_floor"],
        panel=kw["panel"],
        enrich_gate=kw["enrich_gate"],
        enrich_alpha=kw["enrich_alpha"],
        threads_per_worker=kw["threads_per_worker"],
        quiet=kw["quiet"],
        frame_screen_frac=0.0 if kw["force_all"] else 0.1,
        seed=SeedConfig(max_lattices=kw["max_lattices"]),
        refine=kw["refine"],
        integrate=kw["integrate"],
        recalibrate_every=kw["recalibrate_every"],
        cell_calibrate=kw["cell_calibrate"],
        cell_calibrate_after=kw["cell_calibrate_after"],
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
@click.option(
    "--max-lattices",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Lattices to search per frame, peeling indexed peaks between passes.",
)
@click.option(
    "--recalibrate-every",
    type=click.IntRange(min=0),
    default=None,
    help="Input frames between fresh calibrations; 0 freezes the initial estimate.",
)
@click.option(
    "--device",
    default=None,
    help="Torch device, or 'auto' (the default)",
)
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
    "--random-seed",
    type=int,
    default=1988,
    show_default=True,
    help="Seed for the random draw of calibration and radii-training frames.",
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
    "--flux-variance",
    is_flag=True,
    help="Fit a photon-transfer curve during calibration and whiten each pixel "
    "against its own Poisson noise, instead of using a frozen variance floor. "
    "Opt-in; intended for XFEL/SFX or jet-intensity-variable data.",
)
@click.option(
    "--flux-var-floor",
    type=float,
    default=0.15,
    show_default=True,
    help="Variance floor as a fraction of the calibrated per-pixel variance, "
    "applied under --flux-variance.",
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
@click.option(
    "--force-all",
    is_flag=True,
    help="Disable the blank-shot screen and index every frame, however dark",
)
@click.option(
    "--no-refine-cell",
    is_flag=True,
    help="Keep every crystal at the target cell instead of refining the cell "
    "(and orientation) per crystal against its indexed peaks.",
)
@click.option(
    "--aperture",
    type=click.Choice(["snr", "flux"]),
    default="snr",
    show_default=True,
    help="Size the learned integration disk for background-limited "
    "signal-to-noise (smaller on narrow spots) or for total flux (2% profile radius).",
)
@click.option(
    "--cell-calibrate/--no-cell-calibrate",
    default=True,
    show_default=True,
    help="Re-centre the target cell on the median refined cell of the calibration "
    "frames' lattices and then of the first accepted lattices (twice), so the cell "
    "file only has to bootstrap; absorbs a cell/camera-length inconsistency.",
)
@click.option(
    "--cell-calibrate-after",
    type=int,
    default=200,
    show_default=True,
    help="Accepted lattices pooled before each re-centring under --cell-calibrate.",
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
    max_lattices: int,
    recalibrate_every: Optional[int],
    device: Optional[str],
    devices: Optional[str],
    gpus: Optional[int],
    threads_per_worker: Optional[int],
    noise_mode: str,
    warmup_frames: int,
    seed_frames: int,
    random_seed: int,
    target_noise_peaks: float,
    flux_variance: bool,
    flux_var_floor: float,
    panel: str,
    enrich_gate: bool,
    enrich_alpha: float,
    render: tuple,
    render_out: Optional[str],
    quiet: bool,
    force_all: bool,
    no_refine_cell: bool,
    aperture: str,
    cell_calibrate: bool,
    cell_calibrate_after: int,
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
    refine_cfg = RefineConfig(cell=not no_refine_cell)
    integrate_cfg = IntegrateConfig(aperture=aperture)
    if device_list is not None and len(device_list) > 1:
        _run_multi_gpu(
            device_list,
            refine=refine_cfg,
            integrate=integrate_cfg,
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
            max_lattices=max_lattices,
            recalibrate_every=recalibrate_every,
            seed_frames=seed_frames,
            random_seed=random_seed,
            target_noise_peaks=target_noise_peaks,
            noise_mode=noise_mode,
            warmup_frames=warmup_frames,
            flux_variance=flux_variance,
            flux_var_floor=flux_var_floor,
            panel=panel,
            enrich_gate=enrich_gate,
            enrich_alpha=enrich_alpha,
            threads_per_worker=threads_per_worker,
            quiet=quiet,
            cell_calibrate=cell_calibrate,
            cell_calibrate_after=cell_calibrate_after,
        )
        return

    dev = device_list[0] if device_list else None
    probixi = Probixi(
        list_file=list_file,
        geometry_file=geometry_file,
        cell_file=cell_file,
        noise_mode=cast("Literal['online', 'per_frame']", noise_mode),
        warmup_frames=warmup_frames,
        flux_variance=flux_variance,
        flux_var_floor=flux_var_floor,
        device=dev,
        random_seed=random_seed,
        seed=SeedConfig(max_lattices=max_lattices),
        refine=refine_cfg,
        integrate=integrate_cfg,
        cell_calibrate=cell_calibrate,
        cell_calibrate_after=cell_calibrate_after,
    )

    meta = probixi.metadata
    if not quiet:
        click.echo(f"Loaded {meta.n_frames} frames from {meta.n_files} file(s).")
    if force_all:
        click.echo(
            "WARNING: --force-all: indexing every frame including blank/no-beam "
            "shots. Those frames train the background model, which biases the "
            "per-pixel mean and inflates the variance for every other frame."
        )

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
        floor = probixi.blank_frame_floor
        if floor is not None:
            msg += f" blank_floor={floor:.3g}"
        if probixi.shadow_fraction > 0:
            msg += f" shadow={100 * probixi.shadow_fraction:.1f}%"
        radii = probixi.integration_radii
        if radii is not None:
            msg += " radii=({:.1f}, {:.1f}, {:.1f})px".format(*radii)
        else:
            msg += " radii=fallback"
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

    stream = probixi.index_frame_stream(
        frames,
        batch_size=batch_size,
        start_index=start or 0,
        recalibrate_every=recalibrate_every,
        enrich_alpha=enrich_alpha if enrich_gate else None,
    )
    stats = stream.stats
    last_log = time.monotonic() - _PROGRESS_INTERVAL_S
    offload_kwargs: dict[str, Any] = dict(
        geometry=probixi.geometry,
        cell=probixi.target_cell,
        geometry_file=geometry_file,
        files=meta.files,
        panel=panel,
        integration=probixi.integration_recipe,
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
        n_screened = 0
        n_cellcal = 0
        for result in stream:
            off.write(result)
            n += bool(result.crystals)
            if not quiet:
                while n_cellcal < len(probixi.cell_calibrations):
                    click.echo(
                        format_cell_calibration(probixi.cell_calibrations[n_cellcal])
                    )
                    n_cellcal += 1
                # screened_frames grows as the stream consumes frames
                while n_screened < len(probixi.screened_frames):
                    idx = probixi.screened_frames[n_screened]
                    n_screened += 1
                    click.echo(
                        f"  Frame {probixi.frame_source(idx)} seems "
                        f"exceptionally dark, not marking for indexing"
                    )
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
    if not quiet and probixi.screened_frames:
        click.echo(
            f"Blank-shot screen held back {len(probixi.screened_frames)} frame(s) "
            f"({_pct(len(probixi.screened_frames), stats.frames)}) from the "
            f"background model; --force-all to include them"
        )
    click.echo(f"Wrote {n} indexed frame(s), {stats.crystals} crystals to {output}")


if __name__ == "__main__":
    main()
