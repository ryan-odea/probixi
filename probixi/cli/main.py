from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Literal, Optional, cast

import click

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
_PROGRESS_INTERVAL_S = 60.0


def _pct(num: int, denom: int) -> str:
    return f"{(100.0 * num / denom) if denom else 0.0:.1f}%"


def _resolve_cli_devices(
    device: Optional[str], devices: Optional[str], gpus: Optional[int]
) -> Optional[list]:
    # Device list from --device/--devices/--gpus; None keeps the single-device path
    import torch

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


def _run_multi_gpu(device_list: list, opts: dict, refine, integrate) -> None:
    from probixi.indexer import SeedConfig
    from probixi.tools.multigpu import run_data_parallel

    kw = dict(opts)
    if kw.pop("peaks_only") or kw.pop("render") or kw.pop("gif"):
        raise click.UsageError(
            "--devices/--gpus supports the indexing path only "
            "(not --peaks-only, --render, or --gif)"
        )
    for name in (
        "device",
        "devices",
        "gpus",
        "render_out",
        "no_refine_cell",
        "aperture",
    ):
        kw.pop(name)
    if kw["cell_file"] is None:
        raise click.UsageError("a unit cell (-p/--cell) is required for multi-GPU")
    if kw["output"] is None:
        raise click.UsageError("-o/--output is required for multi-GPU indexing")
    if kw.pop("force_all"):
        kw["frame_screen_frac"] = 0.0
    run_data_parallel(
        kw.pop("list_file"),
        kw.pop("geometry_file"),
        kw.pop("cell_file"),
        kw.pop("output"),
        devices=device_list,
        seed=SeedConfig(max_lattices=kw.pop("max_lattices")),
        refine=refine,
        integrate=integrate,
        **kw,
    )


@click.command()
@click.option(
    "-i",
    "--input",
    "list_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CrystFEL list file of HDF5, CXI, or CBF (.cbf, .cbf.gz, .cbf.bz2) "
    "inputs, optionally as 'path //event' or 'path tag-N//'.",
)
@click.option(
    "-g",
    "--geometry",
    "geometry_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CrystFEL geometry file.",
)
@click.option(
    "-p",
    "--cell",
    "cell_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="CrystFEL cell file; approximate is enough. Required unless "
    "--peaks-only or --render.",
)
@click.option(
    "--mask",
    "mask_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="HDF5 file holding the bad-pixel mask, such as an Eiger master file; "
    "overrides the geometry's mask_file. Reads the geometry's 'mask' dataset, "
    "else the NeXus /entry/instrument/detector/[detectorSpecific/]pixel_mask. "
    "An unreadable mask is an error.",
)
@click.option(
    "-o",
    "--output",
    "output",
    required=False,
    default=None,
    type=click.Path(writable=True),
    help="Output; the extension sets the format: .stream (CrystFEL stream) or "
    ".db/.duckdb (DuckDB). Under --peaks-only, a directory of CXI peak files or "
    "a .db/.duckdb file, which CBF input requires. Optional with --render alone.",
)
@click.option(
    "--peaks-only",
    is_flag=True,
    help="Peakfinding only. -o is a directory of CXI peak files for "
    "'indexamajig --peaks=cxi' (HDF5 input), or a DuckDB file. No cell needed.",
)
@click.option(
    "--gif",
    "gif",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Write a GIF of the noise model over the first --seed-frames frames.",
)
@click.option("--start", type=int, default=None, help="First frame index (inclusive).")
@click.option("--stop", type=int, default=None, help="Last frame index (exclusive).")
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
    help="Lattices to search per frame; indexed peaks are removed between passes.",
)
@click.option(
    "--recalibrate-every",
    type=click.IntRange(min=0),
    default=None,
    help="Rebuild the calibration every N input frames and freeze the statistics "
    "in between; 0 freezes the initial calibration.",
)
@click.option(
    "--device",
    default=None,
    help="Torch device, such as cuda, cuda:1, mps, or cpu. Default auto: CUDA, "
    "then MPS, then CPU.",
)
@click.option(
    "--devices",
    default=None,
    help="Comma-separated devices for multi-GPU indexing, such as "
    "'cuda:0,cuda:1'; the run is split into blocks and the outputs merged in "
    "order. Indexing only.",
)
@click.option(
    "--gpus",
    type=int,
    default=None,
    help="Multi-GPU indexing on the first N CUDA devices "
    "(--devices cuda:0,...,cuda:N-1).",
)
@click.option(
    "--threads-per-worker",
    type=int,
    default=None,
    help="CPU threads per multi-GPU worker (default: cpu_count // n_workers).",
)
@click.option(
    "--noise-mode",
    type=click.Choice(["online", "per_frame"]),
    default="online",
    show_default=True,
    help="online keeps updating the background after calibration; per_frame "
    "rebuilds it for each frame.",
)
@click.option(
    "--warmup-frames",
    type=int,
    default=16,
    show_default=True,
    help="Frames seen before zero-variance pixels are masked as dead.",
)
@click.option(
    "--noise-decay",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True),
    default=1.0,
    show_default=True,
    help="Per-frame weight of the running background; 1 averages the run, "
    "d < 1 remembers about 1/d frames.",
)
@click.option(
    "--seed-frames",
    type=int,
    default=32,
    show_default=True,
    help="Frames drawn at random to calibrate the noise model and threshold.",
)
@click.option(
    "--random-seed",
    type=int,
    default=1988,
    show_default=True,
    help="Seed for the draw of calibration and aperture-training frames.",
)
@click.option(
    "--target-noise-peaks",
    type=float,
    default=None,
    help="Noise budget: spurious detections allowed on a signal-free frame; "
    "default 1.25 per million valid pixels. Lower is stricter.",
)
@click.option(
    "--flux-variance/--no-flux-variance",
    default=None,
    help="Whiten each pixel against the variance predicted at its frame's own "
    "level. Default: decided at calibration, on when the background changes "
    "between shots beyond photon noise.",
)
@click.option(
    "--flux-var-floor",
    type=float,
    default=0.15,
    show_default=True,
    help="Lower bound on the flux-scaled variance, as a fraction of the running "
    "variance.",
)
@click.option(
    "--panel",
    default="0",
    show_default=True,
    help="Panel name for peaks outside every geometry panel.",
)
@click.option(
    "--enrich-gate",
    is_flag=True,
    help="Drop lattices whose predicted spots land on signal no more often "
    "than chance.",
)
@click.option(
    "--enrich-alpha",
    type=float,
    default=1e-3,
    show_default=True,
    help="Largest chance probability a lattice may have under --enrich-gate.",
)
@click.option(
    "--render",
    "render",
    multiple=True,
    metavar="FRAME",
    help="Draw a frame with its peaks and predicted reflections: an index, "
    "'filename//event', or 'filename tag-N//'. Repeatable; runs after "
    "calibration, before indexing.",
)
@click.option(
    "--render-out",
    "render_out",
    default=None,
    type=click.Path(),
    help="Image file for a single --render, else a directory.",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress the progress line and the citation.",
)
@click.option(
    "--force-all",
    is_flag=True,
    help="Disable the blank-shot screen so every frame trains the background.",
)
@click.option(
    "--no-refine-cell",
    is_flag=True,
    help="Hold every crystal at the target cell and refine orientation only.",
)
@click.option(
    "--aperture",
    type=click.Choice(["snr", "flux", "auto"]),
    default="auto",
    show_default=True,
    help="Integration disk: snr maximises background-limited signal-to-noise, "
    "flux reaches the 2% profile radius, and auto keeps snr unless its captured "
    "flux fraction varies between crystals by more than the flux disk's extra "
    "background noise costs.",
)
@click.option(
    "--cell-calibrate/--no-cell-calibrate",
    default=True,
    show_default=True,
    help="Re-centre the target cell on the median refined cell of the "
    "calibration lattices, then twice more during the run, and re-fit the "
    "camera length.",
)
@click.option(
    "--cell-calibrate-after",
    type=int,
    default=200,
    show_default=True,
    help="Accepted lattices pooled before each re-centring during the run.",
)
def main(
    list_file: str,
    geometry_file: str,
    cell_file: Optional[str],
    mask_file: Optional[str],
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
    noise_decay: float,
    seed_frames: int,
    random_seed: int,
    target_noise_peaks: Optional[float],
    flux_variance: Optional[bool],
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
    """Find peaks, index, and integrate every frame of a CrystFEL list file.

    The -o extension selects a CrystFEL stream or a DuckDB database.

    --peaks-only writes peaks without indexing.
    """
    opts = dict(locals())
    if not peaks_only and not render and cell_file is None:
        raise click.UsageError(
            "a unit cell (-p/--cell) is required unless --peaks-only or --render"
        )
    if output is None and not render:
        raise click.UsageError("-o/--output is required unless only --render is used")

    from probixi.indexer import IntegrateConfig, RefineConfig, SeedConfig
    from probixi.io import (
        CbfInfo,
        DataOffloader,
        DuckDBOffloader,
        PeakOffloader,
        is_duckdb_path,
    )
    from probixi.probixi import Probixi, citation

    if not quiet:
        click.echo("If you use probixi, please cite:")
        citation()  # the current entry from GitHub, else the bundled one
        click.echo()

    device_list = _resolve_cli_devices(device, devices, gpus)
    refine_cfg = RefineConfig(cell=not no_refine_cell)
    integrate_cfg = IntegrateConfig(aperture=aperture)
    if device_list is not None and len(device_list) > 1:
        _run_multi_gpu(device_list, opts, refine_cfg, integrate_cfg)
        return

    dev = device_list[0] if device_list else None
    probixi = Probixi(
        list_file=list_file,
        geometry_file=geometry_file,
        cell_file=cell_file,
        mask_file=mask_file,
        noise_mode=cast("Literal['online', 'per_frame']", noise_mode),
        warmup_frames=warmup_frames,
        noise_decay=noise_decay,
        flux_variance=flux_variance,
        flux_var_floor=flux_var_floor,
        device=dev,
        random_seed=random_seed,
        seed=SeedConfig(max_lattices=max_lattices),
        refine=refine_cfg,
        integrate=integrate_cfg,
        cell_calibrate=cell_calibrate,
        cell_calibrate_after=cell_calibrate_after,
        frame_screen_frac=0.0 if force_all else Probixi.frame_screen_frac,
    )

    meta = probixi.metadata
    if (
        peaks_only
        and not is_duckdb_path(output)
        and any(isinstance(info, CbfInfo) for info in meta.files.values())
    ):
        raise click.UsageError(
            "the CXI peak export links the HDF5 image stack and cannot "
            "reference CBF input; write a .duckdb output instead"
        )
    if not quiet:
        click.echo(f"Loaded {meta.n_frames} frames from {meta.n_files} file(s).")
    if force_all:
        click.echo(
            "WARNING: --force-all: blank/no-beam shots train the background "
            "model, which biases the per-pixel mean and inflates the variance "
            "for every other frame."
        )

    if gif:
        probixi.noise_diagnostics(gif, stop=seed_frames, batch_size=max(1, batch_size))
        if not quiet:
            click.echo(f"Wrote noise diagnostic GIF to {gif}")

    cal = probixi.calibrate(
        n_seed=seed_frames,
        target_noise_peaks="auto" if target_noise_peaks is None else target_noise_peaks,
    )
    if not quiet:
        tc = probixi.threshold_calibration
        msg = (
            f"Calibrated on {probixi.n_seed_frames} frames: kappa={cal.kappa:.2f} "
            f"prior_peak={cal.prior_peak:.4f} var_scale={cal.var_scale:.3f}"
        )
        if tc is not None:
            msg += f" mf_threshold={tc.threshold:.2f} (noise target {tc.target_noise_peaks:g})"
            if tc.n_quiet_frames == 0:  # no signal-free seed frame: mirrored tail
                msg += " (noise from the mirrored tail)"
        msg += f" flux_variance={'on' if probixi.flux_variance else 'off'}"
        if probixi.background_fluctuation is not None:
            msg += f" (background fluctuation {probixi.background_fluctuation:.2f})"
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
        choice = probixi.aperture_choice
        if choice is not None:
            ap, d = choice
            msg += f" aperture={ap}"
            if "f_mean" in d:
                msg += f" (f={d['f_mean']:.2f} n={d['n']}"
                if "crystal_spread" in d:
                    msg += " trend={:.1%} spread={:.1%} z={:.1f}".format(
                        d["residual_trend"], d["crystal_spread"], d["spread_z"]
                    )
                if "noise_ratio" in d:
                    msg += f" noise={d['noise_ratio']:.2f}"
                msg += ")"
            msg += f": {d['reason']}"
        click.echo(msg)
        pr = probixi.predicted_radii
        if pr and probixi.indexer is not None and probixi.indexer._radius_by_q:
            edges = probixi.indexer._radius_by_q[0].tolist()
            click.echo(
                "Disk radius at predicted positions (px, from d in A): "
                + " ".join(
                    f"{'inf' if e <= 0 else f'{1 / e:.2f}'}:{r:.1f}"
                    for e, r in zip(edges[:-1], pr)
                )
            )
        cl = probixi.camera_length_calibration
        if cl is not None:
            click.echo(
                f"Camera length fitted on {cl.n_lattices} lattices: "
                f"{100 * cl.eps:+.2f}% +- {100 * cl.sigma:.2f}% "
                + (
                    f"-> {1e3 * cl.distance:.3f} mm (was {1e3 * cl.previous:.3f})"
                    if cl.applied
                    else f"(kept {1e3 * cl.previous:.3f} mm)"
                )
            )

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
        offload_kwargs["calibration"] = lambda: probixi.calibration_summary
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
                        f"exceptionally dark, withheld from the background model"
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
