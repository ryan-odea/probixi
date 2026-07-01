from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Optional, cast

import click
import torch

from probixi.io import DataOffloader, PeakOffloader
from probixi.probixi import Probixi

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
_PROGRESS_INTERVAL_S = 60.0


def _pct(num: int, denom: int) -> str:
    return f"{(100.0 * num / denom) if denom else 0.0:.1f}%"


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
    type=click.Path(dir_okay=False, writable=True),
    help="Output CrystFEL-style .stream file. Optional when only --render is used.",
)
@click.option(
    "--peaks-only",
    is_flag=True,
    help="Only run peak finding and export a peaks-only stream "
    "Does not require a unit cell.",
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

    With --peaks-only, peak finding runs but indexing does not, and a
    peaks-only stream is written instead.
    """
    if not peaks_only and not render and cell_file is None:
        raise click.UsageError(
            "a unit cell (-p/--cell) is required unless --peaks-only or --render"
        )
    if output is None and not render:
        raise click.UsageError("-o/--output is required unless only --render is used")
    dev = torch.device(device) if device else None
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
        probixi.noise_diagnostics(
            gif, stop=seed_frames, batch_size=max(2, seed_frames // 8)
        )
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
        with PeakOffloader(
            output,
            geometry=probixi.geometry,
            geometry_file=geometry_file,
            files=meta.files,
            panel=panel,
        ) as off:
            n = 0
            for result in peaks:
                if len(result) == 0:
                    continue
                off.write(result)
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
    with DataOffloader(
        output,
        geometry=probixi.geometry,
        cell=probixi.target_cell,
        geometry_file=geometry_file,
        files=meta.files,
        panel=panel,
    ) as off:
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
