from __future__ import annotations

from typing import Literal, Optional, cast

import click
import torch

from probixi.io import DataOffloader, PeakOffloader
from probixi.probixi import Probixi


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
    required=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Output CrystFEL-style .stream file.",
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
@click.option("--warmup-frames", type=int, default=16, show_default=True)
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
@click.option("-q", "--quiet", is_flag=True, help="Suppress per-frame progress.")
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
    quiet: bool,
) -> None:
    """Run the probixi pipeline and write indexed frames to a CrystFEL stream.

    With --peaks-only, peak finding runs but indexing does not, and a
    peaks-only stream is written instead.
    """
    if not peaks_only and cell_file is None:
        raise click.UsageError(
            "a unit cell (-p/--cell) is required unless --peaks-only"
        )
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
            if not quiet:
                click.echo(
                    f"  frame {result.frame_index}: "
                    f"{result.n_indexed}/{result.n_peaks} peaks indexed "
                    f"(rmsd {result.rmsd:.4f})"
                )

    click.echo(f"Wrote {n} indexed frame(s) to {output}")


if __name__ == "__main__":
    main()
