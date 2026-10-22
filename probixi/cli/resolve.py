from __future__ import annotations

import click


@click.command()
@click.option(
    "-i",
    "--input",
    "database",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="DuckDB database written by probixi (.duckdb/.db).",
)
@click.option(
    "-y", "--symmetry", required=True, help="Point group of the structure, e.g. '3'."
)
@click.option(
    "-w",
    "--apparent",
    required=True,
    help="Apparent point group of the lattice, e.g. '321'; with --symmetry it "
    "fixes the ambiguity operator.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Write the reindexed database here; without it the input is rewritten "
    "in place.",
)
@click.option(
    "-n", "--iterations", default=6, show_default=True, help="Refinement passes."
)
@click.option(
    "--ncorr",
    default=1000,
    show_default=True,
    help="Partner crystals correlated per crystal.",
)
@click.option("--lowres", default=None, type=float, help="Low-resolution cutoff (A).")
@click.option("--highres", default=None, type=float, help="High-resolution cutoff (A).")
@click.option(
    "--seed", default=1988, show_default=True, help="Seed for the starting split."
)
@click.option("--device", default=None, help="Torch device, or 'auto'.")
@click.option("--quiet", is_flag=True, help="Print only the summary and errors.")
def main(
    database,
    symmetry,
    apparent,
    output,
    iterations,
    ncorr,
    lowres,
    highres,
    seed,
    device,
    quiet,
):
    """Resolve the indexing ambiguity of the crystals in a probixi DuckDB run.

    Clusters the crystals into the two indexing choices that --apparent allows
    for --symmetry and reindexes those on the wrong side, following the method
    of Brehm and Diederichs as CrystFEL's ambigator (Thomas White) implements it.

        probixi-resolve -i run.duckdb -y 3 -w 321 -o detwinned.duckdb
    """
    from ..ambigator import Ambigator, _format_operator

    if device and device.strip().lower() == "auto":
        device = None
    try:
        amb = Ambigator(
            database,
            symmetry=symmetry,
            apparent=apparent,
            lowres=lowres,
            highres=highres,
            device=device,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if not quiet:
        click.echo(f"Ambiguity operator: {_format_operator(amb.operator)}")
    try:
        result = amb.resolve(iterations=iterations, ncorr=ncorr, seed=seed)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not quiet:
        for i, (flips, f, g) in enumerate(result.history, 1):
            click.echo(f"  pass {i}: {flips} flipped, mean f = {f:.4f}, g = {g:.4f}")
        if result.n_unused:
            click.echo(f"{result.n_unused} crystal(s) had no usable correlation")
    written = amb.reindex(result, output=output)
    click.echo(f"Reindexed {result.n_reindexed}/{len(result)} crystals in {written}")


if __name__ == "__main__":
    main()
