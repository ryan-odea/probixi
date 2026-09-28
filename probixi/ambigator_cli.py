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
@click.option("-y", "--symmetry", required=True, help="Actual point group, e.g. '3'.")
@click.option(
    "-w",
    "--apparent",
    required=True,
    help="Apparent (lattice) point group, e.g. '321'. The ambiguity operator is "
    "derived from it and --symmetry.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Reindex a copy written here, leaving the input alone. Without it "
    "the input database is rewritten in place.",
)
@click.option(
    "-n", "--iterations", default=6, show_default=True, help="Refinement passes."
)
@click.option(
    "--ncorr", default=1000, show_default=True, help="Partners correlated per crystal."
)
@click.option("--lowres", default=None, type=float, help="Low-resolution cutoff in A.")
@click.option(
    "--highres", default=None, type=float, help="High-resolution cutoff in A."
)
@click.option(
    "--seed", default=1988, show_default=True, help="Seeds the starting split."
)
@click.option("--device", default=None, help="Torch device, or 'auto'.")
@click.option("--quiet", is_flag=True, help="Only report errors.")
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
    """Resolve the indexing ambiguity in a probixi DuckDB run.

    Clusters the crystals into the two indexing choices of --symmetry under
    --apparent, then reindexes the crystals on the wrong side. The algorithm
    is CrystFEL's ambigator, by Thomas White.

        probixi-resolve -i run.duckdb -y 3 -w 321 -o detwinned.duckdb
    """
    from .ambigator import Ambigator, _format_operator

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
