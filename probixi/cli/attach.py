from __future__ import annotations

import click


@click.command()
@click.argument("database", type=click.Path(exists=True, dir_okay=False))
@click.argument("table", type=click.Path(exists=True, dir_okay=False))
@click.option("--sep", default="\t", help="Column separator of TABLE (default: tab).")
def main(database: str, table: str, sep: str) -> None:
    """Attach per-frame values to a probixi database.

    TABLE is a text table with a header ``filename event <name> ...``; every
    further column (a pulse energy, a spectral width, ...) becomes a column of
    the database's ``frames`` table, matched on filename and event, so a merging
    program can read it per image.
    """
    from ..io.db import attach_frame_values

    names = attach_frame_values(database, table, sep=sep)
    click.echo(f"attached {', '.join(names)} to {database}")
