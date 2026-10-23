from __future__ import annotations

import click


@click.command()
@click.argument("database", type=click.Path(exists=True, dir_okay=False))
@click.argument("table", type=click.Path(exists=True, dir_okay=False))
@click.option("--sep", default="\t", help="Column separator of TABLE (default: tab).")
def main(database: str, table: str, sep: str) -> None:
    """Attach per-frame values from TABLE to the frames table of DATABASE.

    TABLE is a text table with the header ``filename event <name> ...``. Each
    further column, such as a pulse energy or a spectral width, becomes a column
    of ``frames``, matched on filename and event.
    """
    from ..io.db import attach_frame_values

    names = attach_frame_values(database, table, sep=sep)
    click.echo(f"attached {', '.join(names)} to {database}")
