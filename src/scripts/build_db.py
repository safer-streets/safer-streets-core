"""
Build the production DuckDB database in one reproducible pass.

Pipeline stages:
  1. extract.to_database   crime_data table (geometry, BNG)
  2. ons_boundaries.load_all   boundary tables (pfa, lad, msoa, lsoa, oa)
  3. transforms.build_all   H3 count tables + geo lookup tables

The pipeline writes to a ``<name>.staging.db`` file and only promotes it to the live
database with an atomic ``os.replace`` once every stage has succeeded. Read-only
consumers therefore always see a complete database  either the old one or the new
one, never a half-built file.

The live database filename is read from the SAFER_STREETS_DATABASE environment
variable (.env); pass --db-path to override.
"""

import os
from pathlib import Path

import typer

from safer_streets_core import transforms
from safer_streets_core.database import duckdb_connector
from safer_streets_core.utils import database_path
from scripts import extract, ons_boundaries

app = typer.Typer(help="Build the production crime + boundaries + H3 DuckDB database.")


@app.command()
def build(
    db_path: Path | None = None,
    resolutions: list[int] = [8, 9, 10],  # noqa: B006
    layers: list[str] = ["all"],  # noqa: B006
    force_download: bool = False,
    replace: bool = True,
) -> None:
    """Run the full pipeline into a staging file, then atomically swap it into place.

    With --replace (default) the staging database is rebuilt from scratch. With
    --no-replace an existing staging database is reused and H3 tables/views that are
    already present are kept (CREATE ... IF NOT EXISTS), letting an interrupted build resume.
    """
    db_path = db_path or database_path()
    staging = db_path.with_suffix(".staging.db")
    if replace:
        staging.unlink(missing_ok=True)

    print(f"\n=== Building {db_path} (staging: {staging}) ===\n")

    print("[1/3] Extracting crime data…")
    extract.to_database(db_path=staging, force_download=force_download)

    print("\n[2/3] Downloading ONS boundaries…")
    ons_boundaries.load_all(db_path=staging, crs="bng", layers=layers)

    print("\n[3/3] Building H3 aggregations…")
    con = duckdb_connector(staging, writeable=True)
    try:
        transforms.build_all(con, resolutions=resolutions, replace=replace)
    finally:
        con.close()

    os.replace(staging, db_path)
    print(f"\n=== Done. Promoted staging database → {db_path} ===")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
