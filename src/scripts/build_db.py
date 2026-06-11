"""
Build the production DuckDB database in one reproducible pass.

Pipeline stages:
  1. extract.to_database   crime_data table (geometry, BNG)
  2. ons_boundaries.load_all   boundary tables (pfa, lad, msoa, lsoa, oa)
  3. open greenspace        OS Open Greenspace polygons (BNG)
  4. transforms.build_all   H3 count tables + geo lookup tables

The pipeline writes to a ``<name>.staging.db`` file and only promotes it to the live
database with an atomic ``os.replace`` once every stage has succeeded. Read-only
consumers therefore always see a complete database  either the old one or the new
one, never a half-built file.

The live database is the standard database (database_path(), under SAFER_STREETS_DATA_DIR);
pass --db-path to override.
"""

import os
from pathlib import Path

import duckdb
import typer

from safer_streets_core import transforms
from safer_streets_core.database import duckdb_connector, index_geometry_tables
from safer_streets_core.utils import data_dir, database_path
from scripts import extract, ons_boundaries

app = typer.Typer(help="Build the production crime + boundaries + H3 DuckDB database.")

# OS Open Greenspace (https://osdatahub.os.uk/data/downloads/open/OpenGreenspace): download the
# "ESRI Shape File" GB bundle and unzip it under the data directory. Already in BNG (EPSG:27700).
GREENSPACE_SHP = "OS Open Greenspace (ESRI Shape File) GB/data/GB_GreenspaceSite.shp"


def load_greenspace(con: duckdb.DuckDBPyConnection, shapefile: str | Path = GREENSPACE_SHP) -> None:
    """
    Create the `open_greenspace` table from the OS Open Greenspace shapefile.

    `shapefile` is resolved relative to the data directory unless it is an absolute path.
    Raises FileNotFoundError if the shapefile is missing. ST_Read yields a `geom` column,
    so index_geometry_tables repairs and RTree-indexes it with the boundary tables.
    """
    shp_path = Path(shapefile) if Path(shapefile).is_absolute() else data_dir() / shapefile
    if not shp_path.exists():
        raise FileNotFoundError(
            f"OS Open Greenspace shapefile not found: {shp_path}\n"
            "Download the 'ESRI Shape File' GB bundle from "
            "https://osdatahub.os.uk/data/downloads/open/OpenGreenspace and unzip it under the data directory."
        )

    # ENCODING=ISO-8859-1 matches the OS Open Greenspace shapefile
    con.execute(f"""
        CREATE OR REPLACE TABLE open_greenspace AS
        SELECT * FROM ST_Read('{shp_path}', open_options=['ENCODING=ISO-8859-1']);
    """)
    row_count = con.execute("SELECT COUNT(*) FROM open_greenspace").fetchone()[0]  # ty:ignore[not-subscriptable]
    print(f"  open_greenspace: {row_count:,} rows")


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

    print("\n[2/4] Downloading ONS boundaries…")
    ons_boundaries.load_all(db_path=staging, crs="bng", layers=layers, force_download=force_download)

    con = duckdb_connector(staging, writeable=True)
    try:
        print("\n[3/4] Loading open greenspace…")
        try:
            load_greenspace(con)
        except FileNotFoundError as exc:
            # greenspace is a manual OS download; warn but don't abort the whole build
            print(f"  Skipping greenspace: {exc}")

        print("\n[4/4] Validating geometries, indexing and building H3 aggregations…")
        # repair invalid geometries and add RTree indexes before the spatial joins below
        index_geometry_tables(con)
        transforms.build_all(con, resolutions=resolutions, replace=replace)
    finally:
        con.close()

    os.replace(staging, db_path)
    print(f"\n=== Done. Promoted staging database → {db_path} ===")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
