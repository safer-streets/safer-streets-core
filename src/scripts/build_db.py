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
from zipfile import ZipFile

import duckdb
import requests
import typer
from tqdm import tqdm

from safer_streets_core import transforms
from safer_streets_core.database import duckdb_connector, index_geometry_tables
from safer_streets_core.utils import data_dir, database_path
from scripts import extract, ons_boundaries

app = typer.Typer(help="Build the production crime + boundaries + H3 DuckDB database.")

# OS Open Greenspace, fetched from the OS Downloads API (open data, no API key needed).
# The GB "ESRI Shapefile" bundle is a zip containing the GreenspaceSite (polygon) and
# AccessPoint (point) layers; we load the polygons. Data is already in BNG (EPSG:27700).
GREENSPACE_URL = (
    "https://api.os.uk/downloads/v1/products/OpenGreenspace/downloads?area=GB&format=ESRI%C2%AE+Shapefile&redirect"
)
GREENSPACE_ZIP = "opgrsp_essh_gb.zip"
GREENSPACE_LAYER = "GB_GreenspaceSite.shp"


def _download_greenspace(zip_path: Path) -> None:
    print(f"  Downloading OS Open Greenspace → {zip_path}…")
    response = requests.get(GREENSPACE_URL, stream=True, timeout=60)
    response.raise_for_status()
    size = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as fd, tqdm(total=size, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(1024**2):
            bar.update(len(chunk))
            fd.write(chunk)


def load_greenspace(con: duckdb.DuckDBPyConnection, *, force_download: bool = False) -> None:
    """
    Create the `open_greenspace` table from the OS Open Greenspace polygons.

    The GB shapefile bundle is downloaded from the OS Downloads API and cached under the
    data directory (reused unless force_download). The polygon layer is read straight from
    the zip via GDAL's /vsizip. ST_Read yields a `geom` column, so index_geometry_tables
    repairs and RTree-indexes it with the boundary tables.
    """
    zip_path = data_dir() / GREENSPACE_ZIP
    if force_download or not zip_path.exists():
        _download_greenspace(zip_path)
    else:
        print(f"  Using cached {zip_path}")

    with ZipFile(zip_path) as z:
        members = [name for name in z.namelist() if name.endswith(GREENSPACE_LAYER)]
    if not members:
        raise FileNotFoundError(f"{GREENSPACE_LAYER} not found in {zip_path}")

    # ENCODING=ISO-8859-1 matches the OS Open Greenspace shapefile
    vsizip = f"/vsizip/{zip_path}/{members[0]}"
    con.execute(f"""
        CREATE OR REPLACE TABLE open_greenspace AS
        SELECT * FROM ST_Read('{vsizip}', open_options=['ENCODING=ISO-8859-1']);
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

    print("[1/4] Extracting crime data…")
    extract.to_database(db_path=staging, force_download=force_download)

    print("\n[2/4] Downloading ONS boundaries…")
    ons_boundaries.load_all(db_path=staging, crs="bng", layers=layers, force_download=force_download)

    con = duckdb_connector(staging, writeable=True)
    try:
        print("\n[3/4] Loading open greenspace…")
        try:
            load_greenspace(con, force_download=force_download)
        except (requests.RequestException, FileNotFoundError) as exc:
            # greenspace is a supplementary dataset; warn but don't abort the whole build
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
