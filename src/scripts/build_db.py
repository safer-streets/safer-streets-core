"""
Build the production DuckDB database in one reproducible pass.

Pipeline stages:
  1. extract.to_database   crime_data table (geometry, BNG)
  2. ons_boundaries.load_all   boundary tables (pfa, lad, msoa, lsoa, oa)
  3. open greenspace + land cover   OS Open Greenspace + UKCEH Land Cover polygons (BNG)
  4. transforms.build_all   H3 count tables + geo/overlap lookup tables

The pipeline writes to a ``<name>.staging.db`` file and only promotes it to the live
database with an atomic ``os.replace`` once every stage has succeeded. Read-only
consumers therefore always see a complete database  either the old one or the new
one, never a half-built file.

The live database is the standard database (database_path(), under SAFER_STREETS_DATA_DIR);
pass --db-path to override.
"""

import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import duckdb
import requests
import typer
from tqdm import tqdm

from safer_streets_core import transforms
from safer_streets_core.database import duckdb_connector, duckdb_context, index_geometry_tables
from safer_streets_core.utils import data_dir, database_path
from scripts import extract, ons_boundaries

app = typer.Typer(help="Build the production crime + boundaries + H3 DuckDB database.")

# TODO merge these with the geodata_sources.json file

# OS Open Greenspace, fetched from the OS Downloads API (open data, no API key needed).
# The GB "ESRI Shapefile" bundle is a zip containing the GreenspaceSite (polygon) and
# AccessPoint (point) layers; we load the polygons. Data is already in BNG (EPSG:27700).
GREENSPACE_URL = (
    "https://api.os.uk/downloads/v1/products/OpenGreenspace/downloads?area=GB&format=ESRI%C2%AE+Shapefile&redirect"
)
GREENSPACE_ZIP = "opgrsp_essh_gb.zip"
GREENSPACE_LAYER = "GB_GreenspaceSite.shp"

# OS Open Roads
ROADS_URL = "https://api.os.uk/downloads/v1/products/OpenRoads/downloads?area=GB&format=GeoPackage&redirect"
ROADS_ZIP = "oproad_gpkg_gb.zip"
ROADS_LAYER = "Data/oproad_gb.gpkg"

LAND_COVER_ZIP = "Download_land+cover+2024_2983803.zip"
LAND_COVER_LAYER = "*.gpkg"  # there should only be one


def _download(url: str, zip_path: Path) -> None:
    print(f"  Downloading {zip_path}…")
    response = requests.get(url, stream=True, timeout=60)
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
        _download(GREENSPACE_URL, zip_path)
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


def _extract_cached(zip_path: Path, member: str) -> Path:
    """
    Extract a single zip member to a cached file beside the zip and return its path.

    ST_Read over /vsizip is much slower for a GeoPackage than reading an extracted file:
    GPKG is SQLite, so every random seek forces /vsizip to re-decompress from a sync point.
    The extracted file is cached and only re-extracted if the zip is newer.
    """
    dest = zip_path.with_suffix("") / Path(member).name
    if not dest.exists() or dest.stat().st_mtime < zip_path.stat().st_mtime:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Extracting {member}…")
        with ZipFile(zip_path) as z, z.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    return dest


# UKCEH Land Cover Map vector GeoPackage. Licensed (EIDC, https://catalogue.ceh.ac.uk/) so it
# cannot be auto-downloaded — download the LCM vector bundle geopackage. Data is already in BNG (EPSG:27700).
def load_land_cover(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create the `land_cover` table from the UKCEH Land Cover Map vector GeoPackage.

    The GeoPackage is located under the data directory by glob. ST_Read yields a `geom`
    column (with `gid` and `_mode`), so index_geometry_tables repairs and RTree-indexes it.
    """

    zip_path = data_dir() / LAND_COVER_ZIP
    if not zip_path.exists():
        raise FileNotFoundError(
            f"UKCEH Land Cover Map GeoPackage not found: {data_dir() / LAND_COVER_ZIP}\n"
            f"Download the LCM vector bundle from the EIDC (https://catalogue.ceh.ac.uk/) and place the zip "
            f"(named {LAND_COVER_ZIP}) in the data directory."
        )

    with ZipFile(zip_path) as z:
        members = [name for name in z.namelist() if name.endswith(".gpkg")]
    if not members:
        raise FileNotFoundError(f"No .gpkg found in {zip_path}")

    gpkg = _extract_cached(zip_path, members[0])
    print(f"  Loading land_cover from {gpkg}…")
    con.execute(f"""
        CREATE OR REPLACE TABLE land_cover AS
        SELECT * FROM ST_Read('{gpkg}');
    """)
    row_count = con.execute("SELECT COUNT(*) FROM land_cover").fetchone()[0]  # ty:ignore[not-subscriptable]
    print(f"  land_cover: {row_count:,} rows")


def _rename_geom_column(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Rename `table`'s geometry column to 'geom' if it has some other name (no-op if already 'geom')."""
    geom_cols = [
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ? AND table_schema = 'main' AND data_type LIKE 'GEOMETRY%'",
            [table],
        ).fetchall()
    ]
    if geom_cols and geom_cols[0] != "geom":
        con.execute(f'ALTER TABLE "{table}" RENAME COLUMN "{geom_cols[0]}" TO geom;')


def load_roads(con: duckdb.DuckDBPyConnection, *, force_download: bool = False) -> None:
    """
    Create the `open_roads` table from the OS Open Roads dataset.

    The GB geopackage is downloaded from the OS Downloads API and cached under the
    data directory (reused unless force_download).
    """
    zip_path = data_dir() / ROADS_ZIP
    if force_download or not zip_path.exists():
        _download(ROADS_URL, zip_path)
    else:
        print(f"  Using cached {zip_path}")

    with ZipFile(zip_path) as z:
        members = [name for name in z.namelist() if name.endswith(ROADS_LAYER)]
    if not members:
        raise FileNotFoundError(f"{ROADS_LAYER} not found in {zip_path}")

    gpkg = _extract_cached(zip_path, members[0])
    con.execute(f"""
        CREATE OR REPLACE TABLE open_roads AS
        SELECT * FROM ST_Read('{gpkg}', layer='road_link');
    """)
    # OS Open Roads names its geometry column 'geometry'; normalise to 'geom' for consistency
    # with the other tables (so index_geometry_tables and the H3 overlap lookups find it).
    _rename_geom_column(con, "open_roads")
    row_count = con.execute("SELECT COUNT(*) FROM open_roads").fetchone()[0]  # ty:ignore[not-subscriptable]
    print(f"  open_roads: {row_count:,} rows")


@app.command()
def build(
    db_path: Path | None = None,
    resolutions: list[int] = [8, 9, 10],  # noqa: B006
    layers: list[str] = ["all"],  # noqa: B006
    force_download: bool = False,
    replace: bool = True,
) -> None:
    """Run the full pipeline into a staging file, then atomically swap it into place.

    With --replace (default) the staging database is rebuilt from scratch. With --no-replace
    an existing staging database is reused: any stage whose output table is already present is
    skipped (crime, boundaries, greenspace, land cover, roads), and existing H3 tables/views
    are kept (CREATE ... IF NOT EXISTS). This lets an interrupted build resume without redoing
    completed stages — notably the large road download.
    """
    db_path = db_path or database_path()
    staging = db_path.with_suffix(".staging.db")
    if replace:
        staging.unlink(missing_ok=True)

    # under --no-replace, find what the staging DB already contains so finished stages are skipped
    existing: set[str] = set()
    if not replace and staging.exists():
        with duckdb_context(staging) as check:
            existing = {
                row[0]
                for row in check.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                ).fetchall()
            }

    print(f"\n=== Building {db_path} (staging: {staging}) ===\n")

    print("[1/4] Extracting crime data…")
    if "crime_data" in existing:
        print("  keeping existing crime_data")
    else:
        extract.to_database(db_path=staging, force_download=force_download)

    print("\n[2/4] Downloading ONS boundaries…")
    if "local_authority_districts" in existing:  # base boundary table → boundaries already loaded
        print("  keeping existing boundaries")
    else:
        ons_boundaries.load_all(db_path=staging, crs="bng", layers=layers, force_download=force_download)

    con = duckdb_connector(staging, writeable=True)
    try:
        print("\n[3/4] Loading greenspace, land cover, roads…")
        if "open_greenspace" in existing:
            print("  keeping existing open_greenspace")
        else:
            try:
                load_greenspace(con, force_download=force_download)
            except (requests.RequestException, FileNotFoundError) as exc:
                # supplementary datasets; warn but don't abort the whole build
                print(f"  Skipping greenspace: {exc}")
        if "land_cover" in existing:
            print("  keeping existing land_cover")
        else:
            try:
                load_land_cover(con)
            except FileNotFoundError as exc:
                print(f"  Skipping land cover: {exc}")
        if "open_roads" in existing:
            print("  keeping existing open_roads")
        else:
            try:
                load_roads(con, force_download=force_download)
            except (requests.RequestException, FileNotFoundError) as exc:
                print(f"  Skipping roads: {exc}")

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
