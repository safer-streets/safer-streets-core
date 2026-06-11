"""
Download generalised boundary shapefiles from the ONS Open Geography Portal
(https://geoportal.statistics.gov.uk/) via the ArcGIS REST API.

Supported boundary types:
  - Police Force Areas (December 2023, England & Wales, BGC)
  - Local Authority Districts (May 2024, UK, BGC)
  - 2021 Census MSOA  Middle Layer Super Output Areas (BGC V3)
  - 2021 Census LSOA  Lower Layer Super Output Areas (BGC V5)
  - 2021 Census OA   Output Areas (BGC V2)

All layers use the BGC (Generalised 20 m, clipped to coastline) variant,
which is the recommended choice for choropleth maps and spatial analysis.

Output formats:
  geojson    GeoJSON file, WGS-84 by default (default)
  shapefile  ESRI Shapefile (requires geopandas + pyogrio)
  duckdb     DuckDB database with spatial extension (requires duckdb + geopandas)

CRS options:
  wgs84  EPSG:4326 lon/lat (default)
  bng    EPSG:27700 British National Grid
          Coordinates are requested from the server in BNG directly, so no
          client-side reprojection is needed.  For DuckDB output the geometry

Requirements (install what you need):
    pip install requests
    pip install geopandas pyogrio        # for shapefile / duckdb output
    pip install duckdb                   # for duckdb output

Usage examples:
    # Download everything as GeoJSON (WGS-84)
    python ons_boundaries.py

    # Specific layers only
    python ons_boundaries.py --layers pfa lad

    # Shapefile in British National Grid
    python ons_boundaries.py --format shapefile --crs bng

    # DuckDB database (WGS-84)  all layers in one file
    python ons_boundaries.py --format duckdb

    # DuckDB database in BNG, custom path
    python ons_boundaries.py --format duckdb --crs bng --duckdb-path ~/data/boundaries.duckdb

    # Custom output directory
    python ons_boundaries.py --output ./boundaries

    # List available layers
    python ons_boundaries.py --list
"""

# TODO merge this script with extract...

import io
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
import typer

from safer_streets_core.database import duckdb_context
from safer_streets_core.utils import data_dir, database_path

# ---------------------------------------------------------------------------
# Layer catalogue
# All endpoints use the ONS ArcGIS tenant (ESMARspQHYMw9BZ9) and return
# BGC (generalised, clipped) boundaries.  Confirmed endpoints as of April 2026.
# ---------------------------------------------------------------------------


@lru_cache
def sources(filename: Path | None = None) -> dict[str, Any]:
    filename = filename or Path("./config/geodata_sources.json")
    with filename.open() as fd:
        return json.load(fd)


# CRS string constants
CRS_WGS84 = "EPSG:4326"
CRS_BNG = "EPSG:27700"

# ArcGIS outSR SRID values
ARCGIS_SRID = {
    "wgs84": "4326",
    "bng": "27700",
}

# ---------------------------------------------------------------------------
# ArcGIS REST helpers
# ---------------------------------------------------------------------------

PAGE_SIZE = 2000


def get_feature_count(endpoint: str, session: requests.Session) -> int:
    """Return the total number of features for the layer."""
    params = {"where": "1=1", "returnCountOnly": "true", "f": "json"}
    resp = session.get(endpoint, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"ArcGIS error: {data['error']}")
    return data["count"]


def fetch_page(
    endpoint: str,
    offset: int,
    page_size: int,
    session: requests.Session,
    out_sr: str = "4326",
    retries: int = 3,
) -> list[dict]:
    """Fetch one page of GeoJSON features from the ArcGIS REST API."""
    params = {
        "where": "1=1",
        "outFields": "*",
        "outSR": out_sr,
        "f": "geojson",
        "resultOffset": offset,
        "resultRecordCount": page_size,
    }
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(endpoint, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"ArcGIS error: {data['error']}")
            return data.get("features", [])
        except (requests.RequestException, RuntimeError) as exc:
            if attempt == retries:
                raise
            wait = 2**attempt
            print(f"    Attempt {attempt} failed ({exc}), retrying in {wait}s…")
            time.sleep(wait)
    return []


def fetch_all_features(
    layer_key: str,
    session: requests.Session,
    crs: str = "wgs84",
) -> tuple[list[dict], int]:
    """
    Page through the ArcGIS REST API and return all GeoJSON features.

    Coordinates are requested in the target CRS directly from the server
    (outSR parameter), so no client-side reprojection is needed.

    Returns (features, total_feature_count).
    """
    base_url = sources()["base_url"]
    endpoint = f"{base_url}{sources()['layers'][layer_key]['endpoint']}"
    out_sr = ARCGIS_SRID.get(crs, "4326")

    print("  Counting features…", end=" ", flush=True)
    total = get_feature_count(endpoint, session)
    print(f"{total:,} features found")

    all_features: list[dict] = []
    offset = 0
    page_num = 0

    while offset < total:
        page_num += 1
        pct = min(100, round(offset / total * 100))
        print(
            f"  Page {page_num:>4}: offset {offset:>7,} / {total:,}  ({pct}%)",
            end="\r",
            flush=True,
        )
        features = fetch_page(
            endpoint,
            offset,
            min(PAGE_SIZE, total - offset),
            session,
            out_sr,
        )
        if not features:
            print(f"\n  Warning: empty page at offset {offset}, stopping early.")
            break
        all_features.extend(features)
        offset += len(features)

    print(f"  Downloaded {len(all_features):,} features.          ")
    return all_features, total


def features_to_gdf(features: list[dict], crs: str):
    """
    Build a GeoDataFrame from a list of raw GeoJSON feature dicts.

    The GDF CRS is set to match the server-returned coordinates (no reprojection).
    """
    fc = {"type": "FeatureCollection", "features": features}
    gdf = gpd.read_file(io.StringIO(json.dumps(fc)))
    # normalise the column names (ST_Read doesn't have this option)
    gdf = gdf.rename(columns={col: col.lower().replace(" ", "_") for col in gdf.columns})
    crs_str = CRS_BNG if crs == "bng" else CRS_WGS84
    # The server returned these coordinates but GeoJSON doesn't embed CRS, so
    # we assign it explicitly (allow_override avoids an error if pyogrio guesses
    # EPSG:4326 for any numeric coordinates).
    gdf = gdf.set_crs(crs_str, allow_override=True)
    return gdf


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_geojson(features: list[dict], out_path: Path, crs: str) -> None:
    """
    Write features to a .geojson file.

    For BNG output a 'crs' member is added to the FeatureCollection so that
    GIS tools can identify the non-standard CRS.  This is a legacy GeoJSON
    extension (pre-RFC 7946); RFC 7946 recommends WGS-84 only, but the extra
    member is widely understood and far better than silent misidentification.
    """
    if crs == "bng":
        fc: dict = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::27700"},
            },
            "features": features,
        }
    else:
        fc = {"type": "FeatureCollection", "features": features}

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(fc, fh, separators=(",", ":"))
    print(f"  Written → {out_path}  ({out_path.stat().st_size / 1_048_576:.1f} MB)")


def write_shapefile(
    features: list[dict],
    shp_dir: Path,
    stem: str,
    crs: str,
) -> None:
    """Write features to an ESRI Shapefile directory."""
    print("  Building GeoDataFrame…", end=" ", flush=True)
    gdf = features_to_gdf(features, crs)
    print("done")
    shp_dir.mkdir(parents=True, exist_ok=True)
    out_path = shp_dir / f"{stem}.shp"
    print(f"  Writing Shapefile → {shp_dir}/")
    gdf.to_file(out_path, driver="ESRI Shapefile")
    total_size = sum(f.stat().st_size for f in shp_dir.iterdir()) / 1_048_576
    print(f"  Done  ({total_size:.1f} MB across component files)")


def write_geopackage(features: list[dict], gpkg_path: Path, crs: str) -> None:
    """
    Build a CRS-aware GeoDataFrame from features and write it to a GeoPackage.

    The GeoPackage is kept under data_dir() as a cache: DuckDB loads boundaries from it
    via ST_Read, and a subsequent run can reuse it instead of re-downloading from the API.
    """
    print("  Building GeoDataFrame…", end=" ", flush=True)
    gdf = features_to_gdf(features, crs)
    print("done")

    # overwrite any existing cache (GPKG would otherwise append a second layer)
    gpkg_path.unlink(missing_ok=True)
    print(f"  Caching GeoPackage → {gpkg_path}")
    gdf.to_file(gpkg_path, driver="GPKG")


def load_geopackage_into_duckdb(
    gpkg_path: Path,
    db_path: Path,
    table_name: str,
    id_column: str,
    crs: str,
) -> None:
    """
    Load a boundary GeoPackage into a DuckDB table using the spatial extension.

    ST_Read is the most reliable path for large feature sets and preserves geometry types.
    All layers are written into the same .duckdb file as separate tables, which makes
    cross-layer spatial joins very easy.
    """
    epsg = 27700 if crs == "bng" else 4326

    print(f"  Loading into DuckDB table '{table_name}'…", end=" ", flush=True)
    with duckdb_context(db_path, writeable=True) as con:
        # ST_Read returns a geometry column named 'geom'
        con.execute(f"""
            CREATE OR REPLACE TABLE "{table_name}" AS
            SELECT * FROM ST_Read('{gpkg_path}');
            ALTER TABLE {table_name} RENAME COLUMN {id_column} TO spatial_id;
        """)
        row_count = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]  # ty:ignore[not-subscriptable]
    print("done")
    print(f"  Table '{table_name}': {row_count:,} rows  (EPSG:{epsg})")

    db_size = db_path.stat().st_size / 1_048_576
    print(f"  DuckDB file size so far: {db_size:.1f} MB")


# ---------------------------------------------------------------------------
# Main download orchestrator
# ---------------------------------------------------------------------------


def download_layer(
    layer_key: str,
    output_dir: Path,
    out_format: str,
    crs: str,
    session: requests.Session,
    duckdb_path: Path | None = None,
    *,
    force_download: bool = False,
) -> Path:
    """
    Download all features for one layer and write to the requested format.

    For the duckdb format the boundary data is cached as a GeoPackage under output_dir;
    a subsequent run reuses it (unless force_download is set) rather than re-downloading.

    Returns the Path of the output file or directory.
    """
    info = sources()["layers"][layer_key]
    filename = info["filename"]
    table_name = info["table"]
    suffix = "_bng" if crs == "bng" else ""
    crs_label = "EPSG:27700 BNG" if crs == "bng" else "EPSG:4326 WGS-84"

    print(f"\n{'=' * 60}")
    print(f"Layer : {info['label']}")
    print(f"Note  : {info['note']}")
    print(f"CRS   : {crs_label}")
    print(f"URL   : {info['endpoint']}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if out_format == "geojson":
        features, _ = fetch_all_features(layer_key, session, crs=crs)
        out_path = output_dir / f"{filename}{suffix}.geojson"
        write_geojson(features, out_path, crs)
        return out_path

    elif out_format == "shapefile":
        features, _ = fetch_all_features(layer_key, session, crs=crs)
        stem = f"{filename}{suffix}"
        shp_dir = output_dir / stem
        write_shapefile(features, shp_dir, stem, crs)
        return shp_dir

    elif out_format == "duckdb":
        assert duckdb_path, "duckdb_path must be set if out_format is duckdb"
        gpkg_cache = output_dir / f"{filename}{suffix}.gpkg"
        if force_download or not gpkg_cache.exists():
            features, _ = fetch_all_features(layer_key, session, crs=crs)
            write_geopackage(features, gpkg_cache, crs)
        else:
            print(f"  Using cached {gpkg_cache}")
        load_geopackage_into_duckdb(gpkg_cache, duckdb_path, table_name, info["id_field"], crs)
        return duckdb_path

    else:
        raise ValueError(f"Unknown output format: {out_format!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(invoke_without_command=True, help="Download ONS Open Geography Portal boundary files.")


def list_layers() -> None:
    print("\nAvailable ONS boundary layers:\n")
    for key, info in sources()["layers"].items():
        print(f"  {key:<8}  {info['label']}")
        print(f"           {info['note']}")
    print()


def _download_layers(
    requested: list[str],
    out_format: str,
    crs: str,
    duckdb_path: Path | None,
    *,
    force_download: bool = False,
) -> list[tuple[str, Path | None, str | None]]:
    """Download each requested layer, collecting (key, output_path, error) tuples."""
    session = requests.Session()
    session.headers.update({"User-Agent": "ONS-Boundary-Downloader/1.0"})

    results: list[tuple[str, Path | None, str | None]] = []
    for key in requested:
        try:
            out_path = download_layer(
                key,
                data_dir(),
                out_format,
                crs,
                session,
                duckdb_path=duckdb_path,
                force_download=force_download,
            )
            results.append((key, out_path, None))
        except Exception as exc:  # noqa: BLE001
            print(f"\n  ERROR downloading {key}: {exc}")
            results.append((key, None, str(exc)))
    return results


def load_all(
    db_path: Path,
    crs: str = "bng",
    layers: list[str] | None = None,
    *,
    force_download: bool = False,
) -> list[tuple[str, Path | None, str | None]]:
    """
    Download boundary layers into a single DuckDB database (one table per layer).

    Callable directly by the build pipeline so it can redirect output to a staging
    database without spawning a subprocess. Each layer is cached as a GeoPackage under
    data_dir() and reused on later runs unless force_download is set. Returns the
    per-layer result tuples.
    """
    available_layers = list(sources()["layers"].keys())
    requested = available_layers if not layers or "all" in layers else layers
    return _download_layers(requested, "duckdb", crs, db_path, force_download=force_download)


@app.callback(invoke_without_command=True)
def main(
    layers: list[str] = typer.Option(  # noqa: B008
        ["all"],
        "--layers",
        help="Which layers to download. Choices: all plus available layer keys.",
        metavar="LAYER",
    ),
    out_format: str = typer.Option(
        "geojson",
        "--format",
        help="Output format: geojson | shapefile | duckdb.",
        show_default=True,
        case_sensitive=False,
        rich_help_panel="Output",
        show_choices=True,
        prompt=False,
        metavar="FORMAT",
    ),
    crs: str = typer.Option(
        "wgs84",
        "--crs",
        help=(
            "Coordinate reference system. wgs84 = EPSG:4326 lon/lat (default). bng = EPSG:27700 British National Grid."
        ),
        show_default=True,
        case_sensitive=False,
        show_choices=True,
    ),
    duckdb_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--duckdb-path",
        help=(
            "Path to the DuckDB file when --format duckdb is used. "
            "Multiple layers are written as separate tables into the same file."
        ),
        metavar="FILE",
    ),
    list_flag: bool = typer.Option(
        False,
        "--list",
        help="List available layers and exit.",
        is_flag=True,
    ),
) -> None:
    if list_flag:
        list_layers()
        raise typer.Exit()

    available_layers = list(sources()["layers"].keys())
    requested = available_layers if "all" in layers else layers

    # default to the standard database (database_path())
    resolved_duckdb_path = duckdb_path or database_path()
    crs_label = "EPSG:27700 BNG" if crs == "bng" else "EPSG:4326 WGS-84"

    print("\nONS Boundary Downloader")
    print(f"Layers  : {', '.join(requested)}")
    print(f"Format  : {out_format}")
    print(f"CRS     : {crs_label}")
    print(f"Output  : {data_dir().resolve()}")
    if out_format == "duckdb":
        print(f"DuckDB  : {resolved_duckdb_path.resolve()}")

    results = _download_layers(requested, out_format, crs, resolved_duckdb_path)

    # ---- summary -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Summary\n")
    for key, path, err in results:
        status = f"OK  → {path}" if path else f"FAIL  {err}"
        print(f"  {key:<8}  {status}")
    print()

    # ---- DuckDB quick-start snippet ------------------------------------
    if out_format == "duckdb":
        resolved_db = resolved_duckdb_path.resolve()
        if resolved_db.exists():
            ok_keys = [k for k, _, e in results if e is None]
            print("DuckDB quick-start:\n")
            print("    import duckdb")
            print(f"    con = duckdb.connect('{resolved_db}')")
            print("    con.execute('LOAD spatial')\n")
            for key in ok_keys:
                tname = sources()["layers"][key]["table"]
                id_f = sources()["layers"][key]["id_field"]
                print(f"    # {sources()['layers'][key]['label']}")
                print(f"    con.sql('SELECT {id_f}, ST_AsText(geom) FROM {tname} LIMIT 3').show()")
                print(f"    con.sql('SELECT COUNT(*) FROM {tname}').show()")
                print()
            print("    # Spatial query example (point-in-polygon, WGS-84):")
            print("    con.sql('''")
            print("        SELECT lad.LAD24NM")
            print("        FROM local_authority_districts lad")
            print("        WHERE ST_Within(")
            print("            ST_Point(-1.143, 53.958),  -- lon, lat")
            print("            lad.geom")
            print("        )")
            print("    ''').show()")
            print()


if __name__ == "__main__":
    app()
