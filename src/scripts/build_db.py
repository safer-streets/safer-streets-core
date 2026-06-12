"""
Build the production DuckDB database in one reproducible pass.

Pipeline stages:
  1. extract.to_database   crime_data table (geometry, BNG)
  2. ons_boundaries.load_all   boundary tables (pfa, lad, msoa, lsoa, oa)
  3. supplementary layers   OS Open Greenspace + UKCEH Land Cover + OS Open Roads + CDRC Retail
                            Centres + Overture POI + GIAS schools (walk isochrones) + English & Welsh IoD
  4. transforms.build_all   H3 count tables + geo/overlap/nearest lookup tables

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
import pandas as pd
import requests
import typer
from overturemaps import core as overture
from tqdm import tqdm

from safer_streets_core import transforms
from safer_streets_core.database import duckdb_connector, duckdb_context, index_geometry_tables
from safer_streets_core.utils import data_dir, data_source, database_path
from scripts import extract, ons_boundaries

app = typer.Typer(help="Build the production crime + boundaries + H3 DuckDB database.")

# Source URLs and cached filenames (plus layer/sheet hints) live in config/data_sources.json, read via
# utils.data_source(key). This module keeps only the bits that aren't "a URL or a file" — the column
# schema and score-to-column mappings below, plus the Overture POI bbox/categories further down.

# Deprivation scores -> per-LSOA percentiles. The `imd_scores_pct` table covers all of England & Wales:
# English LSOAs come from the English IoD (data_source("imd")), Welsh LSOAs from the WIMD
# (data_source("wimd")). Both index the 2021 LSOA geography, so they share `spatial_id` and merge by
# row. Percentiles are computed WITHIN each country (England vs Wales are separate deprivation indices
# and are not comparable across the border), so a percentile always means "relative to other LSOAs in
# the same country".
# original IoD column name -> short name; everything except IMD_PASSTHROUGH is percentile-ranked. This
# is also the column set of the merged table, so the Welsh side maps onto the same short names.
IMD_COLUMNS = {
    "LSOA code (2021)": "spatial_id",
    "Local Authority District code (2024)": "lad24cd",
    "Local Authority District name (2024)": "lad24nm",
    "Index of Multiple Deprivation (IMD) Score": "imd_score",
    "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)": "imd_rank",
    "Income Score (rate)": "income",
    "Employment Score (rate)": "employment",
    "Education, Skills and Training Score": "est",
    "Health Deprivation and Disability Score": "hdd",
    "Crime Score": "crime",
    "Barriers to Housing and Services Score": "bhs",
    "Living Environment Score": "le",
}
IMD_PASSTHROUGH = ("spatial_id", "lad24cd", "lad24nm", "imd_rank")

# WIMD scores use the same "higher = more deprived" convention as the English IoD. Welsh domains map
# onto the English short names below; "Access to Services" + "Housing" are averaged into `bhs`
# (England's single "Barriers to Housing and Services" domain), and the overall WIMD score also yields
# `imd_rank` (1 = most deprived). There is no Welsh equivalent of the English sub-domains, which is why
# they are dropped from IMD_COLUMNS above.
# WIMD "Data" sheet column -> imd_scores_pct short name (each percentile-ranked within Wales)
WIMD_DOMAINS = {
    "WIMD 2025": "imd_score",
    "Income": "income",
    "Employment": "employment",
    "Education": "est",
    "Health": "hdd",
    "Community Safety": "crime",
    "Physical Environment": "le",
}

# Overture Maps places (POI), streamed from S3 via the overturemaps reader (no API key).
# England & Wales bounding box in WGS-84 (xmin, ymin, xmax, ymax).
POI_BBOX = (-5.86, 49.75, 1.81, 56.0)
POI_CATEGORIES = (
    "adult_entertainment_venue",
    "alcoholic_beverage_venue",
    "atm",
    "bar",
    "casino",
    "casual_eatery",
    "emergency_department",
    "fast_food_restaurant",
    "food_service",
    "hospital",
    "inn",
    "lounge",
    "parking",
    "police station",
    "restaurant",
)


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

    The GB shapefile bundle (open data, no API key) is downloaded from the OS Downloads API and
    cached under the data directory (reused unless force_download). The bundle is a zip of the
    GreenspaceSite (polygon) and AccessPoint (point) layers; we load the polygons. Data is already in
    BNG (EPSG:27700). The polygon layer is read straight from the zip via GDAL's /vsizip. ST_Read
    yields a `geom` column, so index_geometry_tables repairs and RTree-indexes it with the boundaries.
    """
    src = data_source("greenspace")
    zip_path = data_dir() / src["zip"]
    if force_download or not zip_path.exists():
        _download(src["url"], zip_path)
    else:
        print(f"  Using cached {zip_path}")

    with ZipFile(zip_path) as z:
        members = [name for name in z.namelist() if name.endswith(src["layer"])]
    if not members:
        raise FileNotFoundError(f"{src['layer']} not found in {zip_path}")

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


def load_land_cover(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create the `land_cover` table from the UKCEH Land Cover Map vector GeoPackage.

    Licensed (EIDC, https://catalogue.ceh.ac.uk/) so it cannot be auto-downloaded — the LCM vector
    bundle zip is located under the data directory (the `.gpkg` inside it is read; already BNG /
    EPSG:27700). ST_Read yields a `geom` column (with `gid` and `_mode`), so index_geometry_tables
    repairs and RTree-indexes it.
    """
    zip_name = data_source("land_cover")["zip"]
    zip_path = data_dir() / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(
            f"UKCEH Land Cover Map GeoPackage not found: {zip_path}\n"
            f"Download the LCM vector bundle from the EIDC (https://catalogue.ceh.ac.uk/) and place the zip "
            f"(named {zip_name}) in the data directory."
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

    The GB geopackage (open data, no API key) is downloaded from the OS Downloads API and cached
    under the data directory (reused unless force_download).
    """
    src = data_source("roads")
    zip_path = data_dir() / src["zip"]
    if force_download or not zip_path.exists():
        _download(src["url"], zip_path)
    else:
        print(f"  Using cached {zip_path}")

    with ZipFile(zip_path) as z:
        members = [name for name in z.namelist() if name.endswith(src["layer"])]
    if not members:
        raise FileNotFoundError(f"{src['layer']} not found in {zip_path}")

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


def load_retail_centres(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create the `retail_centres` table from the GeoDS Retail Centre Boundaries GeoPackage.

    Licensed (Geographic Data Service), so the GeoPackage is downloaded manually into the data
    directory. ST_Read yields a `geom` column, so index_geometry_tables RTree-indexes it.
    """
    gpkg_name = data_source("retail_centres")["gpkg"]
    gpkg = data_dir() / gpkg_name
    if not gpkg.exists():
        raise FileNotFoundError(
            f"GeoDS Retail Centre Boundaries GeoPackage not found: {gpkg}\n"
            f"Download {gpkg_name} from the Geographic Data Service (https://geods.ac.uk/) and place it in the data directory."
        )

    print(f"  Loading retail_centres from {gpkg}…")
    con.execute(f"""
        CREATE OR REPLACE TABLE retail_centres AS
        SELECT
            rc.RC_ID AS rc_id,
            rc.RC_Name AS rc_name,
            rc.Classification AS classification,
            rc.Country AS country,
            rc.Region_NM AS region_nm,
            rc.H3_count AS h3_count,
            rc.Retail_N AS retail_n,
            rc.Area_km2 AS area_km2,
            rc.geom AS geom
        FROM ST_Read('{gpkg}') rc;
    """)
    row_count = con.execute("SELECT COUNT(*) FROM retail_centres").fetchone()[0]  # ty:ignore[not-subscriptable]
    print(f"  retail_centres: {row_count:,} rows")


def load_poi(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create the `poi` table from Overture Maps places (points of interest).

    The Overture `place` theme is streamed from S3 (anonymous, no API key) straight into DuckDB
    via the overturemaps reader — no intermediate file. Only the categories of interest are kept,
    and geometry is transformed from WGS-84 to BNG (yielding a `geom` column that
    index_geometry_tables then RTree-indexes).
    """
    print("  Streaming Overture places…")
    reader = overture.record_batch_reader("place", bbox=POI_BBOX)
    if reader is None:
        raise RuntimeError("Overture Maps download failed (record_batch_reader returned None)")

    # `reader` is consumed directly by DuckDB via an Arrow replacement scan
    con.execute(
        """
        CREATE OR REPLACE TABLE poi AS SELECT
            id AS poi_id,
            ST_Transform(geometry, 'EPSG:4326', 'EPSG:27700', always_xy := true) AS geom,
            names.primary AS name,
            addresses[1].postcode AS postcode,
            basic_category,
            categories.primary AS primary_category,
            categories.alternate AS alternate_category
        FROM reader
        WHERE basic_category = ANY(?)
        """,
        [list(POI_CATEGORIES)],
    )
    row_count = con.execute("SELECT COUNT(*) FROM poi").fetchone()[0]  # ty:ignore[not-subscriptable]
    print(f"  poi: {row_count:,} rows")


# Schools: GIAS "Get Information About Schools" export (data_source("schools")). Isochrones are
# 10-minute walk catchments over the open_roads network.
WALK_TRIP_MINUTES = 10
WALK_SPEED_KMH = 5
WALK_RADIUS_M = WALK_TRIP_MINUTES * WALK_SPEED_KMH * 1000 / 60  # reachable network distance, metres


def _walk_isochrones(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    school_xy: pd.DataFrame,
    radius: float = WALK_RADIUS_M,
) -> list:
    """
    Return a walk-isochrone geometry per school (aligned with `school_xy`).

    `edges` has start_node/end_node/length (a topological road graph), `nodes` has node/x/y, and
    `school_xy` has x/y point coordinates (BNG). Each school is snapped to its nearest road node;
    its isochrone is the convex hull of all nodes reachable within `radius` metres (bounded
    single-source Dijkstra), or the snap point itself if the node is isolated.
    """
    import networkx as nx  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415  # ty:ignore[unresolved-import]
    from shapely.geometry import MultiPoint, Point  # noqa: PLC0415

    graph = nx.from_pandas_edgelist(edges, "start_node", "end_node", edge_attr="length")
    coords = dict(zip(nodes.node, zip(nodes.x, nodes.y, strict=True), strict=True))

    # snap each school to the nearest road node (vectorised KDTree query)
    tree = cKDTree(nodes[["x", "y"]].to_numpy())
    _, idx = tree.query(school_xy[["x", "y"]].to_numpy())
    school_nodes = nodes.node.to_numpy()[idx]

    isochrones = []
    for node in school_nodes:
        if node not in graph:
            isochrones.append(Point(coords[node]))
            continue
        reached = nx.single_source_dijkstra_path_length(graph, node, cutoff=radius, weight="length")
        isochrones.append(MultiPoint([coords[n] for n in reached]).convex_hull)
    return isochrones


def load_schools(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create the `schools` table from the GIAS export, with a 10-minute walk isochrone per school.

    Open schools with valid coordinates are parsed into a point `geom` (BNG); a walk catchment
    `isochrone` polygon is then computed over the open_roads network (so this requires open_roads).
    """
    glob = data_source("schools")["glob"]
    matches = sorted(data_dir().glob(glob))
    if not matches:
        raise FileNotFoundError(
            f"GIAS schools export not found under {data_dir()} (glob {glob}).\n"
            "Download 'all establishment data' from https://get-information-schools.service.gov.uk/ "
            "and place the CSV in the data directory."
        )
    if "open_roads" not in {
        r[0]
        for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
    }:
        raise RuntimeError("schools isochrones require the open_roads table (load roads first)")

    gias = matches[-1]
    print(f"  Parsing GIAS export {gias}…")
    con.execute(f"""
        CREATE OR REPLACE VIEW schools_stg AS
        WITH base AS (
            SELECT
                urn, establishmentnumber, establishmentname,
                typeofestablishment_code, typeofestablishment_name,
                establishmentstatus_code, phaseofeducation_code, phaseofeducation_name,
                statutorylowage, statutoryhighage, schoolcapacity, postcode,
                urbanrural_code, districtadministrative_code, msoa_code, lsoa_code,
                ST_Point(easting, northing) AS geom
            FROM read_csv_auto('{gias}', encoding='CP1252', normalize_names=true)
            WHERE establishmentstatus_code IN (1, 3) AND easting > 0 AND northing > 0
        )
        SELECT * FROM base;
    """)

    # the road links are already a topological graph (start_node -- end_node, weighted by length)
    edges = con.sql("SELECT start_node, end_node, length FROM open_roads WHERE length > 0").df()
    nodes = con.sql("""
        SELECT DISTINCT ON (node) node, ST_X(pt) AS x, ST_Y(pt) AS y
        FROM (
            SELECT start_node AS node, ST_StartPoint(geom) AS pt FROM open_roads
            UNION ALL
            SELECT end_node AS node, ST_EndPoint(geom) AS pt FROM open_roads
        )
    """).df()
    school_xy = con.sql("SELECT urn, ST_X(geom) AS x, ST_Y(geom) AS y FROM schools_stg").df()

    print(f"  Computing {len(school_xy):,} school isochrones over {len(nodes):,} road nodes…")
    isochrones = _walk_isochrones(edges, nodes, school_xy)

    iso_wkt = pd.DataFrame({"urn": school_xy.urn.to_numpy(), "wkt": [g.wkt for g in isochrones]})
    con.register("schools_iso_tmp", iso_wkt)
    try:
        con.execute("""
            CREATE OR REPLACE TABLE schools AS
            SELECT
                s.*,
                ST_GeomFromText(t.wkt) AS isochrone,
                ST_Area(ST_GeomFromText(t.wkt)) / 1e6 AS isochrone_area_km2
            FROM schools_stg s
            LEFT JOIN schools_iso_tmp t USING (urn);
        """)
    finally:
        con.unregister("schools_iso_tmp")
    row_count = con.execute("SELECT COUNT(*) FROM schools").fetchone()[0]  # ty:ignore[not-subscriptable]
    print(f"  schools: {row_count:,} rows")


def _imd_england(*, force_download: bool = False) -> pd.DataFrame:
    """English IoD "File 7" as per-LSOA percentiles (the IMD_COLUMNS short names).

    The CSV is downloaded from gov.uk and cached under the data directory (reused unless
    force_download). Columns of interest are renamed and every score column (all except
    IMD_PASSTHROUGH) is replaced by its percentile rank within England (0–1, higher = more deprived).
    """
    src = data_source("imd")
    matches = sorted(data_dir().glob(src["glob"]))
    if force_download or not matches:
        csv_path = data_dir() / src["csv"]
        _download(src["url"], csv_path)
    else:
        csv_path = matches[-1]
        print(f"  Using cached {csv_path}")

    imd = pd.read_csv(csv_path)[list(IMD_COLUMNS)].rename(columns=IMD_COLUMNS)
    for column in imd.columns:
        if column not in IMD_PASSTHROUGH:
            imd[column] = imd[column].rank(pct=True)
    return imd


def _welsh_lad_codes(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Map LA name -> LAD24 code from the boundary table, so Welsh rows can be given an `lad24cd`
    (the WIMD file carries the LA name but not its code). Empty if boundaries aren't loaded yet or the
    lookup fails, in which case `lad24cd` is left null for Welsh rows."""
    tables = {
        r[0]
        for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
    }
    if "local_authority_districts" not in tables:
        return {}
    try:
        return {
            name: code
            for code, name in con.execute("SELECT spatial_id, lad24nm FROM local_authority_districts").fetchall()
        }
    except duckdb.Error:
        return {}


def _imd_wales(con: duckdb.DuckDBPyConnection, *, force_download: bool = False) -> pd.DataFrame:
    """Welsh WIMD scores as per-LSOA percentiles, on the same IMD_COLUMNS short names as England.

    The ODS is downloaded from gov.wales and cached under the data directory (reused unless
    force_download). WIMD scores use the same "higher = more deprived" convention as the English IoD,
    so each is percentile-ranked within Wales the same way. `imd_rank` is derived from the overall
    score (1 = most deprived); `lad24cd` is looked up from the boundary table by LA name.
    """
    src = data_source("wimd")
    ods_path = data_dir() / src["ods"]
    if force_download or not ods_path.exists():
        _download(src["url"], ods_path)
    else:
        print(f"  Using cached {ods_path}")

    raw = pd.read_excel(ods_path, engine="odf", sheet_name=src["sheet"], header=src["header_row"])
    raw = raw.rename(columns=lambda c: str(c).strip())

    wales = pd.DataFrame({"spatial_id": raw["LSOA code"], "lad24nm": raw["Local Authority name"]})
    wales["lad24cd"] = wales["lad24nm"].map(_welsh_lad_codes(con))
    wales["imd_rank"] = raw["WIMD 2025"].rank(ascending=False, method="min").astype("int64")
    for ods_column, short in WIMD_DOMAINS.items():
        wales[short] = raw[ods_column].rank(pct=True)
    # England's single "Barriers to Housing and Services" domain ≈ Wales' "Access to Services" +
    # "Housing"; average the two scores, then percentile-rank that composite.
    wales["bhs"] = ((raw["Access to Services"] + raw["Housing"]) / 2).rank(pct=True)
    return wales[list(IMD_COLUMNS.values())]


def load_imd(con: duckdb.DuckDBPyConnection, *, force_download: bool = False) -> None:
    """
    Create the `imd_scores_pct` table: England (IoD) + Wales (WIMD) deprivation as per-LSOA percentiles.

    Both indices are reduced to the same IMD_COLUMNS short names, percentile-ranked WITHIN each country
    (higher = more deprived, so percentiles are not comparable across the border), then unioned into one
    table keyed by `spatial_id` (the LSOA21 code) for joining to the lsoa geography. If the Welsh data
    can't be fetched the table is still built from England alone.
    """
    england = _imd_england(force_download=force_download)
    try:
        wales = _imd_wales(con, force_download=force_download)
    except (requests.RequestException, KeyError, ValueError) as exc:
        print(f"  Welsh WIMD unavailable ({exc}); building imd_scores_pct from England only")
        wales = england.iloc[:0]
    imd = pd.concat([england, wales], ignore_index=True)

    con.register("imd_scores_pct_stg", imd)
    try:
        con.execute("CREATE OR REPLACE TABLE imd_scores_pct AS SELECT * FROM imd_scores_pct_stg;")
    finally:
        con.unregister("imd_scores_pct_stg")
    row_count = con.execute("SELECT COUNT(*) FROM imd_scores_pct").fetchone()[0]  # ty:ignore[not-subscriptable]
    print(f"  imd_scores_pct: {row_count:,} rows")


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
        print("\n[3/4] Loading greenspace, land cover, roads, retail centres, POI, schools, IMD…")
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
        if "retail_centres" in existing:
            print("  keeping existing retail_centres")
        else:
            try:
                load_retail_centres(con)
            except FileNotFoundError as exc:
                print(f"  Skipping retail centres: {exc}")
        if "open_roads" in existing:
            print("  keeping existing open_roads")
        else:
            try:
                load_roads(con, force_download=force_download)
            except (requests.RequestException, FileNotFoundError) as exc:
                print(f"  Skipping roads: {exc}")
        if "poi" in existing:
            print("  keeping existing poi")
        else:
            try:
                load_poi(con)
            except Exception as exc:  # noqa: BLE001  # Overture S3 read can fail in various ways
                print(f"  Skipping POI: {exc}")
        if "schools" in existing:
            print("  keeping existing schools")
        else:
            try:
                load_schools(con)  # needs open_roads (loaded above) for the isochrone network
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"  Skipping schools: {exc}")
        if "imd_scores_pct" in existing:
            print("  keeping existing imd_scores_pct")
        else:
            try:
                load_imd(con, force_download=force_download)
            except (requests.RequestException, FileNotFoundError) as exc:
                print(f"  Skipping IMD: {exc}")

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
