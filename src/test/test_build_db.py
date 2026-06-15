import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile

import duckdb
import geopandas as gpd
import pandas as pd
import pytest
import requests
from shapely import LineString, Polygon

from safer_streets_core.database import duckdb_connector, index_geometry_tables
from safer_streets_core.utils import data_source
from scripts import build_db
from scripts.build_db import (
    load_greenspace,
    load_land_cover,
    load_roads,
)

# source filenames now live in config/data_sources.json (read via data_source); fetch the ones the
# fixtures need so tests stay in step with the catalogue
GREENSPACE_ZIP = data_source("greenspace")["zip"]
LAND_COVER_ZIP = data_source("land_cover")["zip"]
ROADS_ZIP = data_source("roads")["zip"]

# inner path mirrors the real OS bundle layout (…/data/GB_GreenspaceSite.shp)
_INNER_DIR = "OS Open Greenspace (ESRI Shape File) GB/data"


def _make_greenspace_zip(zip_path: Path, *, layer: str = "GB_GreenspaceSite") -> None:
    """Build a synthetic OS Open Greenspace bundle zip containing a polygon shapefile."""
    gdf = gpd.GeoDataFrame(
        {"id": ["G1", "G2"], "function": ["Public Park Or Garden", "Play Space"]},
        geometry=[
            Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            Polygon([(200, 200), (300, 200), (300, 300), (200, 300)]),
        ],
        crs="EPSG:27700",
    )
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        gdf.to_file(tmp / f"{layer}.shp")
        with ZipFile(zip_path, "w") as z:
            for f in tmp.iterdir():
                z.write(f, arcname=f"{_INNER_DIR}/{f.name}")


def test_raises_when_layer_absent(tmp_path, monkeypatch):
    # a cached zip that lacks the GreenspaceSite layer should raise (before touching the connection)
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    _make_greenspace_zip(tmp_path / GREENSPACE_ZIP, layer="GB_AccessPoint")
    with pytest.raises(FileNotFoundError, match="GB_GreenspaceSite.shp not found"):
        load_greenspace(MagicMock())


def test_loads_from_cached_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    _make_greenspace_zip(tmp_path / GREENSPACE_ZIP)

    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:  # spatial extension download unavailable
        pytest.skip(f"extension download unavailable: {e}")

    load_greenspace(con)  # zip already cached → no download

    cols = [d[0] for d in con.execute("SELECT * FROM open_greenspace LIMIT 0").description]
    assert "geom" in cols  # ST_Read names the geometry column 'geom', so it gets indexed
    assert con.execute("SELECT COUNT(*) FROM open_greenspace").fetchone()[0] == 2  # ty:ignore[not-subscriptable]

    # ST_Read gives a CRS-qualified GEOMETRY type; index_geometry_tables normalises it and indexes.
    index_geometry_tables(con)
    index_geometry_tables(con)  # idempotent
    indexes = {r[0] for r in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "open_greenspace_geom_rtree" in indexes
    con.close()


def test_downloads_when_zip_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    # stand in for the network download: materialise the cached zip when invoked
    called = {"n": 0}

    def fake_download(url: str, zip_path: Path) -> None:
        called["n"] += 1
        _make_greenspace_zip(zip_path)

    monkeypatch.setattr(build_db, "_download", fake_download)

    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")

    load_greenspace(con)
    assert called["n"] == 1  # download triggered because the zip was absent
    assert con.execute("SELECT COUNT(*) FROM open_greenspace").fetchone()[0] == 2
    con.close()


def _gpkg_in_zip(zip_path: Path, gdf: gpd.GeoDataFrame, arcname: str, *, layer: str | None = None) -> None:
    """Write `gdf` to a GeoPackage and pack it into a zip at `arcname` (mirrors the real bundles)."""
    with tempfile.TemporaryDirectory() as td:
        gpkg = Path(td) / "data.gpkg"
        gdf.to_file(gpkg, driver="GPKG", layer=layer) if layer else gdf.to_file(gpkg, driver="GPKG")
        with ZipFile(zip_path, "w") as z:
            z.write(gpkg, arcname=arcname)


def test_land_cover_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="Land Cover Map GeoPackage not found"):
        load_land_cover(MagicMock())


def test_land_cover_loads_and_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    gdf = gpd.GeoDataFrame(
        {"gid": [1, 2], "_mode": [20, 21]},  # 20 = urban, 21 = suburban
        geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]), Polygon([(200, 200), (300, 200), (300, 300)])],
        crs="EPSG:27700",
    )
    _gpkg_in_zip(tmp_path / LAND_COVER_ZIP, gdf, "lcm-2024.gpkg")

    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")

    load_land_cover(con)
    cols = [d[0] for d in con.execute("SELECT * FROM land_cover LIMIT 0").description]
    assert {"gid", "_mode", "geom"} <= set(cols)
    assert con.execute("SELECT COUNT(*) FROM land_cover").fetchone()[0] == 2  # ty:ignore[not-subscriptable]

    index_geometry_tables(con)
    indexes = {r[0] for r in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "land_cover_geom_rtree" in indexes
    con.close()


def test_roads_loads_road_link_layer(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    gdf = gpd.GeoDataFrame(
        {"id": ["R1", "R2"], "road_function": ["Local Road", "A Road"]},
        geometry=[LineString([(0, 0), (100, 100)]), LineString([(0, 100), (100, 0)])],
        crs="EPSG:27700",
    )
    # mirror the real bundle: a gpkg at Data/oproad_gb.gpkg with a 'road_link' layer
    _gpkg_in_zip(tmp_path / ROADS_ZIP, gdf, "Data/oproad_gb.gpkg", layer="road_link")

    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")

    load_roads(con)  # zip cached → no download
    cols = [d[0] for d in con.execute("SELECT * FROM open_roads LIMIT 0").description]
    assert {"id", "road_function", "geom"} <= set(cols)  # geom column → gets RTree-indexed
    assert con.execute("SELECT COUNT(*) FROM open_roads").fetchone()[0] == 2  # ty:ignore[not-subscriptable]

    index_geometry_tables(con)
    indexes = {r[0] for r in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "open_roads_geom_rtree" in indexes
    con.close()


def test_rename_geom_column_normalises_geometry():
    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")

    # mimic real OS Open Roads, whose geometry column is named 'geometry'
    con.execute("CREATE TABLE open_roads AS SELECT 1 AS id, ST_Point(0, 0) AS geometry;")
    build_db._rename_geom_column(con, "open_roads")
    cols = {d[0] for d in con.execute("SELECT * FROM open_roads LIMIT 0").description}
    assert "geom" in cols and "geometry" not in cols

    # already-'geom' tables are left untouched
    build_db._rename_geom_column(con, "open_roads")
    assert "geom" in {d[0] for d in con.execute("SELECT * FROM open_roads LIMIT 0").description}
    con.close()


def test_no_replace_skips_existing_stages(tmp_path, monkeypatch):
    """--no-replace skips any stage whose output table is already in the staging DB."""
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    db_path = tmp_path / "safer_streets.db"
    staging = db_path.with_suffix(".staging.db")

    try:
        con = duckdb_connector(staging, writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")
    # pre-populate a partial staging DB: crime, boundaries and greenspace done; land cover/roads not
    for table in ("crime_data", "local_authority_districts", "open_greenspace"):
        con.execute(f"CREATE TABLE {table} AS SELECT 1 AS x")
    con.close()

    calls: list[str] = []
    monkeypatch.setattr(build_db.extract, "to_database", lambda **k: calls.append("crime"))
    monkeypatch.setattr(build_db.ons_boundaries, "load_all", lambda **k: calls.append("boundaries"))
    monkeypatch.setattr(build_db, "load_greenspace", lambda *a, **k: calls.append("greenspace"))
    monkeypatch.setattr(build_db, "load_land_cover", lambda *a, **k: calls.append("land_cover"))
    monkeypatch.setattr(build_db, "load_retail_centres", lambda *a, **k: calls.append("retail_centres"))
    monkeypatch.setattr(build_db, "load_roads", lambda *a, **k: calls.append("roads"))
    monkeypatch.setattr(build_db, "load_poi", lambda *a, **k: calls.append("poi"))
    monkeypatch.setattr(build_db, "load_schools", lambda *a, **k: calls.append("schools"))
    monkeypatch.setattr(build_db, "load_imd", lambda *a, **k: calls.append("imd"))
    monkeypatch.setattr(build_db, "index_geometry_tables", lambda con: None)
    monkeypatch.setattr(build_db.transforms, "build_all", lambda con, **k: None)
    monkeypatch.setattr(build_db.os, "replace", lambda src, dst: None)

    build_db.build(db_path=db_path, replace=False)

    # already-present stages (crime, boundaries, greenspace) are skipped; the rest run in order
    assert calls == ["land_cover", "retail_centres", "roads", "poi", "schools", "imd"]


def test_extract_cached_extracts_reuses_and_refreshes(tmp_path):
    import os
    import time

    zp = tmp_path / "bundle.zip"
    with ZipFile(zp, "w") as z:
        z.writestr("Data/file.gpkg", b"v1")

    out = build_db._extract_cached(zp, "Data/file.gpkg")
    assert out == tmp_path / "bundle" / "file.gpkg"
    assert out.read_bytes() == b"v1"

    # second call with an unchanged zip reuses the extracted file (no re-extract)
    mtime = out.stat().st_mtime
    assert build_db._extract_cached(zp, "Data/file.gpkg").stat().st_mtime == mtime

    # a newer zip triggers re-extraction
    with ZipFile(zp, "w") as z:
        z.writestr("Data/file.gpkg", b"v2")
    os.utime(zp, (time.time() + 10, time.time() + 10))
    assert build_db._extract_cached(zp, "Data/file.gpkg").read_bytes() == b"v2"


def test_load_poi_streams_filtered_places(monkeypatch):
    """Integration: stream a tiny Overture bbox into the poi table (skipped if S3 is unreachable)."""
    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")

    # a tiny bbox keeps the download small; the module default is all of E&W
    monkeypatch.setattr(build_db, "POI_BBOX", (-1.84, 53.91, -1.80, 53.94))
    try:
        build_db.load_poi(con)
    except Exception as e:  # noqa: BLE001 — network/S3 unavailable in this environment
        con.close()
        pytest.skip(f"Overture S3 unavailable: {e}")

    cols = {d[0] for d in con.execute("SELECT * FROM poi LIMIT 0").description}
    assert cols == {"poi_id", "geom", "name", "postcode", "basic_category", "primary_category", "alternate_category"}
    # geometry transformed to BNG GEOMETRY (so it gets RTree-indexed)
    geom_type = con.execute(
        "SELECT data_type FROM information_schema.columns WHERE table_name='poi' AND column_name='geom'"
    ).fetchone()[0]  # ty:ignore[not-subscriptable]
    assert geom_type == "GEOMETRY"
    # only the requested categories are kept
    cats = {r[0] for r in con.execute("SELECT DISTINCT basic_category FROM poi").fetchall()}
    assert cats <= set(build_db.POI_CATEGORIES)

    index_geometry_tables(con)
    indexes = {r[0] for r in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "poi_geom_rtree" in indexes
    con.close()


def test_retail_centres_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="Retail Centre Boundaries GeoPackage not found"):
        build_db.load_retail_centres(MagicMock())


def test_retail_centres_loads_and_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    gdf = gpd.GeoDataFrame(
        {
            "RC_ID": ["RC1", "RC2"],
            "RC_Name": ["Centre A", "Centre B"],
            "Classification": ["Regional Centre", "Local Centre"],
            "Country": ["England", "England"],
            "Region_NM": ["Yorkshire", "Yorkshire"],
            "H3_count": [10, 3],
            "Retail_N": [100, 20],
            "Area_km2": [0.8, 0.2],
        },
        # the GeoDS product is supplied in WGS-84 (lon/lat); use UK coordinates so the
        # reprojection to BNG lands in a valid range
        geometry=[
            Polygon([(-1.5, 53.8), (-1.4, 53.8), (-1.4, 53.9), (-1.5, 53.9)]),
            Polygon([(-1.3, 53.7), (-1.2, 53.7), (-1.2, 53.8)]),
        ],
        crs="EPSG:4326",
    )
    gdf.to_file(tmp_path / data_source("retail_centres")["gpkg"], driver="GPKG")

    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")

    build_db.load_retail_centres(con)
    cols = {d[0] for d in con.execute("SELECT * FROM retail_centres LIMIT 0").description}
    assert {"rc_id", "rc_name", "classification", "geom"} <= cols  # columns lower-cased, geom indexable
    assert con.execute("SELECT COUNT(*) FROM retail_centres").fetchone()[0] == 2  # ty:ignore[not-subscriptable]
    # geometry reprojected to BNG metres (eastings ~400-500 km), not left as lon/lat degrees
    assert con.execute("SELECT MIN(ST_XMin(geom)) FROM retail_centres").fetchone()[0] > 1000  # ty:ignore[not-subscriptable]

    index_geometry_tables(con)
    indexes = {r[0] for r in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "retail_centres_geom_rtree" in indexes
    con.close()


# --- schools / isochrones ---

_SQUARE_ROADS = """
    CREATE TABLE open_roads AS SELECT * FROM (VALUES
        ('A','B',100.0, ST_GeomFromText('LINESTRING(0 0,100 0)')),
        ('B','C',100.0, ST_GeomFromText('LINESTRING(100 0,100 100)')),
        ('C','D',100.0, ST_GeomFromText('LINESTRING(100 100,0 100)')),
        ('D','A',100.0, ST_GeomFromText('LINESTRING(0 100,0 0)'))
    ) AS t(start_node, end_node, length, geom);
"""

_GIAS_HEADER = (
    "urn,establishmentnumber,establishmentname,typeofestablishment_code,typeofestablishment_name,"
    "establishmentstatus_code,phaseofeducation_code,phaseofeducation_name,statutorylowage,statutoryhighage,"
    "schoolcapacity,postcode,urbanrural_code,districtadministrative_code,msoa_code,lsoa_code,easting,northing"
)


def _write_gias_csv(path: Path) -> None:
    rows = [
        "1,1,School A,1,Community,1,4,Secondary,11,18,500,LS1 1AA,A1,E08000035,E02,E01,10,10",
        "2,2,Closed School,1,Community,4,4,Secondary,11,18,500,LS1 1AB,A1,E08000035,E02,E01,50,50",  # status 4 → excluded
    ]
    path.write_text("\n".join([_GIAS_HEADER, *rows]) + "\n")


def test_walk_isochrones_is_convex_hull_of_reachable_nodes():
    # square graph; a school by node A reaches all four corners within the radius
    edges = pd.DataFrame({"start_node": ["A", "B", "C", "D"], "end_node": ["B", "C", "D", "A"], "length": [100.0] * 4})
    nodes = pd.DataFrame({"node": ["A", "B", "C", "D"], "x": [0, 100, 100, 0], "y": [0, 0, 100, 100]})
    schools = pd.DataFrame({"x": [10.0], "y": [10.0]})

    geoms = build_db._walk_isochrones(edges, nodes, schools, radius=1000)
    assert len(geoms) == 1
    assert geoms[0].area == 100 * 100  # convex hull of the four corners


def test_download_gias_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    cached = tmp_path / "edubasealldata20990101.csv"
    _write_gias_csv(cached)
    # a cached file is reused without hitting the network
    monkeypatch.setattr(build_db, "_download", lambda *a, **k: pytest.fail("should not download"))
    assert build_db._download_gias() == cached


def test_download_gias_downloads_dated_url(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    urls: list[str] = []

    def fake_download(url, path):
        urls.append(url)
        _write_gias_csv(path)

    monkeypatch.setattr(build_db, "_download", fake_download)
    today = date.today().strftime("%Y%m%d")
    result = build_db._download_gias()
    assert result == tmp_path / f"edubasealldata{today}.csv"
    assert urls == [f"https://ea-edubase-api-prod.azurewebsites.net/edubase/downloads/public/edubasealldata{today}.csv"]


def test_download_gias_falls_back_to_yesterday(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y%m%d")

    def fake_download(url, path):
        if yesterday not in url:
            raise requests.HTTPError("not published yet")
        _write_gias_csv(path)

    monkeypatch.setattr(build_db, "_download", fake_download)
    assert build_db._download_gias() == tmp_path / f"edubasealldata{yesterday}.csv"


def test_download_gias_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(build_db, "_download", MagicMock(side_effect=requests.ConnectionError("offline")))
    with pytest.raises(FileNotFoundError, match="Could not download the GIAS"):
        build_db._download_gias()


def test_schools_requires_open_roads(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    _write_gias_csv(tmp_path / "edubasealldata20990101.csv")
    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")
    with pytest.raises(RuntimeError, match="require the open_roads table"):
        build_db.load_schools(con)
    con.close()


def test_schools_builds_isochrones(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    _write_gias_csv(tmp_path / "edubasealldata20990101.csv")
    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")

    con.execute(_SQUARE_ROADS)
    build_db.load_schools(con)

    cols = {d[0] for d in con.execute("SELECT * FROM schools LIMIT 0").description}
    assert {"urn", "geom", "isochrone", "isochrone_area_km2"} <= cols
    # H3 cell ids (resolutions 8-11) derived from the school location
    assert {"h3_8_id", "h3_9_id", "h3_10_id", "h3_11_id"} <= cols
    assert con.execute("SELECT COUNT(*) FROM schools WHERE h3_9_id IS NULL").fetchone()[0] == 0  # ty:ignore[not-subscriptable]
    # the closed school (status 4) is filtered out
    assert con.execute("SELECT COUNT(*) FROM schools").fetchone()[0] == 1  # ty:ignore[not-subscriptable]
    # the isochrone is a polygon with positive area (the reachable square)
    assert con.execute("SELECT ST_GeometryType(isochrone) FROM schools").fetchone()[0] == "POLYGON"  # ty:ignore[not-subscriptable]
    assert con.execute("SELECT isochrone_area_km2 FROM schools").fetchone()[0] > 0  # ty:ignore[not-subscriptable]
    con.close()


# --- IMD (English Indices of Deprivation) ---


def _write_iod_csv(path: Path) -> None:
    import csv

    from scripts.build_db import IMD_COLUMNS

    # csv.writer quotes fields containing commas (some IoD column names contain a comma); seven
    # trailing values cover the seven domain score columns kept in IMD_COLUMNS
    n_scores = len(IMD_COLUMNS) - 5  # minus spatial_id, lad24cd, lad24nm, imd_score, imd_rank
    rows = [
        ["E01000001", "E09000001", "City of London", 5.0, 100, *([0.1] * n_scores)],
        ["E01000002", "E09000001", "City of London", 15.0, 50, *([0.2] * n_scores)],
        ["E01000003", "E09000001", "City of London", 25.0, 10, *([0.3] * n_scores)],
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(IMD_COLUMNS))  # original long IoD column names
        writer.writerows(rows)


def _write_wimd_ods(path: Path) -> None:
    # WIMD "Data" sheet: three preamble rows, then headers, then one row per LSOA. Higher score = more
    # deprived (same convention as the English IoD).
    wimd = pd.DataFrame(
        {
            "LSOA code": ["W01000001", "W01000002", "W01000003"],
            "LSOA name": ["A", "B", "C"],
            "Local Authority name": ["Cardiff", "Cardiff", "Swansea"],
            "WIMD 2025": [10.0, 20.0, 30.0],
            "Income": [1.0, 2.0, 3.0],
            "Employment": [1.0, 2.0, 3.0],
            "Education": [1.0, 2.0, 3.0],
            "Health": [1.0, 2.0, 3.0],
            "Community Safety": [1.0, 2.0, 3.0],
            "Physical Environment": [1.0, 2.0, 3.0],
            "Access to Services": [1.0, 2.0, 3.0],
            "Housing": [1.0, 2.0, 3.0],
        }
    )
    wimd.to_excel(path, sheet_name="Data", startrow=3, index=False, engine="odf")


def test_imd_england_downloads_when_missing(tmp_path, monkeypatch):
    # with no cached CSV present, the English loader fetches it via _download (here faked)
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(build_db, "_download", lambda url, path: _write_iod_csv(path))
    monkeypatch.setattr(
        build_db, "_imd_wales", lambda *a, **k: pd.DataFrame(columns=list(build_db.IMD_COLUMNS.values()))
    )
    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")
    build_db.load_imd(con)
    assert con.execute("SELECT COUNT(*) FROM imd_scores_pct").fetchone()[0] == 3  # ty:ignore[not-subscriptable]
    assert (tmp_path / data_source("imd")["csv"]).exists()  # the (faked) download was cached
    con.close()


def test_imd_loads_per_lsoa_percentiles(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    # isolate the English path; the Welsh merge is covered separately
    monkeypatch.setattr(
        build_db, "_imd_wales", lambda *a, **k: pd.DataFrame(columns=list(build_db.IMD_COLUMNS.values()))
    )
    _write_iod_csv(tmp_path / "File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv")
    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")

    build_db.load_imd(con)
    cols = {d[0] for d in con.execute("SELECT * FROM imd_scores_pct LIMIT 0").description}
    assert {"spatial_id", "imd_rank", "imd_score", "income", "crime"} <= cols
    # the English sub-domains are dropped (Wales has no equivalent)
    assert {"idac", "idaop", "cyp", "indoors", "outdoors"}.isdisjoint(cols)
    assert con.execute("SELECT COUNT(*) FROM imd_scores_pct").fetchone()[0] == 3  # ty:ignore[not-subscriptable]

    # scores become percentile ranks in (0, 1]; the three distinct imd_scores → 1/3, 2/3, 1
    scores = [r[0] for r in con.execute("SELECT imd_score FROM imd_scores_pct ORDER BY spatial_id").fetchall()]
    assert scores == pytest.approx([1 / 3, 2 / 3, 1.0])
    # imd_rank is passed through unchanged (not percentiled)
    ranks = [r[0] for r in con.execute("SELECT imd_rank FROM imd_scores_pct ORDER BY spatial_id").fetchall()]
    assert ranks == [100, 50, 10]
    con.close()


def test_imd_merges_england_and_wales(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    _write_iod_csv(tmp_path / "File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv")
    _write_wimd_ods(tmp_path / data_source("wimd")["ods"])
    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")

    build_db.load_imd(con)
    # 3 English + 3 Welsh LSOAs, one shared column set
    assert con.execute("SELECT COUNT(*) FROM imd_scores_pct").fetchone()[0] == 6  # ty:ignore[not-subscriptable]
    assert con.execute("SELECT COUNT(*) FROM imd_scores_pct WHERE spatial_id LIKE 'W%'").fetchone()[0] == 3  # ty:ignore[not-subscriptable]

    # Welsh percentiles are ranked within Wales (higher score = more deprived); most-deprived → imd_rank 1
    wales: tuple[float, ...] = con.execute(
        "SELECT imd_score, imd_rank, bhs FROM imd_scores_pct WHERE spatial_id = 'W01000003'"
    ).fetchone()
    assert wales[0] == pytest.approx(1.0)  # highest WIMD score in Wales → top percentile
    assert wales[1] == 1  # …and rank 1 (most deprived)
    assert wales[2] == pytest.approx(1.0)  # bhs = mean(Access, Housing) percentile, also highest
    con.close()
