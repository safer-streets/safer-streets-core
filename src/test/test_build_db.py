import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile

import duckdb
import geopandas as gpd
import pandas as pd
import pytest
from shapely import LineString, Polygon

from safer_streets_core.database import duckdb_connector, index_geometry_tables
from scripts import build_db
from scripts.build_db import (
    GREENSPACE_ZIP,
    LAND_COVER_ZIP,
    ROADS_ZIP,
    load_greenspace,
    load_land_cover,
    load_roads,
)

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
    monkeypatch.setattr(build_db, "index_geometry_tables", lambda con: None)
    monkeypatch.setattr(build_db.transforms, "build_all", lambda con, **k: None)
    monkeypatch.setattr(build_db.os, "replace", lambda src, dst: None)

    build_db.build(db_path=db_path, replace=False)

    assert calls == ["land_cover", "retail_centres", "roads", "poi", "schools"]  # already-present skipped


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
        geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]), Polygon([(200, 200), (300, 200), (300, 300)])],
        crs="EPSG:27700",
    )
    gdf.to_file(tmp_path / build_db.RETAIL_CENTRES_GPKG, driver="GPKG")

    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")

    build_db.load_retail_centres(con)
    cols = {d[0] for d in con.execute("SELECT * FROM retail_centres LIMIT 0").description}
    assert {"rc_id", "rc_name", "classification", "geom"} <= cols  # columns lower-cased, geom indexable
    assert con.execute("SELECT COUNT(*) FROM retail_centres").fetchone()[0] == 2  # ty:ignore[not-subscriptable]

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


def test_schools_missing_gias_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(build_db, "data_dir", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="GIAS schools export not found"):
        build_db.load_schools(MagicMock())


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
    # the closed school (status 4) is filtered out
    assert con.execute("SELECT COUNT(*) FROM schools").fetchone()[0] == 1  # ty:ignore[not-subscriptable]
    # the isochrone is a polygon with positive area (the reachable square)
    assert con.execute("SELECT ST_GeometryType(isochrone) FROM schools").fetchone()[0] == "POLYGON"  # ty:ignore[not-subscriptable]
    assert con.execute("SELECT isochrone_area_km2 FROM schools").fetchone()[0] > 0  # ty:ignore[not-subscriptable]
    con.close()
