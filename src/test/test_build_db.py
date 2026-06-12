import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile

import duckdb
import geopandas as gpd
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
    monkeypatch.setattr(build_db, "load_roads", lambda *a, **k: calls.append("roads"))
    monkeypatch.setattr(build_db, "index_geometry_tables", lambda con: None)
    monkeypatch.setattr(build_db.transforms, "build_all", lambda con, **k: None)
    monkeypatch.setattr(build_db.os, "replace", lambda src, dst: None)

    build_db.build(db_path=db_path, replace=False)

    assert calls == ["land_cover", "roads"]  # the three already-present stages were skipped
