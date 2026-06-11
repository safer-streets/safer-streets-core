import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile

import duckdb
import geopandas as gpd
import pytest
from shapely import Polygon

from safer_streets_core.database import duckdb_connector, index_geometry_tables
from scripts import build_db
from scripts.build_db import GREENSPACE_ZIP, load_greenspace

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

    def fake_download(zip_path: Path) -> None:
        called["n"] += 1
        _make_greenspace_zip(zip_path)

    monkeypatch.setattr(build_db, "_download_greenspace", fake_download)

    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")

    load_greenspace(con)
    assert called["n"] == 1  # download triggered because the zip was absent
    assert con.execute("SELECT COUNT(*) FROM open_greenspace").fetchone()[0] == 2
    con.close()
