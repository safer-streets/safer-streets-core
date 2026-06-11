from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import geopandas as gpd
import pytest
from shapely import Polygon

from safer_streets_core.database import duckdb_connector, index_geometry_tables
from scripts.build_db import load_greenspace


def _write_greenspace_shp(directory: Path) -> Path:
    """Write a tiny BNG greenspace shapefile mimicking GB_GreenspaceSite."""
    gdf = gpd.GeoDataFrame(
        {
            "id": ["G1", "G2"],
            "function": ["Public Park Or Garden", "Play Space"],
            "distName1": ["Hyde Park", "The Rec"],
        },
        geometry=[
            Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            Polygon([(200, 200), (300, 200), (300, 300), (200, 300)]),
        ],
        crs="EPSG:27700",
    )
    shp = directory / "greenspace.shp"
    gdf.to_file(shp)
    return shp


def test_missing_shapefile_raises():
    # the missing-file check happens before the connection is touched, so a mock con suffices
    with pytest.raises(FileNotFoundError, match="OS Open Greenspace shapefile not found"):
        load_greenspace(MagicMock(), shapefile=Path("/nonexistent/nope.shp"))


def test_loads_table_with_indexable_geom(tmp_path):
    shp = _write_greenspace_shp(tmp_path)
    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:  # spatial extension download unavailable
        pytest.skip(f"extension download unavailable: {e}")

    load_greenspace(con, shapefile=shp)

    cols = [d[0] for d in con.execute("SELECT * FROM open_greenspace LIMIT 0").description]
    assert "geom" in cols  # ST_Read names the geometry column 'geom', so it gets indexed
    assert con.execute("SELECT COUNT(*) FROM open_greenspace").fetchone()[0] == 2  # ty:ignore[not-subscriptable]

    # the geometry step picks it up and builds an RTree index. ST_Read gives a CRS-qualified
    # GEOMETRY type, which must be normalised to bare GEOMETRY first; calling twice must be a no-op.
    index_geometry_tables(con)
    index_geometry_tables(con)
    indexes = {r[0] for r in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "open_greenspace_geom_rtree" in indexes
    con.close()
