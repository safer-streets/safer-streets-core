import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest

from safer_streets_core.database import (
    add_table_from_shapefile,
    duckdb_connector,
    duckdb_context,
    fix_force_names,
    get_gdf,
    index_geometry_tables,
    motherduck_connector,
    read_geoparquet,
    write_geoparquet,
)


class TestDuckdbConnector:
    @patch("safer_streets_core.database._load_extensions")
    def test_in_memory_by_default(self, mock_load):
        con = duckdb_connector()
        assert isinstance(con, duckdb.DuckDBPyConnection)
        mock_load.assert_called_once_with(con)
        con.close()

    @patch("safer_streets_core.database._load_extensions")
    @patch("safer_streets_core.database.duckdb.connect")
    def test_file_db_is_read_only_by_default(self, mock_connect, mock_load):
        mock_connect.return_value = MagicMock()
        db = Path("/tmp/some.db")
        duckdb_connector(db)
        mock_connect.assert_called_once_with(database=str(db), read_only=True)

    @patch("safer_streets_core.database._load_extensions")
    @patch("safer_streets_core.database.duckdb.connect")
    def test_file_db_writeable(self, mock_connect, mock_load):
        mock_connect.return_value = MagicMock()
        db = Path("/tmp/some.db")
        duckdb_connector(db, writeable=True)
        mock_connect.assert_called_once_with(database=str(db), read_only=False)

    @patch("safer_streets_core.database._load_extensions")
    @patch("safer_streets_core.database.duckdb.connect")
    def test_exception_closes_connection(self, mock_connect, mock_load):
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        mock_load.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            duckdb_connector()
        mock_con.close.assert_called_once()

    def test_spatial_extension_loaded(self):
        """Integration test: requires network to install the spatial extension."""
        try:
            with duckdb_context() as con:
                # duckdb stores the function name as 'ST_Read', so match case-insensitively
                result = con.execute(
                    "SELECT COUNT(*) FROM duckdb_functions() WHERE function_name ILIKE 'st_read';"
                ).fetchall()
                assert result[0][0] > 0
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")


class TestDuckdbContext:
    @patch("safer_streets_core.database._load_extensions")
    def test_yields_connection_and_closes(self, mock_load):
        with duckdb_context() as con:
            assert isinstance(con, duckdb.DuckDBPyConnection)
            mock_load.assert_called_once_with(con)
        # the connection is closed on context exit
        with pytest.raises(duckdb.ConnectionException):
            con.execute("SELECT 1")

    @patch("safer_streets_core.database._load_extensions")
    @patch("safer_streets_core.database.duckdb.connect")
    def test_closes_on_exception(self, mock_connect, mock_load):
        mock_con = MagicMock()
        mock_connect.return_value = mock_con

        with pytest.raises(RuntimeError), duckdb_context():
            raise RuntimeError("boom")
        mock_con.close.assert_called_once()


class TestMotherduckConnector:
    def test_raises_when_token_missing(self, monkeypatch):
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        with pytest.raises(OSError, match="MOTHERDUCK_TOKEN not set"):
            motherduck_connector("mydb")

    def test_raises_when_rw_token_missing(self, monkeypatch):
        monkeypatch.delenv("MOTHERDUCK_TOKEN_RW", raising=False)
        with pytest.raises(OSError, match="MOTHERDUCK_TOKEN_RW not set"):
            motherduck_connector("mydb", writeable=True)


class TestAddTableFromShapefile:
    @patch("safer_streets_core.database.ZipFile")
    @patch("safer_streets_core.database.data_dir")
    def test_finds_shapefile_in_zip(self, mock_data_dir, mock_zipfile):
        mock_data_dir.return_value = Path("/data")
        mock_zip_instance = MagicMock()
        mock_zip_instance.namelist.return_value = ["test.shp", "test.shx"]
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

        mock_con = MagicMock()
        add_table_from_shapefile(mock_con, "test_table", "col1, col2", "test.zip")

        mock_con.execute.assert_called_once()

    @patch("safer_streets_core.database.ZipFile")
    @patch("safer_streets_core.database.data_dir")
    def test_raises_error_no_shapefiles(self, mock_data_dir, mock_zipfile):
        mock_data_dir.return_value = Path("/data")
        mock_zip_instance = MagicMock()
        mock_zip_instance.namelist.return_value = ["test.txt"]
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

        mock_con = MagicMock()
        with pytest.raises(FileNotFoundError):
            add_table_from_shapefile(mock_con, "test_table", "col1", "test.zip")

    @patch("safer_streets_core.database.ZipFile")
    @patch("safer_streets_core.database.data_dir")
    def test_raises_error_multiple_shapefiles(self, mock_data_dir, mock_zipfile):
        mock_data_dir.return_value = Path("/data")
        mock_zip_instance = MagicMock()
        mock_zip_instance.namelist.return_value = ["test1.shp", "test2.shp"]
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

        mock_con = MagicMock()
        with pytest.raises(ValueError):
            add_table_from_shapefile(mock_con, "test_table", "col1", "test.zip")

    @patch("safer_streets_core.database.data_dir")
    def test_with_list_columns(self, mock_data_dir):
        mock_data_dir.return_value = Path("/data")
        mock_con = MagicMock()

        add_table_from_shapefile(mock_con, "test_table", ["col1", "col2"], "test.zip", "test.shp")

        call_args = mock_con.execute.call_args[0][0]
        assert "col1, col2" in call_args

    @patch("safer_streets_core.database.data_dir")
    def test_exists_ok_parameter(self, mock_data_dir):
        mock_data_dir.return_value = Path("/data")
        mock_con = MagicMock()

        add_table_from_shapefile(mock_con, "test_table", "col1", "test.zip", "test.shp", exists_ok=True)

        call_args = mock_con.execute.call_args[0][0]
        assert "IF NOT EXISTS" in call_args


class TestToGdf:
    def test_converts_to_geodataframe(self):
        df = pd.DataFrame(
            {
                "col1": [1, 2],
                "col2": ["a", "b"],
                "wkt": [
                    "POINT (500000 200000)",
                    "POINT (500001 200001)",
                ],
            }
        )
        with duckdb_context() as con:
            con.register("data", df)
            gdf = get_gdf(con, "SELECT * FROM data")

            assert isinstance(gdf, gpd.GeoDataFrame)
            assert gdf.crs == "EPSG:27700"
            assert len(gdf) == 2
            assert "wkt_geom" not in gdf.columns
            assert "col1" in gdf.columns
            assert "col2" in gdf.columns

    def test_geometry_correctly_parsed(self):
        df = pd.DataFrame(
            {
                "id": [1],
                "wkt_geom": ["POINT (500000 200000)"],
            }
        )

        with duckdb_context() as con:
            con.register("data", df)
            gdf = get_gdf(con, "SELECT * FROM data", wkt_col="wkt_geom")

            assert gdf.geometry[0].x == 500000
            assert gdf.geometry[0].y == 200000


class TestGeoparquetRoundTrip:
    def test_write_then_read_preserves_geometry(self, tmp_path):
        try:
            con = duckdb_connector(writeable=True)
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        con.execute("CREATE TABLE src AS SELECT 1 AS id, ST_Point(500000, 200000) AS geom;")
        out = tmp_path / "out.parquet"
        write_geoparquet(con, "SELECT * FROM src", out)
        assert out.exists()

        con.execute(f"CREATE TABLE back AS {read_geoparquet(out)}")
        row = con.execute("SELECT id, ST_X(geom), ST_Y(geom) FROM back").fetchone()
        assert row == (1, 500000, 200000)
        con.close()

    def test_write_is_atomic_no_tmp_left_behind(self, tmp_path):
        try:
            con = duckdb_connector(writeable=True)
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        out = tmp_path / "out.parquet"
        write_geoparquet(con, "SELECT 1 AS x", out)
        assert out.exists()
        assert not out.with_suffix(out.suffix + ".tmp").exists()
        con.close()

    def test_read_geoparquet_sql(self):
        path = Path("/data/x.parquet")
        assert read_geoparquet(path) == f"SELECT * FROM read_parquet('{path}')"

    def test_geopandas_reads_back_bng_not_crs84(self, tmp_path):
        """The written file names its CRS; without it geopandas assumes OGC:CRS84 and mislabels BNG."""
        try:
            con = duckdb_connector(writeable=True)
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        con.execute("CREATE TABLE src AS SELECT 1 AS id, ST_Point(500000, 200000) AS geom;")
        out = tmp_path / "out.parquet"
        write_geoparquet(con, "SELECT * FROM src", out)

        gdf = gpd.read_parquet(out)
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 27700
        assert (gdf.geometry[0].x, gdf.geometry[0].y) == (500000, 200000)
        con.close()

    def test_crs_override_is_honoured(self, tmp_path):
        try:
            con = duckdb_connector(writeable=True)
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        con.execute("CREATE TABLE src AS SELECT ST_Point(-1.5, 53.8) AS geom;")
        out = tmp_path / "wgs84.parquet"
        write_geoparquet(con, "SELECT * FROM src", out, crs="EPSG:4326")
        crs = gpd.read_parquet(out).crs
        assert crs is not None and crs.to_epsg() == 4326
        con.close()

    def test_geo_metadata_records_types_and_bbox(self, tmp_path):
        """geometry_types uses the spec's casing, not DuckDB's, and the bbox survives KV_METADATA."""
        try:
            con = duckdb_connector(writeable=True)
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        con.execute("""
            CREATE TABLE src AS
            SELECT ST_Point(400000, 100000) AS geom UNION ALL SELECT ST_Point(500000, 200000);
        """)
        out = tmp_path / "out.parquet"
        write_geoparquet(con, "SELECT * FROM src", out)

        geo = json.loads(pq.read_schema(out).metadata[b"geo"])
        assert geo["primary_column"] == "geom"
        assert geo["columns"]["geom"]["geometry_types"] == ["Point"]
        assert geo["columns"]["geom"]["bbox"] == [400000, 100000, 500000, 200000]
        con.close()

    def test_empty_geometry_table_omits_bbox(self, tmp_path):
        """An empty result has no extent; the bbox key is dropped rather than written as nulls."""
        try:
            con = duckdb_connector(writeable=True)
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        con.execute("CREATE TABLE src AS SELECT ST_Point(1, 2) AS geom WHERE false;")
        out = tmp_path / "empty.parquet"
        write_geoparquet(con, "SELECT * FROM src", out)

        geo = json.loads(pq.read_schema(out).metadata[b"geo"])
        assert "bbox" not in geo["columns"]["geom"]
        assert geo["columns"]["geom"]["geometry_types"] == []
        crs = gpd.read_parquet(out).crs
        assert crs is not None and crs.to_epsg() == 27700
        con.close()

    def test_every_geometry_column_is_described(self, tmp_path):
        """schools carries both geom and isochrone; a column missing from the blob reads back as BLOB."""
        try:
            con = duckdb_connector(writeable=True)
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        con.execute("""
            CREATE TABLE src AS
            SELECT 1 AS urn, ST_Point(400000, 100000) AS geom,
                   ST_Buffer(ST_Point(400000, 100000), 10) AS isochrone;
        """)
        out = tmp_path / "two.parquet"
        write_geoparquet(con, "SELECT * FROM src", out)

        geo = json.loads(pq.read_schema(out).metadata[b"geo"])
        assert set(geo["columns"]) == {"geom", "isochrone"}
        assert geo["primary_column"] == "geom"

        con.execute(f"CREATE TABLE back AS {read_geoparquet(out)}")
        types = con.execute("SELECT ST_GeometryType(geom), ST_GeometryType(isochrone) FROM back").fetchone()
        assert types == ("POINT", "POLYGON")
        con.close()

    def test_non_geometry_query_writes_plain_parquet(self, tmp_path):
        """No GEOMETRY column means no geo metadata at all — not an empty or bogus one."""
        try:
            con = duckdb_connector(writeable=True)
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        out = tmp_path / "plain.parquet"
        write_geoparquet(con, "SELECT 1 AS x", out)
        assert b"geo" not in (pq.read_schema(out).metadata or {})
        con.close()


class TestIndexGeometryTables:
    def test_repairs_invalid_and_creates_rtree_index(self):
        try:
            con = duckdb_connector(writeable=True)
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        # a self-intersecting ("bowtie") polygon is invalid; plus a valid one and a NULL
        con.execute("""
            CREATE TABLE boundaries AS SELECT * FROM (VALUES
                (1, ST_GeomFromText('POLYGON((0 0,2 0,2 2,0 2,0 0))')),
                (2, ST_GeomFromText('POLYGON((0 0,2 2,2 0,0 2,0 0))')),
                (3, CAST(NULL AS GEOMETRY))
            ) AS v(id, geom);
        """)
        # a table without a geom column should be ignored
        con.execute("CREATE TABLE other AS SELECT 1 AS x;")

        assert con.execute("SELECT COUNT(*) FROM boundaries WHERE NOT ST_IsValid(geom)").fetchone()[0] == 1  # ty:ignore[not-subscriptable]

        index_geometry_tables(con)

        assert (
            con.execute("SELECT COUNT(*) FROM boundaries WHERE geom IS NOT NULL AND NOT ST_IsValid(geom)").fetchone()[0]  # ty:ignore[not-subscriptable]
            == 0
        )
        indexes = {r[0] for r in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
        assert "boundaries_geom_rtree" in indexes
        assert "other_geom_rtree" not in indexes
        con.close()

    def test_idempotent(self):
        try:
            con = duckdb_connector(writeable=True)
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        con.execute("CREATE TABLE boundaries AS SELECT 1 AS id, ST_Point(0, 0) AS geom;")
        index_geometry_tables(con)
        index_geometry_tables(con)  # second call must not raise (CREATE INDEX IF NOT EXISTS)
        con.close()

    def test_crime_data_excluded_by_default(self):
        try:
            con = duckdb_connector(writeable=True)
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        con.execute("CREATE TABLE crime_data AS SELECT 1 AS id, ST_Point(0, 0) AS geom;")
        con.execute("CREATE TABLE boundaries AS SELECT 1 AS id, ST_Point(0, 0) AS geom;")
        index_geometry_tables(con)

        indexes = {r[0] for r in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
        assert "boundaries_geom_rtree" in indexes
        assert "crime_data_geom_rtree" not in indexes
        con.close()


class TestFixForceNames:
    def test_builds_expected_case_statement(self):
        mock_con = MagicMock()
        fix_force_names(mock_con, "crimes", "force")

        sql = mock_con.execute.call_args[0][0]
        assert "UPDATE crimes" in sql
        assert "'Metropolitan Police' THEN 'Metropolitan'" in sql
        assert "'Devon &amp; Cornwall' THEN 'Devon and Cornwall'" in sql
        assert "'London, City of' THEN 'City of London'" in sql
        assert "'Dyfed-Powys' THEN 'Dyfed Powys'" in sql
