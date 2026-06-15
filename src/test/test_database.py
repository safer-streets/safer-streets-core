from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import geopandas as gpd
import pandas as pd
import pytest

from safer_streets_core.database import (
    add_table_from_shapefile,
    duckdb_connector,
    duckdb_context,
    fix_force_names,
    get_gdf,
    index_geometry_tables,
    motherduck_connector,
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
