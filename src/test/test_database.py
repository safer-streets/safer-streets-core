from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import geopandas as gpd
import pandas as pd
import pytest

from safer_streets_core.database import (
    add_table_from_shapefile,
    duckdb_spatial_connector,
    ephemeral_duckdb_spatial_connector,
    to_gdf,
)


class TestEphemeralDuckdbSpatialConnector:
    def test_returns_duckdb_connection(self):
        con = ephemeral_duckdb_spatial_connector()
        assert isinstance(con, duckdb.DuckDBPyConnection)
        con.close()

    def test_spatial_extension_loaded(self):
        con = ephemeral_duckdb_spatial_connector()
        result = con.execute("SELECT COUNT(*) FROM duckdb_functions() WHERE function_name='st_read';").fetchall()
        assert len(result) > 0
        con.close()

    def test_exception_closes_connection(self):
        with patch("duckdb.connect") as mock_connect:
            mock_con = MagicMock()
            mock_con.execute.side_effect = Exception("Test error")
            mock_connect.return_value = mock_con

            _ = ephemeral_duckdb_spatial_connector()
            mock_con.close.assert_called_once()


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


class TestDuckdbSpatialConnector:
    @patch("duckdb.connect")
    def test_context_manager_yields_connection(self, mock_connect):
        mock_con = MagicMock()
        mock_connect.return_value = mock_con

        with duckdb_spatial_connector("test.db") as con:
            assert con == mock_con

        mock_con.close.assert_called_once()

    @patch("duckdb.connect")
    def test_closes_connection_on_exception(self, mock_connect):
        mock_con = MagicMock()
        mock_connect.return_value = mock_con

        with pytest.raises(ValueError), duckdb_spatial_connector("test.db"):
            raise ValueError("Test error")

        mock_con.close.assert_called_once()

    @patch("duckdb.connect")
    def test_read_only_parameter(self, mock_connect):
        mock_con = MagicMock()
        mock_connect.return_value = mock_con

        with duckdb_spatial_connector("test.db", read_only=False):
            pass

        mock_connect.assert_called_with(database="test.db", read_only=False)


class TestToGdf:
    def test_converts_to_geodataframe(self):
        df = pd.DataFrame(
            {
                "col1": [1, 2],
                "col2": ["a", "b"],
                "wkt_geom": [
                    "POINT (500000 200000)",
                    "POINT (500001 200001)",
                ],
            }
        )

        gdf = to_gdf(df, "wkt_geom")

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

        gdf = to_gdf(df, "wkt_geom")

        assert gdf.geometry[0].x == 500000
        assert gdf.geometry[0].y == 200000
