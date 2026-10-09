import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import geopandas as gpd
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

    def test_transform_is_always_xy(self):
        """Without geometry_always_xy, EPSG:4326 output follows the authority's lat/lon axis order."""
        try:
            con = duckdb_connector()
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        x, y = con.execute(
            "SELECT ST_X(g), ST_Y(g) FROM (SELECT ST_Transform(ST_Point(500000, 200000), 'EPSG:27700', 'EPSG:4326') g)"
        ).fetchone()  # ty:ignore[not-iterable]
        assert x == pytest.approx(-0.5547, abs=1e-4)
        assert y == pytest.approx(51.6899, abs=1e-4)
        con.close()


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


@pytest.fixture
def spatial_con() -> Generator[duckdb.DuckDBPyConnection]:
    try:
        con = duckdb_connector()
    except duckdb.HTTPException as e:
        pytest.skip(f"extension download unavailable: {e}")
    yield con
    con.close()


class TestGetGdf:
    def test_converts_geometry_and_keeps_other_columns(self, spatial_con):
        gdf = get_gdf(
            spatial_con,
            """
            SELECT * FROM (VALUES
                (1, 'a', ST_Point(500000, 200000)),
                (2, 'b', ST_Point(500001, 200001))
            ) AS v(id, name, geom)
            """,
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert list(gdf.columns) == ["id", "name", "geom"]
        assert gdf.geometry.name == "geom"
        assert gdf["name"].tolist() == ["a", "b"]
        assert [(p.x, p.y) for p in gdf.geometry] == [(500000, 200000), (500001, 200001)]

    def test_bare_geometry_defaults_to_bng(self, spatial_con):
        gdf = get_gdf(spatial_con, "SELECT ST_Point(500000, 200000) AS geom")
        assert gdf.crs is not None and gdf.crs.to_epsg() == 27700

    def test_crs_override_for_bare_geometry(self, spatial_con):
        gdf = get_gdf(spatial_con, "SELECT ST_Point(-1.5, 53.8) AS geom", crs="EPSG:4326")
        assert gdf.crs is not None and gdf.crs.to_epsg() == 4326

    def test_qualified_geometry_keeps_its_own_crs(self, spatial_con):
        """The column's CRS wins over the default — relabelling lon/lat as BNG would be silently wrong."""
        gdf = get_gdf(spatial_con, "SELECT ST_Point(-1.5, 53.8)::GEOMETRY('EPSG:4326') AS geom")
        assert gdf.crs is not None and gdf.crs.to_epsg() == 4326

    def test_empty_result(self, spatial_con):
        gdf = get_gdf(spatial_con, "SELECT 1 AS id, ST_Point(0, 0) AS geom WHERE false")
        assert gdf.empty
        assert gdf.geometry.name == "geom"
        assert gdf.crs is not None and gdf.crs.to_epsg() == 27700

    def test_null_geometry_preserved(self, spatial_con):
        gdf = get_gdf(spatial_con, "SELECT * FROM (VALUES (1, ST_Point(0, 0)), (2, NULL)) AS v(id, geom)")
        assert len(gdf) == 2
        assert gdf.geometry.isna().tolist() == [False, True]

    def test_query_kwargs_passed_through(self, spatial_con):
        gdf = get_gdf(
            spatial_con,
            "SELECT * FROM (VALUES (1, ST_Point(0, 0)), (2, ST_Point(1, 1))) AS v(id, geom) WHERE id = $id",
            params={"id": 2},
        )
        assert gdf["id"].tolist() == [2]

    def test_raises_without_geometry_column(self, spatial_con):
        with pytest.raises(ValueError, match="No geometry column"):
            get_gdf(spatial_con, "SELECT 1 AS id")


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

    def test_null_geometry_is_not_a_geometry_type(self, tmp_path):
        """A NULL geometry contributes no type — crime_data keeps the crimes police.uk gives no
        coordinates for, and ST_GeometryType returns NULL for each of them."""
        try:
            con = duckdb_connector(writeable=True)
        except duckdb.HTTPException as e:
            pytest.skip(f"extension download unavailable: {e}")

        con.execute("""
            CREATE TABLE src AS
            SELECT ST_Point(400000, 100000) AS geom UNION ALL SELECT NULL;
        """)
        out = tmp_path / "nulls.parquet"
        write_geoparquet(con, "SELECT * FROM src", out)

        geo = json.loads(pq.read_schema(out).metadata[b"geo"])
        assert geo["columns"]["geom"]["geometry_types"] == ["Point"]
        assert geo["columns"]["geom"]["bbox"] == [400000, 100000, 400000, 100000]
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

    def test_bare_geometry_labelled_bng(self, spatial_con):
        spatial_con.execute("CREATE TABLE boundaries AS SELECT ST_Point(0, 0) AS geom UNION ALL SELECT NULL;")
        index_geometry_tables(spatial_con)
        assert spatial_con.execute("SELECT DISTINCT typeof(geom) FROM boundaries").fetchall() == [
            ("GEOMETRY('EPSG:27700')",)
        ]

    def test_qualified_geometry_keeps_its_crs(self, spatial_con):
        spatial_con.execute("CREATE TABLE wgs84 AS SELECT ST_Point(-1.5, 53.8)::GEOMETRY('EPSG:4326') AS geom;")
        index_geometry_tables(spatial_con)
        assert spatial_con.execute("SELECT typeof(geom) FROM wgs84").fetchone() == ("GEOMETRY('EPSG:4326')",)
        indexes = {r[0] for r in spatial_con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
        assert "wgs84_geom_rtree" in indexes

    def test_relabels_bare_column_already_indexed(self, spatial_con):
        """A database indexed before columns were labelled has an RTree blocking the type change."""
        spatial_con.execute("CREATE TABLE boundaries AS SELECT ST_Point(0, 0) AS geom;")
        spatial_con.execute('CREATE INDEX "boundaries_geom_rtree" ON boundaries USING RTREE (geom);')
        index_geometry_tables(spatial_con)
        assert spatial_con.execute("SELECT typeof(geom) FROM boundaries").fetchone() == ("GEOMETRY('EPSG:27700')",)
        indexes = {r[0] for r in spatial_con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
        assert "boundaries_geom_rtree" in indexes

    def test_labelled_tables_refuse_mixed_crs(self, spatial_con):
        spatial_con.execute("CREATE TABLE boundaries AS SELECT ST_Point(500000, 200000) AS geom;")
        spatial_con.execute("CREATE TABLE wgs84 AS SELECT ST_Point(-1.5, 53.8)::GEOMETRY('EPSG:4326') AS geom;")
        index_geometry_tables(spatial_con)
        with pytest.raises(duckdb.BinderException, match="different coordinate reference systems"):
            spatial_con.execute("SELECT ST_Intersects(b.geom, w.geom) FROM boundaries b, wgs84 w")

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
