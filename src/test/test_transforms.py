import duckdb
import pytest

from safer_streets_core import transforms
from safer_streets_core.database import duckdb_connector

# H3 cell index string length for resolution 9 (canonical lowercase hex)
H3_RES9_STR_LEN = 15


def _make_db() -> duckdb.DuckDBPyConnection:
    """In-memory DB with a small crime_data table and one boundary table per geography."""
    try:
        con = duckdb_connector(writeable=True)
    except duckdb.HTTPException as e:  # extension download unavailable
        pytest.skip(f"extension download unavailable: {e}")

    # the real crime_data table comes from read_csv(normalize_names=true); the police.uk
    # "Month" column normalises to "_month" (month is a DuckDB keyword, so it is prefixed)
    con.execute("""
        CREATE TABLE crime_data AS
        SELECT * FROM (VALUES
            (-1.549, 53.801, 'Burglary', '2025-01'),
            (-1.549, 53.801, 'Burglary', '2025-01'),
            (-1.548, 53.802, 'Robbery',  '2025-01'),
            (-1.550, 53.800, 'Drugs',    '2025-02'),
            (NULL,   NULL,    'Drugs',    '2025-02')
        ) AS t(longitude, latitude, crime_type, _month);
    """)
    # one boundary per geography, each covering the crime locations (BNG geom)
    poly = "POLYGON((-1.6 53.75,-1.5 53.75,-1.5 53.85,-1.6 53.85,-1.6 53.75))"
    for table, code in [
        ("police_force_areas", "E23000010"),
        ("local_authority_districts", "E08000035"),
        ("msoa_2021", "E02002330"),
        ("lsoa_2021", "E01011264"),
        ("output_areas_2021", "E00056789"),
    ]:
        con.execute(f"""
            CREATE TABLE {table} AS
            SELECT '{code}' AS spatial_id,
                   ST_Transform(ST_GeomFromText('{poly}'), 'EPSG:4326', 'EPSG:27700', always_xy := true) AS geom;
        """)
    return con


class TestBuildCrimeCountsH3:
    def test_counts_are_grouped_and_nulls_excluded(self):
        con = _make_db()
        transforms.build_crime_counts_h3(con, resolutions=[9])

        rows = con.execute("SELECT COUNT(*) FROM crime_counts_h3_9").fetchone()[0]  # ty:ignore[not-subscriptable]
        assert rows > 0

        # the row with NULL coordinates is dropped, leaving 4 crimes
        total = con.execute("SELECT SUM(count) FROM crime_counts_h3_9").fetchone()[0]  # ty:ignore[not-subscriptable]
        assert total == 4

    def test_spatial_ids_are_valid_h3_strings(self):
        con = _make_db()
        transforms.build_crime_counts_h3(con, resolutions=[9])

        ids = [r[0] for r in con.execute("SELECT DISTINCT spatial_id FROM crime_counts_h3_9").fetchall()]
        assert ids
        for cell in ids:
            assert len(cell) == H3_RES9_STR_LEN
            assert cell == cell.lower()
            # round-trips through the h3 extension as a valid cell
            valid = con.execute("SELECT h3_is_valid_cell(?)", [cell]).fetchone()[0]  # ty:ignore[not-subscriptable]
            assert valid


class TestBuildH3Geogs:
    def test_geogs_have_all_codes_and_no_duplicates(self):
        con = _make_db()  # no open_greenspace table → no greenspace_ids column
        transforms.build_all(con, resolutions=[9])

        cols = [d[0] for d in con.execute("SELECT * FROM h3_9_geogs LIMIT 0").description]
        assert set(cols) == {"spatial_id", "pfa23", "lad24", "msoa21", "lsoa21", "oa21"}

        dupes = con.execute("SELECT spatial_id, COUNT(*) c FROM h3_9_geogs GROUP BY spatial_id HAVING c > 1").fetchall()
        assert dupes == []

        # every cell resolves to the (single) boundary code we created
        assert con.execute("SELECT COUNT(*) FROM h3_9_geogs WHERE lad24 = 'E08000035'").fetchone()[0] > 0  # ty:ignore[not-subscriptable]


class TestGreenspaceLookup:
    def test_greenspace_ids_folded_into_geogs(self):
        con = _make_db()
        # a greenspace polygon covering the crime locations (BNG), mirroring open_greenspace
        poly = "POLYGON((-1.6 53.75,-1.5 53.75,-1.5 53.85,-1.6 53.85,-1.6 53.75))"
        con.execute(f"""
            CREATE TABLE open_greenspace AS
            SELECT 'GS1' AS id, 'Public Park Or Garden' AS function,
                   ST_Transform(ST_GeomFromText('{poly}'), 'EPSG:4326', 'EPSG:27700', always_xy := true) AS geom;
        """)
        transforms.build_all(con, resolutions=[9])

        # the lookup view keeps one row per (cell, overlapping greenspace)
        assert con.execute("SELECT COUNT(*) FROM h3_9_greenspace_lookup").fetchone()[0] > 0  # ty:ignore[not-subscriptable]

        cols = [d[0] for d in con.execute("SELECT * FROM h3_9_geogs LIMIT 0").description]
        assert "greenspace_ids" in cols

        # every cell overlaps the single park, so greenspace_ids lists it; still one row per cell
        ids = con.execute("SELECT DISTINCT greenspace_ids FROM h3_9_geogs").fetchall()
        assert ids == [(["GS1"],)]
        dupes = con.execute("SELECT spatial_id, COUNT(*) c FROM h3_9_geogs GROUP BY spatial_id HAVING c > 1").fetchall()
        assert dupes == []

    def test_greenspace_skipped_when_table_absent(self):
        con = _make_db()  # no open_greenspace
        transforms.build_all(con, resolutions=[9])
        views = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE schema_name='main'").fetchall()}
        assert "h3_9_greenspace_lookup" not in views


class TestOverlapFeatures:
    @staticmethod
    def _add_polygon_table(con, table, cols_sql):
        poly = "POLYGON((-1.6 53.75,-1.5 53.75,-1.5 53.85,-1.6 53.85,-1.6 53.75))"
        con.execute(f"""
            CREATE TABLE {table} AS
            SELECT {cols_sql},
                   ST_Transform(ST_GeomFromText('{poly}'), 'EPSG:4326', 'EPSG:27700', always_xy := true) AS geom;
        """)

    def test_land_cover_ids_folded_into_geogs(self):
        con = _make_db()
        # land_cover mirrors the UKCEH LCM schema (gid, _mode)
        self._add_polygon_table(con, "land_cover", "42 AS gid, 20 AS _mode")
        transforms.build_all(con, resolutions=[9])

        lc = con.execute("SELECT DISTINCT mode FROM h3_9_land_cover_lookup").fetchall()
        assert lc == [(20,)]
        ids = con.execute("SELECT DISTINCT land_cover_ids FROM h3_9_geogs").fetchall()
        assert ids == [([42],)]

    def test_road_network_uses_length_and_road_prefix(self):
        con = _make_db()
        # road_network mirrors OS Open Roads (id, road_function); lines → ST_Length overlap.
        # this diagonal passes through the crime cluster (at lon -1.549 the line is at lat 53.801)
        line = "LINESTRING(-1.6 53.75, -1.5 53.85)"
        con.execute(f"""
            CREATE TABLE open_roads AS
            SELECT 'R1' AS id, 'Local Road' AS road_function,
                   ST_Transform(ST_GeomFromText('{line}'), 'EPSG:4326', 'EPSG:27700', always_xy := true) AS geom;
        """)
        transforms.build_all(con, resolutions=[9])

        # the lookup carries a road_ prefix and an overlap_length (not overlap_area)
        cols = {d[0] for d in con.execute("SELECT * FROM h3_9_road_network_lookup LIMIT 0").description}
        assert cols == {"spatial_id", "road_id", "type", "overlap_length"}
        # road_ids list is folded into geogs for the cells the line crosses
        geog_cols = [d[0] for d in con.execute("SELECT * FROM h3_9_geogs LIMIT 0").description]
        assert "road_ids" in geog_cols
        assert con.execute("SELECT COUNT(*) FROM h3_9_geogs WHERE road_ids IS NOT NULL").fetchone()[0] > 0  # ty:ignore[not-subscriptable]

    def test_all_features_folded_and_no_duplicate_cells(self):
        con = _make_db()
        self._add_polygon_table(con, "open_greenspace", "'GS1' AS id, 'Play Space' AS function")
        self._add_polygon_table(con, "land_cover", "42 AS gid, 20 AS _mode")
        self._add_polygon_table(con, "open_roads", "'R1' AS id, 'Local Road' AS road_function")
        transforms.build_all(con, resolutions=[9])

        cols = [d[0] for d in con.execute("SELECT * FROM h3_9_geogs LIMIT 0").description]
        assert {"greenspace_ids", "land_cover_ids", "road_ids"} <= set(cols)
        dupes = con.execute("SELECT spatial_id, COUNT(*) c FROM h3_9_geogs GROUP BY spatial_id HAVING c > 1").fetchall()
        assert dupes == []


class TestRetailCentreLookup:
    def test_nearest_retail_centre_folded_into_geogs(self):
        con = _make_db()
        # a retail centre covering the crime cells (BNG) → distance 0, one per cell
        poly = "POLYGON((-1.6 53.75,-1.5 53.75,-1.5 53.85,-1.6 53.85,-1.6 53.75))"
        con.execute(f"""
            CREATE TABLE retail_centres AS
            SELECT 'RC1' AS rc_id,
                   ST_Transform(ST_GeomFromText('{poly}'), 'EPSG:4326', 'EPSG:27700', always_xy := true) AS geom;
        """)
        transforms.build_all(con, resolutions=[9])

        # scalar nearest-centre columns (not a list)
        cols = [d[0] for d in con.execute("SELECT * FROM h3_9_geogs LIMIT 0").description]
        assert {"retail_centre_id", "retail_centre_distance"} <= set(cols)
        rows = con.execute("SELECT DISTINCT retail_centre_id, retail_centre_distance FROM h3_9_geogs").fetchall()
        assert rows == [("RC1", 0.0)]
        dupes = con.execute("SELECT spatial_id, COUNT(*) c FROM h3_9_geogs GROUP BY spatial_id HAVING c > 1").fetchall()
        assert dupes == []

    def test_skipped_when_table_absent(self):
        con = _make_db()
        transforms.build_all(con, resolutions=[9])
        cols = [d[0] for d in con.execute("SELECT * FROM h3_9_geogs LIMIT 0").description]
        assert "retail_centre_id" not in cols


class TestReplaceFlag:
    def test_replace_true_rebuilds_table(self):
        con = _make_db()
        transforms.build_crime_counts_h3(con, resolutions=[9])
        con.execute("DELETE FROM crime_data")  # so a rebuild would empty the table

        transforms.build_crime_counts_h3(con, resolutions=[9], replace=True)
        assert con.execute("SELECT COUNT(*) FROM crime_counts_h3_9").fetchone()[0] == 0  # ty:ignore[not-subscriptable]

    def test_replace_false_keeps_existing_table(self):
        con = _make_db()
        transforms.build_crime_counts_h3(con, resolutions=[9])
        before = con.execute("SELECT COUNT(*) FROM crime_counts_h3_9").fetchone()[0]  # ty:ignore[not-subscriptable]
        con.execute("DELETE FROM crime_data")  # a rebuild would change the result

        # IF NOT EXISTS: the existing table is left untouched
        transforms.build_crime_counts_h3(con, resolutions=[9], replace=False)
        assert con.execute("SELECT COUNT(*) FROM crime_counts_h3_9").fetchone()[0] == before  # ty:ignore[not-subscriptable]
