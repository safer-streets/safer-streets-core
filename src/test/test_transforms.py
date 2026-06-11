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

        rows = con.execute("SELECT COUNT(*) FROM crime_counts_h3_9").fetchone()[0]
        assert rows > 0

        # the row with NULL coordinates is dropped, leaving 4 crimes
        total = con.execute("SELECT SUM(count) FROM crime_counts_h3_9").fetchone()[0]
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
            valid = con.execute("SELECT h3_is_valid_cell(?)", [cell]).fetchone()[0]
            assert valid


class TestBuildH3Geogs:
    def test_geogs_have_all_codes_and_no_duplicates(self):
        con = _make_db()
        transforms.build_all(con, resolutions=[9])

        cols = [d[0] for d in con.execute("SELECT * FROM h3_9_geogs LIMIT 0").description]
        assert set(cols) == {"spatial_id", "pfa23", "lad24", "msoa21", "lsoa21", "oa21"}

        dupes = con.execute("SELECT spatial_id, COUNT(*) c FROM h3_9_geogs GROUP BY spatial_id HAVING c > 1").fetchall()
        assert dupes == []

        # every cell resolves to the (single) boundary code we created
        assert con.execute("SELECT COUNT(*) FROM h3_9_geogs WHERE lad24 = 'E08000035'").fetchone()[0] > 0


class TestReplaceFlag:
    def test_replace_true_rebuilds_table(self):
        con = _make_db()
        transforms.build_crime_counts_h3(con, resolutions=[9])
        con.execute("DELETE FROM crime_data")  # so a rebuild would empty the table

        transforms.build_crime_counts_h3(con, resolutions=[9], replace=True)
        assert con.execute("SELECT COUNT(*) FROM crime_counts_h3_9").fetchone()[0] == 0

    def test_replace_false_keeps_existing_table(self):
        con = _make_db()
        transforms.build_crime_counts_h3(con, resolutions=[9])
        before = con.execute("SELECT COUNT(*) FROM crime_counts_h3_9").fetchone()[0]
        con.execute("DELETE FROM crime_data")  # a rebuild would change the result

        # IF NOT EXISTS: the existing table is left untouched
        transforms.build_crime_counts_h3(con, resolutions=[9], replace=False)
        assert con.execute("SELECT COUNT(*) FROM crime_counts_h3_9").fetchone()[0] == before
