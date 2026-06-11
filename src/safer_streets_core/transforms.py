"""
H3 aggregation transforms for the build pipeline.

These functions operate on an open, writable DuckDB connection that already
contains:
  - a ``crime_data`` table (street-level crimes with latitude/longitude/crime_type/month)
  - one boundary table per ONS geography (each with a ``spatial_id`` code and a BNG
    ``geom`` column), as written by ``scripts.ons_boundaries.load_all``.

They build, for each H3 resolution:
  - ``crime_counts_h3_{res}``   counts grouped by H3 cell / crime type / month
  - ``h3_{res}_{key}_lookup``   a view mapping each H3 cell to one ONS geography code
  - ``h3_{res}_geogs``          one row per H3 cell with every ONS code

Ported from the ``duckdb-spatial`` prototype notebook (safer-streets-eda).
"""

import duckdb

H3_RESOLUTIONS = [8, 9, 10]

# short code -> boundary table name (the tables created by ons_boundaries.load_all).
# lad24 is listed for full-UK coverage; it is used as the base for h3_*_geogs.
GEOGRAPHY_MAPPINGS = {
    "pfa23": "police_force_areas",
    "lad24": "local_authority_districts",
    "msoa21": "msoa_2021",
    "lsoa21": "lsoa_2021",
    "oa21": "output_areas_2021",
}

# the geography used as the base table for h3_*_geogs (broadest coverage: incl. NI/Scotland)
_BASE_KEY = "lad24"


def _create(kind: str, name: str, *, replace: bool) -> str:
    """Build the leading CREATE clause for a table or view.

    replace=True  -> ``CREATE OR REPLACE {kind} {name}``    (always rebuilt)
    replace=False -> ``CREATE {kind} IF NOT EXISTS {name}`` (kept if it already exists)
    """
    return f"CREATE OR REPLACE {kind} {name}" if replace else f"CREATE {kind} IF NOT EXISTS {name}"


def build_crime_counts_h3(
    con: duckdb.DuckDBPyConnection,
    resolutions: list[int] = H3_RESOLUTIONS,
    *,
    replace: bool = True,
) -> None:
    """Create ``crime_counts_h3_{res}`` counting crimes per H3 cell / crime type / month.

    The H3 cell index is stored as its canonical lowercase-hex string in ``spatial_id``.
    """
    for res in resolutions:
        con.execute(f"""
            {_create("TABLE", f"crime_counts_h3_{res}", replace=replace)} AS
            SELECT
                lower(hex(h3_latlng_to_cell(latitude, longitude, {res}))) AS spatial_id,
                crime_type,
                month,
                COUNT(*) AS count
            FROM crime_data
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            GROUP BY spatial_id, crime_type, month;
        """)


def build_h3_geo_lookups(
    con: duckdb.DuckDBPyConnection,
    resolutions: list[int] = H3_RESOLUTIONS,
    mappings: dict[str, str] = GEOGRAPHY_MAPPINGS,
    *,
    replace: bool = True,
) -> None:
    """Create ``h3_{res}_{key}_lookup`` views mapping each H3 cell to one ONS geography code.

    The H3 cell boundary (WGS-84) is transformed to BNG and intersected with each boundary
    table. A cell may straddle several boundaries, so it is assigned to the one it overlaps
    most, guaranteeing a single row per cell.
    """
    for res in resolutions:
        for key, table in mappings.items():
            con.execute(f"""
                {_create("VIEW", f"h3_{res}_{key}_lookup", replace=replace)} AS
                SELECT c.spatial_id, b.spatial_id AS {key}
                FROM (
                    SELECT DISTINCT
                        spatial_id,
                        ST_Transform(
                            ST_GeomFromText(h3_cell_to_boundary_wkt(spatial_id)),
                            'EPSG:4326', 'EPSG:27700', always_xy := true
                        ) AS cell_geom
                    FROM crime_counts_h3_{res}
                ) c
                JOIN {table} b ON ST_Intersects(c.cell_geom, b.geom)
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY c.spatial_id
                    ORDER BY ST_Area(ST_Intersection(c.cell_geom, b.geom)) DESC
                ) = 1;
            """)


def build_h3_geogs(
    con: duckdb.DuckDBPyConnection,
    resolutions: list[int] = H3_RESOLUTIONS,
    mappings: dict[str, str] = GEOGRAPHY_MAPPINGS,
    *,
    replace: bool = True,
) -> None:
    """Create ``h3_{res}_geogs`` with one row per H3 cell carrying every ONS code.

    Built by LEFT JOINing the per-geography lookup views on ``spatial_id``, starting from
    the broadest-coverage geography so cells outside England & Wales are still retained.
    """
    base = _BASE_KEY if _BASE_KEY in mappings else next(iter(mappings))
    others = [key for key in mappings if key != base]

    for res in resolutions:
        select_cols = ", ".join([f"base.{base}", *(f"{key}.{key}" for key in others)])
        joins = "\n".join(f"LEFT JOIN h3_{res}_{key}_lookup {key} USING (spatial_id)" for key in others)
        con.execute(f"""
            {_create("TABLE", f"h3_{res}_geogs", replace=replace)} AS
            SELECT base.spatial_id, {select_cols}
            FROM h3_{res}_{base}_lookup base
            {joins};
        """)


def build_all(
    con: duckdb.DuckDBPyConnection,
    resolutions: list[int] = H3_RESOLUTIONS,
    mappings: dict[str, str] = GEOGRAPHY_MAPPINGS,
    *,
    replace: bool = True,
) -> None:
    """Run all three H3 transforms in dependency order.

    When ``replace`` is False, tables/views that already exist are left untouched
    (``CREATE ... IF NOT EXISTS``) rather than rebuilt (``CREATE OR REPLACE``).
    """
    build_crime_counts_h3(con, resolutions=resolutions, replace=replace)
    build_h3_geo_lookups(con, resolutions=resolutions, mappings=mappings, replace=replace)
    build_h3_geogs(con, resolutions=resolutions, mappings=mappings, replace=replace)
