import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

import duckdb
import geopandas as gpd

from safer_streets_core.utils import data_dir


def _load_extensions(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    INSTALL spatial;LOAD spatial;
    INSTALL vss;LOAD vss;
    INSTALL h3 FROM community;LOAD h3;
    """)


def duckdb_connector(db: Path | None = None, *, writeable: bool = False) -> duckdb.DuckDBPyConnection:
    """Connect to a local DuckDB file, or in-memory if db is None."""
    con = duckdb.connect(database=str(db) if db else ":memory:", read_only=db is not None and not writeable)
    try:
        _load_extensions(con)
        return con
    except Exception:
        con.close()
        raise


@contextmanager
def duckdb_context(db: Path | None = None, *, writeable: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Context managed DB connection.
    Limited usefulness for in-memory databases as DB will exist only within the context
    """
    con = duckdb.connect(database=str(db) if db else ":memory:", read_only=db is not None and not writeable)
    try:
        _load_extensions(con)
        yield con
    finally:
        con.close()


def motherduck_connector(db: str, *, writeable: bool = False) -> duckdb.DuckDBPyConnection:
    """Connect to a MotherDuck database. Uses MOTHERDUCK_TOKEN_RW when writeable, else MOTHERDUCK_TOKEN."""
    token_var = "MOTHERDUCK_TOKEN_RW" if writeable else "MOTHERDUCK_TOKEN"
    token = os.environ.get(token_var)
    if not token:
        raise OSError(f"{token_var} not set")
    con = duckdb.connect(database=f"md:{db}?motherduck_token={token}")
    try:
        con.execute("INSTALL spatial;LOAD spatial;")
        return con
    except Exception:
        con.close()
        raise


def add_table_from_shapefile(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: str | list[str],
    zipfile: str,
    shapefile: str | None = None,
    *,
    exists_ok: bool = False,
) -> None:
    if not shapefile:
        with ZipFile(data_dir() / zipfile) as z:
            shp_files = [p for p in z.namelist() if p.lower().endswith(".shp")]
            if not shp_files:
                raise FileNotFoundError(f"No .shp files found in {zipfile}")
            if len(shp_files) > 1:
                raise ValueError(f"Multiple .shp files found in {zipfile}: {shp_files}")
            shapefile = shp_files[0]

    shapefile_path = f"/vsizip/{data_dir() / zipfile}/{shapefile}"
    if isinstance(columns, list):
        columns = ", ".join(columns)
    con.execute(
        f"""CREATE TABLE {"IF NOT EXISTS" if exists_ok else ""} {table_name} AS
        SELECT {columns}, geom AS geometry FROM ST_Read('{shapefile_path}');
        """
    )


def get_gdf(
    con: duckdb.DuckDBPyConnection, query: str, *, wkt_col: str = "wkt", crs="EPSG:27700", **kwargs
) -> gpd.GeoDataFrame:
    """
    Runs a query returning a dataframe and converts to GeoDataFrame
    SQL should generate a geometry column matching wkt_col
    By default assumes BNG CRS
    """
    df = con.sql(query, **kwargs).df()
    return gpd.GeoDataFrame(df.drop(columns=wkt_col), geometry=gpd.GeoSeries.from_wkt(df[wkt_col]), crs=crs)


def fix_force_names(con, table: str, column: str) -> None:
    con.execute(f"""
    UPDATE {table}
    SET {column} =
        CASE {column}
            WHEN 'Metropolitan Police' THEN 'Metropolitan'
            WHEN 'Devon &amp; Cornwall' THEN 'Devon and Cornwall'
            WHEN 'London, City of' THEN 'City of London'
            WHEN 'Dyfed-Powys' THEN 'Dyfed Powys'
            ELSE {column}
        END;
    """)
