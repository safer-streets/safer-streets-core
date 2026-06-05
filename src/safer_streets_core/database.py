import os
from pathlib import Path
from zipfile import ZipFile

import duckdb
import geopandas as gpd
import pandas as pd

from safer_streets_core.utils import data_dir


def _load_extensions(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    INSTALL spatial;LOAD spatial;
    INSTALL vss;LOAD vss;
    INSTALL h3 FROM community;LOAD h3;
    """)


def duckdb_connector(db: Path | None = None, *, writeable: bool = False) -> duckdb.DuckDBPyConnection:
    """Connect to a local DuckDB file, or in-memory if db is None."""
    con = duckdb.connect(
        database=str(db) if db else ":memory:",
        read_only=db is not None and not writeable,
    )
    try:
        _load_extensions(con)
        return con
    except Exception:
        con.close()
        raise


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


# # this is of little use for emphemeral (in-memory parquet)
# @contextmanager
# def duckdb_spatial_connector(dbname: str, *, read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
#     con = duckdb.connect(database=dbname, read_only=read_only)
#     try:
#         con.execute("INSTALL spatial;")
#         con.execute("LOAD spatial;")
#         yield con
#     finally:
#         con.close()


def to_gdf(df: pd.DataFrame, wkt_col: str) -> gpd.GeoDataFrame:
    """Coverts a pandas df output from duckdb with wkt format geometry. Assumes BNG CRS"""
    return gpd.GeoDataFrame(
        df.drop(columns=wkt_col),
        geometry=gpd.GeoSeries.from_wkt(df[wkt_col]),
        crs="EPSG:27700",
    )


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
