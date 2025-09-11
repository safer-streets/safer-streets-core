from contextlib import contextmanager
from typing import Iterator

import duckdb
import geopandas as gpd
import pandas as pd

from safer_streets_core.utils import data_dir


@contextmanager
def duckdb_spatial_connector(dbname: str, *, read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
    con = duckdb.connect(database=data_dir() / dbname, read_only=read_only)
    try:
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
        yield con
    finally:
        con.close()


def to_gdf(df: pd.DataFrame, wkt_col: str) -> gpd.GeoDataFrame:
    """Coverts a pandas df output from duckdb with wkt format geometry. Assumes BNG CRS"""
    return gpd.GeoDataFrame(df.drop(columns=wkt_col), geometry=gpd.GeoSeries.from_wkt(df[wkt_col]), crs="EPSG:27700")
