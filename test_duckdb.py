
import duckdb
import geopandas as gpd
import pandas as pd
from shapely import Point, wkb

from safer_streets_core.spatial import load_population_data
from safer_streets_core.utils import data_dir

def load() -> None:
    pop = load_population_data("West Yorkshire")
    print(pop)

    pop["wkb_geometry"] = pop.geometry.apply(lambda g: g.wkb)
    # srid = pop.crs.to_epsg()
    # pop["srid"] = srid
    pop = pop.drop("geometry", axis=1)

    con = duckdb.connect(database=data_dir() / "test.duckdb")
    try:
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")

        con.register("temp_population", pop)

        # Create final tables with GEOMETRY column using ST_GeomFromWKB
        # ST_GeomFromWKB(wkb_blob, srid_int)
        response = con.execute("""
            CREATE OR REPLACE TABLE population AS
            SELECT
                GEOGRAPHY_CODE,
                C2021_ETH_20_NAME,
                C2021_AGE_6_NAME,
                C_SEX_NAME,
                ST_GeomFromWKB(wkb_geometry) AS geometry
            FROM temp_population;
        """)
        print(response.fetchall())
    finally:
        con.close()


def query() -> None:
    con = duckdb.connect(database=data_dir() / "test.duckdb")
    try:
        # geometry is in geoparquet format which is not recognised by shapely
        con.execute("INSTALL spatial;LOAD spatial;")
        # simple query returning readable geometry
        response = con.execute("""
            SELECT
                ST_AsText(ST_Centroid(ST_Collect(list(geometry)))) AS centroid_wkt
            FROM population
            LIMIT 5
        """)
        result = gpd.GeoDataFrame(response.fetch_df())
        print(result)


        point = Point(432500, 422500)
        # count people within radius of a point, grouped by OA, computing the centroid of the group
        response = con.execute(f"""
            SELECT
            GEOGRAPHY_CODE,
                COUNT(GEOGRAPHY_CODE) AS count,
                ST_AsText(ST_Centroid(ST_Collect(list(geometry)))) AS wkt
                FROM population
            WHERE
                C_SEX_NAME = 'Male'
                -- AND ST_Within(geometry, ST_MakeEnvelope(432000, 433000, 421000, 422000))
            AND
                ST_Distance(geometry, ST_GeomFromText('{point.wkt}')) < 200
            GROUP BY GEOGRAPHY_CODE
            """)
        result = gpd.GeoDataFrame(response.fetch_df())
        # result.geometry = result.geometry.apply(lambda g: from_wkb(g))
        print(result)
    finally:
        con.close()


if __name__ == "__main__":
    # load()
    query()


