from typing import Iterator

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import Point

from safer_streets_core.database import duckdb_spatial_connector, to_gdf
from safer_streets_core.spatial import get_force_boundary, get_street_network, load_population_data
from safer_streets_core.utils import data_dir


def load() -> None:
    pop = load_population_data("West Yorkshire")
    print(pop)

    pop["wkb_geometry"] = pop.geometry.apply(lambda g: g.wkb)
    # srid = pop.crs.to_epsg()
    # pop["srid"] = srid
    pop = pop.drop("geometry", axis=1)

    with duckdb_spatial_connector("test.duckdb", read_only=False) as con:
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


def load_streets() -> None:
    boundary = get_force_boundary("West Yorkshire")
    streets = get_street_network(boundary).reset_index()

    streets["wkb_geometry"] = streets.geometry.apply(lambda g: g.wkb)
    # srid = pop.crs.to_epsg()
    # pop["srid"] = srid
    streets = streets.drop("geometry", axis=1)

    with duckdb_spatial_connector("test.duckdb", read_only=False) as con:
        con.register("temp_streets", streets)

        query = f"""
            CREATE OR REPLACE TABLE streets AS
            SELECT
                {",\n".join(col for col in streets.columns if col != "wkb_geometry")},
                ST_GeomFromWKB(wkb_geometry) AS geometry
            FROM temp_streets;
        """
        response = con.execute(query)
        print(response.fetchall())


def query() -> None:
    with duckdb_spatial_connector("test.duckdb") as con:
        # geometry is in geoparquet format which is not recognised by shapely
        # simple query returning readable geometry
        response = con.execute("""
            SELECT
                ST_AsText(ST_Centroid(ST_Collect(list(geometry)))) AS centroid_wkt
            FROM population
            LIMIT 5
        """)
        result = to_gdf(response.fetch_df(), "centroid_wkt")
        print(result.area)

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
        result = to_gdf(response.fetch_df(), "wkt")
        print(result)


def query_streets() -> gpd.GeoDataFrame:
    with duckdb_spatial_connector("test.duckdb") as con:
        # geometry is in geoparquet format which is not recognised by shapely
        # simple query returning readable geometry
        point = Point(412000, 448500)
        response = con.execute(f"""
            SELECT
                u, v, key, osmid, name, highway, length, width, maxspeed, ST_AsText(geometry) AS geometry
            FROM streets
            WHERE
                ST_Distance(geometry, ST_GeomFromText('{point.wkt}')) < 5000
        """)
        # response = con.execute("SELECT DISTINCT(maxspeed) FROM streets")
        result = response.fetchdf()
    return to_gdf(result, "geometry")



if __name__ == "__main__":
    # load()
    # load_streets()

    query()
    df = query_streets()

    print(df.name) #[["max_speed"]]) #gpd.GeoSeries.from_wkt(df.geometry))
    ax = df.plot(figsize=(10,10))
    ax.set_axis_off()
    plt.show()
